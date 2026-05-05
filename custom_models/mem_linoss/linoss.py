import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from fla.modules.l2norm import l2norm
from .kernel.linoss_kernel import fused_chunk_linoss

class LinOSS(nn.Module):
    def __init__(self, num_heads: int, 
                 head_dim: int, 
                 delta_t: float, 
                 damping: bool, 
                 grad_clip_scale: float,
                 y_init: str,
                 monitor: bool = True):

        super().__init__()
        self.delta_t = delta_t
        self.damping = damping
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.grad_clip_scale = grad_clip_scale
        self.y_init = y_init
        self.monitor = monitor  # Set to False for torch.compile compatibility

        # Learnable initial state for y (MAML-style: learn the best initialization)
        # 'zero-fix': fixed zero init (not learnable)
        # 'zero': start from zero but learnable (MAML-style)
        # 'xavier'/'kaiming'/'orthogonal'/'normal'/'identity': specific init + learnable
        self.learnable_init = (y_init != 'zero-fix')
        if self.learnable_init:
            self.y_init_param = nn.Parameter(torch.zeros(num_heads, head_dim, head_dim))
            self._init_y()
            self.z_init = nn.Parameter(torch.zeros(num_heads, head_dim, head_dim))

        self.osc_w = nn.Parameter(torch.Tensor(num_heads, head_dim, head_dim))
        self.osc_damp = nn.Parameter(torch.zeros(num_heads, head_dim, head_dim), requires_grad=self.damping)

        self._keep_default_init = True  # Flag to control default initialization in reset_parameters

    def reset_parameters(self):    
        # Log-space parameterization: freq = exp(osc_w)
        # Initialize with log-uniform distribution covering multiple scales
        self.osc_w_scale = 1.0
        
        # uniformly distributed in log-space
        # dt=1, seq_len=4096
        P_min, P_max = 4.0, 4096.0
        log_freq_min = math.log(2 * math.pi / P_max)
        log_freq_max = math.log(2 * math.pi / P_min)

        # noise_base = (log_freq_max - log_freq_min) / self.head_dim

        # Create linspace for head_dim, then broadcast to full shape
        init_values = torch.linspace(log_freq_min, log_freq_max, self.head_dim**2, device=self.osc_w.device, dtype=self.osc_w.dtype)  # [head_dim]
        # Add small noise to break symmetry across different heads/ranks
        noise = 0.05 * torch.randn(self.num_heads, self.head_dim, self.head_dim, device=self.osc_w.device, dtype=self.osc_w.dtype)
        # Broadcast and add noise
        with torch.no_grad():
            self.osc_w.copy_(
                init_values.unsqueeze(0).expand(self.num_heads, -1).reshape(self.num_heads, self.head_dim, self.head_dim) + noise
            )

    def get_osc_frequencies(self):
        """Return actual oscillation frequencies for monitoring.
        Returns dict with min, max, mean, std of frequencies."""
        with torch.no_grad():
            freq = torch.exp(self.osc_w_scale * self.osc_w)
            return {
                'min': freq.min().item(),
                'max': freq.max().item(),
                'mean': freq.mean().item(),
                'std': freq.std().item(),
            }

    def _init_y(self):
        """Initialize y_init_param. All options create learnable parameters."""
        if self.y_init == 'zero':
            # Start from zero, let gradient descent find optimal init
            nn.init.zeros_(self.y_init_param)
        elif self.y_init == 'normal':
            # Small normal initialization
            nn.init.normal_(self.y_init_param, mean=0.0, std=0.02)

    def _get_initial_state(self, batch_size, device, dtype):
        """Get initial y, z, S states. S is the primary state, y is derived from S."""
        if not self.learnable_init:
            # Fixed zero initialization
            y = torch.zeros(batch_size, self.num_heads,  self.head_dim, self.head_dim, device=device, dtype=dtype)
            z = torch.zeros(batch_size, self.num_heads,  self.head_dim, self.head_dim, device=device, dtype=dtype) 
        else:
            # Learnable initialization (MAML-style)
            y = self.y_init_param.unsqueeze(0).expand(batch_size, -1, -1).clone().to(dtype)
            z = self.z_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(dtype) 
        return y, z

    def _resolve_initial_state(self, initial_state, batch_size, device, dtype):
        if initial_state is None:
            return self._get_initial_state(batch_size, device, dtype)

        if isinstance(initial_state, dict):
            y = initial_state.get('y', None)
            z = initial_state.get('z', None)
        elif isinstance(initial_state, (tuple, list)) and len(initial_state) == 2:
            y, z = initial_state
        else:
            raise ValueError("initial_state must be None, a dict with keys 'y'/'z', or a 2-tuple (y, z).")

        if y is None or z is None:
            raise ValueError("initial_state must include both y and z tensors.")

        expected_tail = (self.num_heads, self.head_dim, self.head_dim)
        if y.ndim != 4 or z.ndim != 4 or y.shape[1:] != expected_tail or z.shape[1:] != expected_tail:
            raise ValueError(
                f"Expected y/z shapes [batch, {self.num_heads}, {self.head_dim}, {self.head_dim}], "
                f"got y={tuple(y.shape)}, z={tuple(z.shape)}"
            )
        if y.shape[0] != batch_size or z.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch in initial_state: expected {batch_size}, got y={y.shape[0]}, z={z.shape[0]}"
            )

        return y.to(device=device, dtype=dtype), z.to(device=device, dtype=dtype)

    def chunk_forward(self, q, k, v, beta, chunk_size, initial_state=None, output_final_state=False, cu_seqlens=None, use_qk_l2norm_in_kernel=False):
        device_type = q.device.type
        orig_dtype = q.dtype

        if use_qk_l2norm_in_kernel:
            q = l2norm(q)
            k = l2norm(k)

        batch_size, seq_len, _, _ = q.shape

        y0, z0 = self._resolve_initial_state(initial_state, batch_size, q.device, torch.float32)

        osc_term = torch.exp(self.osc_w_scale * self.osc_w)
        damping_param = self.osc_damp
        damping_term = nn.functional.sigmoid(damping_param) if self.damping else damping_param
        damping_term = damping_term / self.delta_t

        output, y, z = fused_chunk_linoss(
            k=k,
            q=q,
            v=v,
            beta=beta,
            y0=y0,
            z0=z0,
            A=osc_term.float(),
            B=damping_term.float(),
            dt=self.delta_t,
            chunk_size=chunk_size,
        )

        final_state = None
        if output_final_state:
            final_state = (y, z)
        return output.to(orig_dtype), final_state

    def recurrent_forward(self, 
                q, 
                k, 
                v, 
                beta,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=None,
                use_qk_l2norm_in_kernel=False
                ):

        device_type = q.device.type

        if use_qk_l2norm_in_kernel:
            q = l2norm(q)
            k = l2norm(k)

        batch_size, T, h, d = q.shape

        orig_dtype = q.dtype
        q, k, v, beta = map(lambda x: x.float(), [q, k, v, beta])
        scale = k.shape[-1] ** -0.5
        q = q * scale

        # Get initial states using learnable or zero initialization
        y, z = self._resolve_initial_state(initial_state, batch_size, q.device, torch.float32)

        if beta.ndim == 3:
            beta = beta[..., None, None]  # [batch, T, num_heads, 1, 1] for broadcasting

        # Log-space with scale: exp(scale * osc_w) for more stable gradients
        osc_term = torch.exp(self.osc_w_scale * self.osc_w)
        # osc_term shape: [num_heads, head_dim, head_dim] - same as y (without batch)
        damping_param = self.osc_damp
        damping_term = nn.functional.sigmoid(damping_param) if self.damping else damping_param
        damping_term = damping_term / self.delta_t

        output = torch.zeros(batch_size, T, h, d, device=q.device, dtype=orig_dtype)
        for t in range(T):
            _q = q[:, t, :, :]
            _k = k[:, t, :, :]
            _v = v[:, t, :, :].clone()
            beta_i = beta[:, t, :]  # [batch, num_heads, 1]

            # _v = _v - torch.einsum('bhd,bhde->bhe', _k, y)
            # u = kv^T = einsum('bhd,bhrd,bhre->bhdr', k, y, v) -> [batch, num_heads, head_dim, head_dim]
            u_t = torch.einsum('bhd,bhe->bhde', _k, _v)

            # gradient clipping on u_t (same as parallel_forward)
            # u_norm = torch.linalg.vector_norm(u_t, dim=-1, keepdim=True)
            # scale = torch.minimum(self.grad_clip_scale / (u_norm + 1e-6), torch.ones_like(u_norm))
            # u_t = u_t * scale

            z = z.clone() + self.delta_t * beta_i * u_t - self.delta_t * osc_term * y - self.delta_t * damping_term * z
            y = y.clone() + self.delta_t * z

            output[:, t, :, :] = torch.einsum('bhd,bhde->bhe', _q, y)

        final_state = None
        if output_final_state:
            final_state = (y.contiguous(), z.contiguous())
        return output.to(orig_dtype), final_state
