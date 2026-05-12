import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from fla.modules.l2norm import l2norm
from .kernel.hebb_linoss_kernel import fused_hebb_linoss
from .kernel.delta_linoss_kernel import fused_delta_linoss

class LinOSS(nn.Module):
    def __init__(self, num_heads: int, 
                 head_dim: int, 
                 delta_t: float, 
                 damping: bool, 
                 grad_clip_scale: float,
                 y_init: str,
                 update_rule: str,
                 monitor: bool = True):

        super().__init__()
        self.delta_t = delta_t
        self.damping = damping
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.grad_clip_scale = grad_clip_scale
        self.y_init = y_init
        self.update_rule = update_rule
        self.monitor = monitor  # Set to False for torch.compile compatibility

        if update_rule == 'hebb':
            self.kernel_func = fused_hebb_linoss
        elif update_rule == 'delta':
            self.kernel_func = fused_delta_linoss

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
        # Exclude LinOSS parameters from optimizer weight decay (use custom reg instead)
        self.osc_w._optim = {"weight_decay": 0.0}
        self.osc_damp._optim = {"weight_decay": 0.0}

        self._keep_default_init = True  # Flag to control default initialization in reset_parameters

        # Nyquist upper bound with safety margin
        self.log_omega_max = math.log(2.0 / self.delta_t * 0.99)

    def reset_parameters(self):
        # Start near zero (DeltaNet regime) and let the model learn oscillation.
        # exp(-9) ≈ 1.2e-4 keeps Euler stable over long sequences.
        with torch.no_grad():
            nn.init.normal_(self.osc_w, mean=-9.0, std=0.5)

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
            y = self.y_init_param.unsqueeze(0).expand(batch_size, -1, -1, -1).clone().to(dtype)
            z = self.z_init.unsqueeze(0).expand(batch_size, -1, -1, -1).clone().to(dtype)
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

    def get_A(self):
        log_omega = self.osc_w
        log_omega_capped = self.log_omega_max - F.softplus(self.log_omega_max - log_omega)
        osc_term = torch.exp(2.0 * log_omega_capped)
        return osc_term

    def get_B(self):
        damping_term = nn.functional.sigmoid(self.osc_damp) if self.damping else self.osc_damp
        damping_term = damping_term / self.delta_t
        return damping_term

    def chunk_forward(self, q, k, v, beta, chunk_size, initial_state=None, output_final_state=False, cu_seqlens=None, use_qk_l2norm_in_kernel=False):
        orig_dtype = q.dtype

        if use_qk_l2norm_in_kernel:
            q = l2norm(q)
            k = l2norm(k)

        batch_size = cu_seqlens.numel() - 1 if cu_seqlens is not None else q.shape[0]

        y0, z0 = self._resolve_initial_state(initial_state, batch_size, q.device, torch.float32)

        osc_term = self.get_A()
        damping_param = self.get_B()

        # osc_term = torch.zeros_like(self.osc_w)
        # damping_term = torch.ones_like(self.osc_damp)

        output, y, z = self.kernel_func(
            q=q,
            k=k,
            v=v,
            beta=beta,
            y0=y0,
            z0=z0,
            A=osc_term.float(),
            B=damping_term.float(),
            cu_seqlens=cu_seqlens,
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

        if use_qk_l2norm_in_kernel:
            q = l2norm(q)
            k = l2norm(k)

        # varlen packed: q/k/v are (N, H, D); cu_seqlens is (Bsz+1,)
        batch_size = cu_seqlens.numel() - 1 if cu_seqlens is not None else q.shape[0]

        orig_dtype = q.dtype
        q, k, v, beta = map(lambda x: x.float(), [q, k, v, beta])
        scale = k.shape[-1] ** -0.5
        q = q * scale

        y, z = self._resolve_initial_state(initial_state, batch_size, q.device, torch.float32)

        osc_term = self.get_A()
        damping_param = self.get_B()

        if cu_seqlens is not None:
            output = torch.zeros(q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=orig_dtype)
            for b in range(batch_size):
                seq_start = cu_seqlens[b].item()
                seq_end = cu_seqlens[b + 1].item()
                yb, zb = y[b], z[b]  # (H, D, D)
                for t in range(seq_start, seq_end):
                    _k = k[t]
                    _v = v[t].clone()
                    if self.update_rule == 'delta':
                        _v = _v - torch.einsum('hd,hde->he', _k, yb)
                    u_t = torch.einsum('hd,he->hde', _k, _v)
                    zb = zb + self.delta_t * beta[t, :, None, None] * u_t - self.delta_t * osc_term * yb - self.delta_t * damping_term * zb
                    yb = yb + self.delta_t * zb
                    output[t] = torch.einsum('hd,hde->he', q[t], yb).to(orig_dtype)
                y[b], z[b] = yb, zb
        else:
            B, T, h, d = q.shape
            output = torch.zeros(B, T, h, d, device=q.device, dtype=orig_dtype)
            beta_b = beta[..., None, None] if beta.ndim == 3 else beta
            for t in range(T):
                _k = k[:, t]
                _v = v[:, t].clone()
                if self.update_rule == 'delta':
                    _v = _v - torch.einsum('bhd,bhde->bhe', _k, y)
                u_t = torch.einsum('bhd,bhe->bhde', _k, _v)
                z = z + self.delta_t * beta_b[:, t] * u_t - self.delta_t * osc_term * y - self.delta_t * damping_term * z
                y = y + self.delta_t * z
                output[:, t] = torch.einsum('bhd,bhde->bhe', q[:, t], y).to(orig_dtype)

        final_state = None
        if output_final_state:
            final_state = (y.contiguous(), z.contiguous())
        return output, final_state
