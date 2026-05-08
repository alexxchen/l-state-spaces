import torch
import triton
import triton.language as tl


@triton.jit
def _linoss_varlen_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    y0_ptr,
    z0_ptr,
    A_ptr,
    B_ptr,
    cu_seqlens_ptr,
    out_ptr,
    y_out_ptr,
    z_out_ptr,
    y_check_ptr,
    z_check_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    DT: tl.constexpr,
    SCALE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_CHUNKS_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    col = pid % D
    bh = pid // D
    head = bh % H
    batch = bh // H

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    state_base = ((batch * H + head) * D + offs) * D + col
    param_base = (head * D + offs) * D + col

    seq_start = tl.load(cu_seqlens_ptr + batch).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + batch + 1).to(tl.int32)
    T_b = seq_end - seq_start

    y = tl.load(y0_ptr + state_base, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z0_ptr + state_base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_ptr + param_base, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + param_base, mask=mask, other=0.0).to(tl.float32)

    ckpt_chunk_stride = H * D * D
    ckpt_batch_stride = (NUM_CHUNKS_MAX + 1) * ckpt_chunk_stride
    head_stride = D * D

    ckpt_base0 = batch * ckpt_batch_stride + head * head_stride + offs * D + col
    tl.store(y_check_ptr + ckpt_base0, y, mask=mask)
    tl.store(z_check_ptr + ckpt_base0, z, mask=mask)

    for t in tl.range(0, T_b, 1):
        n = seq_start + t
        seq_base = (n * H + head) * D
        k_t = tl.load(k_ptr + seq_base + offs, mask=mask, other=0.0).to(tl.float32)
        q_t = (
            tl.load(q_ptr + seq_base + offs, mask=mask, other=0.0).to(tl.float32)
            * SCALE
        )
        v_t = tl.load(v_ptr + seq_base + col).to(tl.float32)
        beta_t = tl.load(beta_ptr + n * H + head).to(tl.float32)

        residual = v_t - tl.sum(k_t * y, axis=0)
        z = z - DT * a * y + DT * beta_t * k_t * residual - DT * b * z
        y = y + DT * z

        out_t = tl.sum(q_t * y, axis=0)
        tl.store(out_ptr + seq_base + col, out_t)

        if ((t + 1) % CHUNK_SIZE == 0) or (t == T_b - 1):
            chunk_idx = tl.cdiv(t + 1, CHUNK_SIZE)
            ckpt_base_t = (
                batch * ckpt_batch_stride
                + chunk_idx * ckpt_chunk_stride
                + head * head_stride
                + offs * D
                + col
            )
            tl.store(y_check_ptr + ckpt_base_t, y, mask=mask)
            tl.store(z_check_ptr + ckpt_base_t, z, mask=mask)

    tl.store(y_out_ptr + state_base, y, mask=mask)
    tl.store(z_out_ptr + state_base, z, mask=mask)


@triton.jit
def _linoss_varlen_recompute_chunk_kernel(
    k_ptr,
    v_ptr,
    beta_ptr,
    A_ptr,
    B_ptr,
    cu_seqlens_ptr,
    y_check_ptr,
    z_check_ptr,
    y_hist_ptr,
    z_hist_ptr,
    chunk_idx,
    H: tl.constexpr,
    D: tl.constexpr,
    DT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_CHUNKS_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    col = pid % D
    bh = pid // D
    head = bh % H
    batch = bh // H

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    seq_start = tl.load(cu_seqlens_ptr + batch).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + batch + 1).to(tl.int32)
    T_b = seq_end - seq_start
    chunk_start = chunk_idx * CHUNK_SIZE

    if chunk_start < T_b:
        remaining = T_b - chunk_start
        chunk_len = tl.where(remaining < CHUNK_SIZE, remaining, CHUNK_SIZE)

        hist_stride = (CHUNK_SIZE + 1) * D * D
        hist_base = ((batch * H + head) * hist_stride) + offs * D + col

        ckpt_chunk_stride = H * D * D
        ckpt_batch_stride = (NUM_CHUNKS_MAX + 1) * ckpt_chunk_stride
        head_stride = D * D
        ckpt_base = (
            batch * ckpt_batch_stride
            + chunk_idx * ckpt_chunk_stride
            + head * head_stride
            + offs * D
            + col
        )

        param_base = (head * D + offs) * D + col

        y = tl.load(y_check_ptr + ckpt_base, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_check_ptr + ckpt_base, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(A_ptr + param_base, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + param_base, mask=mask, other=0.0).to(tl.float32)

        tl.store(y_hist_ptr + hist_base, y, mask=mask)
        tl.store(z_hist_ptr + hist_base, z, mask=mask)

        for local_t in tl.range(0, chunk_len, 1):
            t = chunk_start + local_t
            n = seq_start + t
            seq_base = (n * H + head) * D
            k_t = tl.load(k_ptr + seq_base + offs, mask=mask, other=0.0).to(tl.float32)
            v_t = tl.load(v_ptr + seq_base + col).to(tl.float32)
            beta_t = tl.load(beta_ptr + n * H + head).to(tl.float32)

            residual = v_t - tl.sum(k_t * y, axis=0)
            z = z - DT * a * y + DT * beta_t * k_t * residual - DT * b * z
            y = y + DT * z

            hist_t_base = hist_base + (local_t + 1) * D * D
            tl.store(y_hist_ptr + hist_t_base, y, mask=mask)
            tl.store(z_hist_ptr + hist_t_base, z, mask=mask)


@triton.jit
def _linoss_varlen_backward_chunk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    A_ptr,
    B_ptr,
    cu_seqlens_ptr,
    grad_out_ptr,
    y_hist_ptr,
    z_hist_ptr,
    gy_end_ptr,
    gz_end_ptr,
    grad_q_ptr,
    grad_k_ptr,
    grad_v_ptr,
    grad_beta_ptr,
    grad_A_ptr,
    grad_B_ptr,
    gy_start_ptr,
    gz_start_ptr,
    chunk_idx,
    H: tl.constexpr,
    D: tl.constexpr,
    DT: tl.constexpr,
    SCALE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    col = pid % D
    bh = pid // D
    head = bh % H
    batch = bh // H

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    state_base = ((batch * H + head) * D + offs) * D + col
    param_base = (head * D + offs) * D + col

    seq_start = tl.load(cu_seqlens_ptr + batch).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + batch + 1).to(tl.int32)
    T_b = seq_end - seq_start
    chunk_start = chunk_idx * CHUNK_SIZE

    gy = tl.load(gy_end_ptr + state_base, mask=mask, other=0.0).to(tl.float32)
    gz = tl.load(gz_end_ptr + state_base, mask=mask, other=0.0).to(tl.float32)

    if chunk_start < T_b:
        remaining = T_b - chunk_start
        chunk_len = tl.where(remaining < CHUNK_SIZE, remaining, CHUNK_SIZE)

        hist_stride = (CHUNK_SIZE + 1) * D * D
        hist_base = ((batch * H + head) * hist_stride) + offs * D + col

        a = tl.load(A_ptr + param_base, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + param_base, mask=mask, other=0.0).to(tl.float32)

        for rev_t in tl.range(0, chunk_len, 1):
            local_t = chunk_len - 1 - rev_t
            t = chunk_start + local_t
            n = seq_start + t
            seq_base = (n * H + head) * D

            y_prev = tl.load(
                y_hist_ptr + hist_base + local_t * D * D, mask=mask, other=0.0
            ).to(tl.float32)
            z_prev = tl.load(
                z_hist_ptr + hist_base + local_t * D * D, mask=mask, other=0.0
            ).to(tl.float32)
            y_next = tl.load(
                y_hist_ptr + hist_base + (local_t + 1) * D * D, mask=mask, other=0.0
            ).to(tl.float32)

            q_t = tl.load(q_ptr + seq_base + offs, mask=mask, other=0.0).to(tl.float32)
            k_t = tl.load(k_ptr + seq_base + offs, mask=mask, other=0.0).to(tl.float32)
            v_t = tl.load(v_ptr + seq_base + col).to(tl.float32)
            beta_t = tl.load(beta_ptr + n * H + head).to(tl.float32)
            go = tl.load(grad_out_ptr + seq_base + col).to(tl.float32)

            tl.atomic_add(
                grad_q_ptr + seq_base + offs,
                go * SCALE * y_next,
                sem="relaxed",
                mask=mask,
            )
            gy = gy + go * SCALE * q_t

            gz_total = gz + DT * gy
            residual = v_t - tl.sum(k_t * y_prev, axis=0)
            gr = tl.sum(gz_total * (DT * beta_t) * k_t, axis=0)

            tl.store(grad_v_ptr + seq_base + col, gr)
            tl.atomic_add(
                grad_beta_ptr + n * H + head,
                tl.sum(gz_total * DT * k_t * residual, axis=0),
                sem="relaxed",
            )
            tl.atomic_add(
                grad_k_ptr + seq_base + offs,
                gz_total * (DT * beta_t * residual) - gr * y_prev,
                sem="relaxed",
                mask=mask,
            )
            tl.atomic_add(
                grad_A_ptr + param_base,
                -DT * gz_total * y_prev,
                sem="relaxed",
                mask=mask,
            )
            tl.atomic_add(
                grad_B_ptr + param_base,
                -DT * gz_total * z_prev,
                sem="relaxed",
                mask=mask,
            )

            gy = gy - DT * a * gz_total - k_t * gr
            gz = gz_total * (1.0 - DT * b)

    tl.store(gy_start_ptr + state_base, gy, mask=mask)
    tl.store(gz_start_ptr + state_base, gz, mask=mask)


class _ParallelRNNVarlenTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        beta,
        y0,
        z0,
        A,
        B,
        cu_seqlens,
        dt,
        chunk_size,
        scale,
    ):
        # q, k, v: (N, H, D); beta: (N, H); y0, z0: (Bsz, H, D, D); A, B: (H, D, D)
        # cu_seqlens: (Bsz + 1,) int32
        H_, D_ = q.shape[1], q.shape[2]
        Bsz = cu_seqlens.numel() - 1
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        num_chunks_max = triton.cdiv(max_seqlen, int(chunk_size))

        output = torch.empty_like(q)
        y_final = torch.empty_like(y0)
        z_final = torch.empty_like(z0)
        y_checkpoints = torch.empty(
            (Bsz, num_chunks_max + 1, H_, D_, D_),
            device=q.device,
            dtype=torch.float32,
        )
        z_checkpoints = torch.empty_like(y_checkpoints)

        block_d = triton.next_power_of_2(D_)
        grid = (Bsz * H_ * D_,)

        _linoss_varlen_forward_kernel[grid](
            q,
            k,
            v,
            beta,
            y0,
            z0,
            A,
            B,
            cu_seqlens,
            output,
            y_final,
            z_final,
            y_checkpoints,
            z_checkpoints,
            H_,
            D_,
            float(dt),
            float(scale),
            int(chunk_size),
            int(num_chunks_max),
            BLOCK_D=block_d,
            num_warps=4,
        )

        ctx.save_for_backward(q, k, v, beta, A, B, cu_seqlens, y_checkpoints, z_checkpoints)
        ctx.dt = float(dt)
        ctx.chunk_size = int(chunk_size)
        ctx.scale = float(scale)
        ctx.num_chunks_max = int(num_chunks_max)
        ctx.batch_size = int(Bsz)
        return output, y_final, z_final

    @staticmethod
    def backward(ctx, grad_output, grad_y_final, grad_z_final):
        q, k, v, beta, A, B, cu_seqlens, y_checkpoints, z_checkpoints = ctx.saved_tensors
        needs_grad = ctx.needs_input_grad[:8]

        if not any(needs_grad):
            return (None,) * 12

        H_, D_ = q.shape[1], q.shape[2]
        Bsz = ctx.batch_size
        chunk_size = ctx.chunk_size
        num_chunks_max = ctx.num_chunks_max
        block_d = triton.next_power_of_2(D_)
        grid = (Bsz * H_ * D_,)

        if grad_output is None:
            grad_output = torch.zeros_like(q)
        grad_output = grad_output.contiguous()

        grad_q = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        grad_k = torch.zeros_like(grad_q)
        grad_v = torch.empty_like(grad_q)
        grad_beta = torch.zeros(beta.shape, device=beta.device, dtype=torch.float32)
        grad_A = torch.zeros(A.shape, device=A.device, dtype=torch.float32)
        grad_B = torch.zeros(B.shape, device=B.device, dtype=torch.float32)

        if grad_y_final is None:
            gy_curr = torch.zeros(
                (Bsz, H_, D_, D_), device=q.device, dtype=torch.float32
            )
        else:
            gy_curr = grad_y_final.contiguous().to(torch.float32)
        if grad_z_final is None:
            gz_curr = torch.zeros_like(gy_curr)
        else:
            gz_curr = grad_z_final.contiguous().to(torch.float32)
        gy_next = torch.empty_like(gy_curr)
        gz_next = torch.empty_like(gz_curr)

        y_hist = torch.empty(
            (Bsz, H_, chunk_size + 1, D_, D_),
            device=q.device,
            dtype=torch.float32,
        )
        z_hist = torch.empty_like(y_hist)

        for chunk_idx in range(num_chunks_max - 1, -1, -1):
            _linoss_varlen_recompute_chunk_kernel[grid](
                k,
                v,
                beta,
                A,
                B,
                cu_seqlens,
                y_checkpoints,
                z_checkpoints,
                y_hist,
                z_hist,
                int(chunk_idx),
                H_,
                D_,
                ctx.dt,
                chunk_size,
                num_chunks_max,
                BLOCK_D=block_d,
                num_warps=4,
            )
            _linoss_varlen_backward_chunk_kernel[grid](
                q,
                k,
                v,
                beta,
                A,
                B,
                cu_seqlens,
                grad_output,
                y_hist,
                z_hist,
                gy_curr,
                gz_curr,
                grad_q,
                grad_k,
                grad_v,
                grad_beta,
                grad_A,
                grad_B,
                gy_next,
                gz_next,
                int(chunk_idx),
                H_,
                D_,
                ctx.dt,
                ctx.scale,
                chunk_size,
                BLOCK_D=block_d,
                num_warps=4,
            )
            gy_curr, gy_next = gy_next, gy_curr
            gz_curr, gz_next = gz_next, gz_curr

        result = [
            grad_q.to(q.dtype) if needs_grad[0] else None,
            grad_k.to(k.dtype) if needs_grad[1] else None,
            grad_v.to(v.dtype) if needs_grad[2] else None,
            grad_beta.to(beta.dtype) if needs_grad[3] else None,
            gy_curr if needs_grad[4] else None,
            gz_curr if needs_grad[5] else None,
            grad_A if needs_grad[6] else None,
            grad_B if needs_grad[7] else None,
        ]
        # forward args: q, k, v, beta, y0, z0, A, B, cu_seqlens, dt, chunk_size, scale
        return (*result, None, None, None, None)


def fused_delta_linoss(q, k, v, beta, y0, z0, A, B, cu_seqlens, dt, chunk_size):
    """Parallel RNN with a unified (Bsz, N, H, D) input layout.

    q, k, v:    (Bsz, N, H, D)
    beta:       (Bsz, N, H)
    y0, z0:     (Bsz, H, D, D)
    A, B:       (H, D, D)
    cu_seqlens: optional (Bsz_seq + 1,) int32, cumulative sequence lengths.
        When provided, Bsz must be 1 and N is the total packed length across
        all sequences (state batch is taken from cu_seqlens).
        When None, each row in the leading dim is treated as one sequence of
        length N (a uniform cu_seqlens = [0, N, 2N, ..., Bsz*N] is built).

    Output is returned with shape (Bsz, N, H, D).
    """
    scale = k.shape[-1] ** -0.5
    Bsz, N, H_, D_ = q.shape
    q = q.reshape(Bsz * N, H_, D_)
    k = k.reshape(Bsz * N, H_, D_)
    v = v.reshape(Bsz * N, H_, D_)
    beta = beta.reshape(Bsz * N, H_)

    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0, (Bsz + 1) * N, N, dtype=torch.int32, device=q.device
        )
    else:
        assert Bsz == 1, "When cu_seqlens is provided, leading batch dim must be 1."

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    y0 = y0.contiguous()
    z0 = z0.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    cu_seqlens = cu_seqlens.contiguous().to(torch.int32)
    output, y_final, z_final = _ParallelRNNVarlenTriton.apply(
        q, k, v, beta, y0, z0, A, B, cu_seqlens, dt, chunk_size, scale
    )
    output = output.reshape(Bsz, N, H_, D_)
    return output, y_final, z_final
