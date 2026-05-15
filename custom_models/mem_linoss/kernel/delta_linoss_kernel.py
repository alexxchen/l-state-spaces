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
    COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    col_block = pid % (D // COLS)
    bh = pid // (D // COLS)
    head = bh % H
    batch = bh // H

    rows = tl.arange(0, BLOCK_D)
    cols = col_block * COLS + tl.arange(0, COLS)
    rmask = rows < D

    # 2D pointers: rows in axis 0, cols in axis 1
    state_base = ((batch * H + head) * D + rows[:, None]) * D + cols[None, :]
    param_base = (head * D + rows[:, None]) * D + cols[None, :]
    state_mask = rmask[:, None]

    seq_start = tl.load(cu_seqlens_ptr + batch).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + batch + 1).to(tl.int32)
    T_b = seq_end - seq_start

    y = tl.load(y0_ptr + state_base, mask=state_mask, other=0.0).to(tl.float32)
    z = tl.load(z0_ptr + state_base, mask=state_mask, other=0.0).to(tl.float32)
    a = tl.load(A_ptr + param_base, mask=state_mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + param_base, mask=state_mask, other=0.0).to(tl.float32)
    dt_a = DT * a
    one_minus_dtb = 1.0 - DT * b

    ckpt_chunk_stride = H * D * D
    ckpt_batch_stride = (NUM_CHUNKS_MAX + 1) * ckpt_chunk_stride
    head_stride = D * D

    ckpt_base0 = batch * ckpt_batch_stride + head * head_stride + rows[:, None] * D + cols[None, :]
    tl.store(y_check_ptr + ckpt_base0, y, mask=state_mask)
    tl.store(z_check_ptr + ckpt_base0, z, mask=state_mask)

    for t in tl.range(0, T_b, 1, num_stages=2):
        n = seq_start + t
        seq_base = (n * H + head) * D
        k_t = tl.load(k_ptr + seq_base + rows, mask=rmask, other=0.0).to(tl.float32)
        q_t = (
            tl.load(q_ptr + seq_base + rows, mask=rmask, other=0.0).to(tl.float32)
            * SCALE
        )
        v_t = tl.load(v_ptr + seq_base + cols).to(tl.float32)  # shape (COLS,)
        beta_t = tl.load(beta_ptr + n * H + head).to(tl.float32)

        # residual: shape (COLS,)
        residual = v_t - tl.sum(k_t[:, None] * y, axis=0)
        # outer product k_t[:, None] * (DT*beta_t*residual)[None, :] -> (BLOCK_D, COLS)
        z = z * one_minus_dtb - dt_a * y + k_t[:, None] * (DT * beta_t * residual)[None, :]
        y = y + DT * z

        out_t = tl.sum(q_t[:, None] * y, axis=0)  # shape (COLS,)
        tl.store(out_ptr + seq_base + cols, out_t)

        if ((t + 1) % CHUNK_SIZE == 0) or (t == T_b - 1):
            chunk_idx = tl.cdiv(t + 1, CHUNK_SIZE)
            ckpt_base_t = (
                batch * ckpt_batch_stride
                + chunk_idx * ckpt_chunk_stride
                + head * head_stride
                + rows[:, None] * D
                + cols[None, :]
            )
            tl.store(y_check_ptr + ckpt_base_t, y, mask=state_mask)
            tl.store(z_check_ptr + ckpt_base_t, z, mask=state_mask)

    tl.store(y_out_ptr + state_base, y, mask=state_mask)
    tl.store(z_out_ptr + state_base, z, mask=state_mask)


@triton.jit
def _linoss_varlen_recompute_backward_chunk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    A_ptr,
    B_ptr,
    cu_seqlens_ptr,
    grad_out_ptr,
    y_check_ptr,
    z_check_ptr,
    y_hist_ptr,
    gy_end_ptr,
    gz_end_ptr,
    grad_q_part_ptr,
    grad_k_part_ptr,
    grad_v_ptr,
    grad_beta_part_ptr,
    grad_A_batch_ptr,
    grad_B_batch_ptr,
    gy_start_ptr,
    gz_start_ptr,
    chunk_idx,
    H: tl.constexpr,
    D: tl.constexpr,
    DT: tl.constexpr,
    SCALE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_CHUNKS_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    col_block = pid % (D // COLS)
    bh = pid // (D // COLS)
    head = bh % H
    batch = bh // H

    rows = tl.arange(0, BLOCK_D)
    cols = col_block * COLS + tl.arange(0, COLS)
    rmask = rows < D

    state_base = ((batch * H + head) * D + rows[:, None]) * D + cols[None, :]
    param_base = (head * D + rows[:, None]) * D + cols[None, :]
    state_mask = rmask[:, None]

    seq_start = tl.load(cu_seqlens_ptr + batch).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + batch + 1).to(tl.int32)
    T_b = seq_end - seq_start
    chunk_start = chunk_idx * CHUNK_SIZE

    gy = tl.load(gy_end_ptr + state_base, mask=state_mask, other=0.0).to(tl.float32)
    gz = tl.load(gz_end_ptr + state_base, mask=state_mask, other=0.0).to(tl.float32)

    if chunk_start < T_b:
        remaining = T_b - chunk_start
        chunk_len = tl.where(remaining < CHUNK_SIZE, remaining, CHUNK_SIZE)

        hist_stride = (CHUNK_SIZE + 1) * D * D
        hist_base = ((batch * H + head) * hist_stride) + rows[:, None] * D + cols[None, :]

        ckpt_chunk_stride = H * D * D
        ckpt_batch_stride = (NUM_CHUNKS_MAX + 1) * ckpt_chunk_stride
        head_stride = D * D
        ckpt_base = (
            batch * ckpt_batch_stride
            + chunk_idx * ckpt_chunk_stride
            + head * head_stride
            + rows[:, None] * D
            + cols[None, :]
        )

        a = tl.load(A_ptr + param_base, mask=state_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + param_base, mask=state_mask, other=0.0).to(tl.float32)
        dt_a = DT * a
        one_minus_dtb = 1.0 - DT * b
        inv_one_minus_dtb = 1.0 / one_minus_dtb

        # ---- Recompute pass: forward through chunk, write y_hist ----
        y = tl.load(y_check_ptr + ckpt_base, mask=state_mask, other=0.0).to(tl.float32)
        z = tl.load(z_check_ptr + ckpt_base, mask=state_mask, other=0.0).to(tl.float32)
        tl.store(y_hist_ptr + hist_base, y, mask=state_mask)

        for local_t in tl.range(0, chunk_len, 1):
            t = chunk_start + local_t
            n = seq_start + t
            seq_base = (n * H + head) * D
            k_t = tl.load(k_ptr + seq_base + rows, mask=rmask, other=0.0).to(tl.float32)
            v_t = tl.load(v_ptr + seq_base + cols).to(tl.float32)  # (COLS,)
            beta_t = tl.load(beta_ptr + n * H + head).to(tl.float32)

            residual = v_t - tl.sum(k_t[:, None] * y, axis=0)  # (COLS,)
            z = z * one_minus_dtb - dt_a * y + k_t[:, None] * (DT * beta_t * residual)[None, :]
            y = y + DT * z

            hist_t_base = hist_base + (local_t + 1) * D * D
            tl.store(y_hist_ptr + hist_t_base, y, mask=state_mask)

        # ---- Backward pass: read y_hist in reverse ----
        grad_a_acc = tl.zeros((BLOCK_D, COLS), dtype=tl.float32)
        grad_b_acc = tl.zeros((BLOCK_D, COLS), dtype=tl.float32)

        # y at chunk end is in `y` from the recompute loop
        y_next = y

        for rev_t in tl.range(0, chunk_len, 1):
            local_t = chunk_len - 1 - rev_t
            t = chunk_start + local_t
            n = seq_start + t
            seq_base = (n * H + head) * D

            y_prev = tl.load(
                y_hist_ptr + hist_base + local_t * D * D, mask=state_mask, other=0.0
            ).to(tl.float32)

            q_t = tl.load(q_ptr + seq_base + rows, mask=rmask, other=0.0).to(tl.float32)
            k_t = tl.load(k_ptr + seq_base + rows, mask=rmask, other=0.0).to(tl.float32)
            v_t = tl.load(v_ptr + seq_base + cols).to(tl.float32)  # (COLS,)
            beta_t = tl.load(beta_ptr + n * H + head).to(tl.float32)
            go = tl.load(grad_out_ptr + seq_base + cols).to(tl.float32)  # (COLS,)

            # z_prev via single-step inversion
            dt_beta = DT * beta_t
            residual = v_t - tl.sum(k_t[:, None] * y_prev, axis=0)  # (COLS,)
            z_new = (y_next - y_prev) * (1.0 / DT)
            z_prev = (z_new + dt_a * y_prev - k_t[:, None] * (dt_beta * residual)[None, :]) * inv_one_minus_dtb

            grad_q_contrib = tl.sum((SCALE * go)[None, :] * y_next, axis=1)  # (BLOCK_D,)
            partial_base = (
                ((batch * H + head) * (D // COLS) + col_block) * CHUNK_SIZE
                + local_t
            )
            tl.store(grad_q_part_ptr + partial_base * D + rows, grad_q_contrib, mask=rmask)
            gy = gy + (SCALE * go)[None, :] * q_t[:, None]

            gz_total = gz + DT * gy
            kdot = DT * tl.sum(gz_total * k_t[:, None], axis=0)  # (COLS,)
            gr = beta_t * kdot  # (COLS,)

            tl.store(grad_v_ptr + seq_base + cols, gr)
            tl.store(grad_beta_part_ptr + partial_base, tl.sum(kdot * residual, axis=0))
            # grad_k contribution per col: gz_total[r,c]*(dt_beta*residual[c]) - gr[c]*y_prev[r,c]
            grad_k_contrib = tl.sum(
                gz_total * (dt_beta * residual)[None, :] - gr[None, :] * y_prev,
                axis=1,
            )  # (BLOCK_D,)
            tl.store(grad_k_part_ptr + partial_base * D + rows, grad_k_contrib, mask=rmask)
            dt_gz = DT * gz_total
            grad_a_acc += -dt_gz * y_prev
            grad_b_acc += -dt_gz * z_prev

            gy = gy - a * dt_gz - k_t[:, None] * gr[None, :]
            gz = gz_total * one_minus_dtb

            y_next = y_prev

        grad_a_prev = tl.load(grad_A_batch_ptr + state_base, mask=state_mask, other=0.0)
        grad_b_prev = tl.load(grad_B_batch_ptr + state_base, mask=state_mask, other=0.0)
        tl.store(grad_A_batch_ptr + state_base, grad_a_prev + grad_a_acc, mask=state_mask)
        tl.store(grad_B_batch_ptr + state_base, grad_b_prev + grad_b_acc, mask=state_mask)

    tl.store(gy_start_ptr + state_base, gy, mask=state_mask)
    tl.store(gz_start_ptr + state_base, gz, mask=state_mask)


@triton.jit
def _linoss_varlen_reduce_col_partials_kernel(
    cu_seqlens_ptr,
    grad_q_part_ptr,
    grad_k_part_ptr,
    grad_beta_part_ptr,
    grad_q_ptr,
    grad_k_ptr,
    grad_beta_ptr,
    chunk_idx,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_COL_BLOCKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    local_t = pid % CHUNK_SIZE
    bh = pid // CHUNK_SIZE
    head = bh % H
    batch = bh // H

    rows = tl.arange(0, BLOCK_D)
    rmask = rows < D

    seq_start = tl.load(cu_seqlens_ptr + batch).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + batch + 1).to(tl.int32)
    T_b = seq_end - seq_start
    chunk_start = chunk_idx * CHUNK_SIZE

    if local_t < T_b - chunk_start:
        q_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        k_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        beta_acc = tl.zeros((), dtype=tl.float32)

        for col_block in tl.range(0, NUM_COL_BLOCKS, 1):
            partial_base = (
                ((batch * H + head) * NUM_COL_BLOCKS + col_block) * CHUNK_SIZE
                + local_t
            )
            q_acc += tl.load(grad_q_part_ptr + partial_base * D + rows, mask=rmask, other=0.0)
            k_acc += tl.load(grad_k_part_ptr + partial_base * D + rows, mask=rmask, other=0.0)
            beta_acc += tl.load(grad_beta_part_ptr + partial_base)

        n = seq_start + chunk_start + local_t
        seq_base = (n * H + head) * D
        tl.store(grad_q_ptr + seq_base + rows, q_acc, mask=rmask)
        tl.store(grad_k_ptr + seq_base + rows, k_acc, mask=rmask)
        tl.store(grad_beta_ptr + n * H + head, beta_acc)


@triton.jit
def _linoss_reduce_batch_matrix_grads_kernel(
    grad_A_batch_ptr,
    grad_B_batch_ptr,
    grad_A_ptr,
    grad_B_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    col_block = pid % (D // COLS)
    head = pid // (D // COLS)

    rows = tl.arange(0, BLOCK_D)
    cols = col_block * COLS + tl.arange(0, COLS)
    rmask = rows < D
    state_mask = rmask[:, None]

    grad_a = tl.zeros((BLOCK_D, COLS), dtype=tl.float32)
    grad_b = tl.zeros((BLOCK_D, COLS), dtype=tl.float32)

    for batch in tl.range(0, BATCH_SIZE, 1):
        state_base = ((batch * H + head) * D + rows[:, None]) * D + cols[None, :]
        grad_a += tl.load(grad_A_batch_ptr + state_base, mask=state_mask, other=0.0)
        grad_b += tl.load(grad_B_batch_ptr + state_base, mask=state_mask, other=0.0)

    param_base = (head * D + rows[:, None]) * D + cols[None, :]
    tl.store(grad_A_ptr + param_base, grad_a, mask=state_mask)
    tl.store(grad_B_ptr + param_base, grad_b, mask=state_mask)


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
        cols_per_program = 16
        assert D_ % cols_per_program == 0
        grid = (Bsz * H_ * (D_ // cols_per_program),)

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
            COLS=cols_per_program,
            num_warps=2,
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
        cols_per_program = 16
        assert D_ % cols_per_program == 0
        num_col_blocks = D_ // cols_per_program
        grid = (Bsz * H_ * (D_ // cols_per_program),)
        reduce_col_grid = (Bsz * H_ * chunk_size,)
        reduce_batch_grid = (H_ * num_col_blocks,)

        if grad_output is None:
            grad_output = torch.zeros_like(q)
        grad_output = grad_output.contiguous()

        grad_q = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        grad_k = torch.empty_like(grad_q)
        grad_v = torch.empty_like(grad_q)
        grad_beta = torch.empty(beta.shape, device=beta.device, dtype=torch.float32)
        grad_A = torch.empty(A.shape, device=A.device, dtype=torch.float32)
        grad_B = torch.empty(B.shape, device=B.device, dtype=torch.float32)
        grad_q_part = torch.empty(
            (Bsz, H_, num_col_blocks, chunk_size, D_),
            device=q.device,
            dtype=torch.float32,
        )
        grad_k_part = torch.empty_like(grad_q_part)
        grad_beta_part = torch.empty(
            (Bsz, H_, num_col_blocks, chunk_size),
            device=beta.device,
            dtype=torch.float32,
        )
        grad_A_batch = torch.zeros(
            (Bsz, H_, D_, D_), device=A.device, dtype=torch.float32
        )
        grad_B_batch = torch.zeros_like(grad_A_batch)

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

        for chunk_idx in range(num_chunks_max - 1, -1, -1):
            _linoss_varlen_recompute_backward_chunk_kernel[grid](
                q,
                k,
                v,
                beta,
                A,
                B,
                cu_seqlens,
                grad_output,
                y_checkpoints,
                z_checkpoints,
                y_hist,
                gy_curr,
                gz_curr,
                grad_q_part,
                grad_k_part,
                grad_v,
                grad_beta_part,
                grad_A_batch,
                grad_B_batch,
                gy_next,
                gz_next,
                int(chunk_idx),
                H_,
                D_,
                ctx.dt,
                ctx.scale,
                chunk_size,
                num_chunks_max,
                BLOCK_D=block_d,
                COLS=cols_per_program,
                num_warps=4,
            )
            _linoss_varlen_reduce_col_partials_kernel[reduce_col_grid](
                cu_seqlens,
                grad_q_part,
                grad_k_part,
                grad_beta_part,
                grad_q,
                grad_k,
                grad_beta,
                int(chunk_idx),
                H_,
                D_,
                chunk_size,
                num_col_blocks,
                BLOCK_D=block_d,
                num_warps=4,
            )
            gy_curr, gy_next = gy_next, gy_curr
            gz_curr, gz_next = gz_next, gz_curr

        _linoss_reduce_batch_matrix_grads_kernel[reduce_batch_grid](
            grad_A_batch,
            grad_B_batch,
            grad_A,
            grad_B,
            H_,
            D_,
            Bsz,
            BLOCK_D=block_d,
            COLS=cols_per_program,
            num_warps=4,
        )

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

@torch.compiler.disable
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

    if y0.dim() == 3 and z0.dim() == 3:
        expand_dim = cu_seqlens.numel() - 1
        y0 = y0.unsqueeze(0).expand(expand_dim, -1, -1, -1)
        z0 = z0.unsqueeze(0).expand(expand_dim, -1, -1, -1)

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
