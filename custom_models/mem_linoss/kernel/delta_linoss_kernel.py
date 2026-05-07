import torch
import triton
import triton.language as tl


@triton.jit
def _linoss_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    y0_ptr,
    z0_ptr,
    A_ptr,
    B_ptr,
    out_ptr,
    y_out_ptr,
    z_out_ptr,
    y_check_ptr,
    z_check_ptr,
    T: tl.constexpr,
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

    y = tl.load(y0_ptr + state_base, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z0_ptr + state_base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_ptr + param_base, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + param_base, mask=mask, other=0.0).to(tl.float32)
    dt_a = DT * a
    decay = 1.0 - DT * b

    check_stride = tl.program_id(0) * D
    tl.store(y_check_ptr + check_stride + offs, y, mask=mask)
    tl.store(z_check_ptr + check_stride + offs, z, mask=mask)

    for chunk in tl.range(0, T // CHUNK_SIZE, 1):
        chunk_start = chunk * CHUNK_SIZE
        for local_t in tl.range(0, CHUNK_SIZE, 1):
            t = chunk_start + local_t
            seq_base = ((batch * T + t) * H + head) * D
            k_t = tl.load(k_ptr + seq_base + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            q_t = (
                tl.load(q_ptr + seq_base + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                * SCALE
            )
            v_t = tl.load(v_ptr + seq_base + col).to(tl.float32)
            beta_t = tl.load(beta_ptr + (batch * T + t) * H + head).to(tl.float32)
            dt_beta = DT * beta_t

            residual = v_t - tl.sum(k_t * y, axis=0)
            z = decay * z - dt_a * y + dt_beta * k_t * residual
            y = y + DT * z

            out_t = tl.sum(q_t * y, axis=0)
            tl.store(out_ptr + seq_base + col, out_t)

        check_idx = chunk + 1
        check_base = (check_idx * tl.num_programs(0) + tl.program_id(0)) * D
        tl.store(y_check_ptr + check_base + offs, y, mask=mask)
        tl.store(z_check_ptr + check_base + offs, z, mask=mask)

    if T % CHUNK_SIZE != 0:
        chunk_start = (T // CHUNK_SIZE) * CHUNK_SIZE
        for local_t in tl.range(0, T % CHUNK_SIZE, 1):
            t = chunk_start + local_t
            seq_base = ((batch * T + t) * H + head) * D
            k_t = tl.load(k_ptr + seq_base + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            q_t = (
                tl.load(q_ptr + seq_base + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                * SCALE
            )
            v_t = tl.load(v_ptr + seq_base + col).to(tl.float32)
            beta_t = tl.load(beta_ptr + (batch * T + t) * H + head).to(tl.float32)
            dt_beta = DT * beta_t

            residual = v_t - tl.sum(k_t * y, axis=0)
            z = decay * z - dt_a * y + dt_beta * k_t * residual
            y = y + DT * z

            out_t = tl.sum(q_t * y, axis=0)
            tl.store(out_ptr + seq_base + col, out_t)

        check_idx = T // CHUNK_SIZE + 1
        check_base = (check_idx * tl.num_programs(0) + tl.program_id(0)) * D
        tl.store(y_check_ptr + check_base + offs, y, mask=mask)
        tl.store(z_check_ptr + check_base + offs, z, mask=mask)

    tl.store(y_out_ptr + state_base, y, mask=mask)
    tl.store(z_out_ptr + state_base, z, mask=mask)


@triton.jit
def _linoss_backward_chunk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    A_ptr,
    B_ptr,
    y_check_ptr,
    z_check_ptr,
    grad_out_ptr,
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
    GRAD_OUT_IS_ZERO: tl.constexpr,
    GRAD_OUT_IS_CONSTANT: tl.constexpr,
    GY_END_IS_ZERO: tl.constexpr,
    GZ_END_IS_ZERO: tl.constexpr,
    CHUNK_INDEX: tl.constexpr,
    CHUNK_START: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    T: tl.constexpr,
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
    check_base = ((CHUNK_INDEX + 1) * tl.num_programs(0) + pid) * D

    if GY_END_IS_ZERO:
        gy = tl.zeros((BLOCK_D,), dtype=tl.float32)
    else:
        gy = tl.load(gy_end_ptr + state_base, mask=mask, other=0.0).to(tl.float32)
    if GZ_END_IS_ZERO:
        gz = tl.zeros((BLOCK_D,), dtype=tl.float32)
    else:
        gz = tl.load(gz_end_ptr + state_base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_ptr + param_base, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + param_base, mask=mask, other=0.0).to(tl.float32)
    dt_a = DT * a
    decay = 1.0 - DT * b
    inv_decay = 1.0 / decay

    y_next = tl.load(y_check_ptr + check_base + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    z_next = tl.load(z_check_ptr + check_base + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    if GRAD_OUT_IS_CONSTANT:
        grad_out_constant = tl.load(grad_out_ptr).to(tl.float32)
    else:
        grad_out_constant = 0.0

    for rev_t in tl.range(0, CHUNK_LEN, 1):
        local_t = CHUNK_LEN - 1 - rev_t
        t = CHUNK_START + local_t
        seq_base = ((batch * T + t) * H + head) * D

        q_t = tl.load(q_ptr + seq_base + offs, mask=mask, other=0.0).to(tl.float32)
        k_t = tl.load(k_ptr + seq_base + offs, mask=mask, other=0.0).to(tl.float32)
        v_t = tl.load(v_ptr + seq_base + col).to(tl.float32)
        beta_t = tl.load(beta_ptr + (batch * T + t) * H + head).to(tl.float32)
        dt_beta = DT * beta_t
        if GRAD_OUT_IS_ZERO:
            go = 0.0
        elif GRAD_OUT_IS_CONSTANT:
            go = grad_out_constant
        else:
            go = tl.load(grad_out_ptr + seq_base + col).to(tl.float32)

        tl.atomic_add(
            grad_q_ptr + seq_base + offs,
            go * SCALE * y_next,
            sem="relaxed",
            mask=mask,
        )
        gy = gy + go * SCALE * q_t

        gz_total = gz + DT * gy
        y_prev = y_next - DT * z_next
        residual = v_t - tl.sum(k_t * y_prev, axis=0)
        dt_beta_residual = dt_beta * residual
        z_prev = (
            z_next + dt_a * y_prev - dt_beta_residual * k_t
        ) * inv_decay
        gr = tl.sum(gz_total * dt_beta * k_t, axis=0)

        tl.store(grad_v_ptr + seq_base + col, gr)
        tl.atomic_add(
            grad_beta_ptr + (batch * T + t) * H + head,
            tl.sum(gz_total * DT * k_t * residual, axis=0),
            sem="relaxed",
        )
        tl.atomic_add(
            grad_k_ptr + seq_base + offs,
            gz_total * dt_beta_residual - gr * y_prev,
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

        gy = gy - dt_a * gz_total - k_t * gr
        gz = gz_total * decay
        y_next = y_prev
        z_next = z_prev

    tl.store(gy_start_ptr + state_base, gy, mask=mask)
    tl.store(gz_start_ptr + state_base, gz, mask=mask)


class _ParallelRNNTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, y0, z0, A, B, dt, chunk_size, scale):
        ctx.set_materialize_grads(False)
        batch_size, T, h, d = q.shape
        output = torch.empty_like(q)
        y_final = torch.empty_like(y0)
        z_final = torch.empty_like(z0)
        num_chunks = triton.cdiv(T, chunk_size)
        y_checkpoints = torch.empty(
            (num_chunks + 1, batch_size, h, d, d),
            device=q.device,
            dtype=torch.float32,
        )
        z_checkpoints = torch.empty_like(y_checkpoints)

        block_d = triton.next_power_of_2(d)
        grid = (batch_size * h * d,)

        _linoss_forward_kernel[grid](
            q,
            k,
            v,
            beta,
            y0,
            z0,
            A,
            B,
            output,
            y_final,
            z_final,
            y_checkpoints,
            z_checkpoints,
            T,
            h,
            d,
            float(dt),
            float(scale),
            int(chunk_size),
            BLOCK_D=block_d,
            num_warps=1,
        )

        ctx.save_for_backward(q, k, v, beta, A, B, y_checkpoints, z_checkpoints)
        ctx.dt = float(dt)
        ctx.chunk_size = int(chunk_size)
        ctx.scale = float(scale)
        return output, y_final, z_final

    @staticmethod
    def backward(ctx, grad_output, grad_y_final, grad_z_final):
        q, k, v, beta, A, B, y_checkpoints, z_checkpoints = ctx.saved_tensors
        needs_grad = ctx.needs_input_grad[:8]

        if not any(needs_grad):
            return (None, None, None, None, None, None, None, None, None, None, None)

        batch_size, T, h, d = q.shape
        chunk_size = ctx.chunk_size
        num_chunks = triton.cdiv(T, chunk_size)
        block_d = triton.next_power_of_2(d)
        grid = (batch_size * h * d,)
        grad_output_is_none = grad_output is None
        grad_output_is_constant = (
            not grad_output_is_none
            and grad_output.numel() > 0
            and all(stride == 0 for stride in grad_output.stride())
        )
        if grad_output_is_none:
            grad_output_ptr = q
        elif grad_output_is_constant:
            grad_output_ptr = grad_output
        else:
            grad_output_ptr = grad_output.contiguous()

        grad_q = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        grad_k = torch.zeros_like(grad_q)
        grad_v = torch.empty_like(grad_q)
        grad_beta = torch.zeros(beta.shape, device=beta.device, dtype=torch.float32)
        grad_A = torch.zeros(A.shape, device=A.device, dtype=torch.float32)
        grad_B = torch.zeros(B.shape, device=B.device, dtype=torch.float32)

        grad_y_final_is_none = grad_y_final is None
        grad_z_final_is_none = grad_z_final is None

        if grad_y_final_is_none:
            gy_curr = torch.empty(
                (batch_size, h, d, d), device=q.device, dtype=torch.float32
            )
        else:
            gy_curr = grad_y_final.contiguous().to(torch.float32)
        if grad_z_final_is_none:
            gz_curr = torch.empty_like(gy_curr)
        else:
            gz_curr = grad_z_final.contiguous().to(torch.float32)
        gy_next = torch.empty_like(gy_curr)
        gz_next = torch.empty_like(gz_curr)

        for chunk_idx in range(num_chunks - 1, -1, -1):
            chunk_start = chunk_idx * chunk_size
            chunk_len = min(chunk_size, T - chunk_start)

            _linoss_backward_chunk_kernel[grid](
                q,
                k,
                v,
                beta,
                A,
                B,
                y_checkpoints,
                z_checkpoints,
                grad_output_ptr,
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
                grad_output_is_none,
                grad_output_is_constant,
                grad_y_final_is_none and chunk_idx == num_chunks - 1,
                grad_z_final_is_none and chunk_idx == num_chunks - 1,
                chunk_idx,
                chunk_start,
                chunk_len,
                T,
                h,
                d,
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
        return (*result, None, None, None)


def fused_delta_linoss(q, k, v, beta, y0, z0, A, B, dt, chunk_size):
    scale = k.shape[-1] ** -0.5
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    y0 = y0.contiguous()
    z0 = z0.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    return _ParallelRNNTriton.apply(q, k, v, beta, y0, z0, A, B, dt, chunk_size, scale)
