import torch
import triton
import triton.language as tl


def _launch_config(d):
    block_d = min(256, triton.next_power_of_2(d))
    if block_d >= 256:
        num_warps = 4
    elif block_d >= 128:
        num_warps = 2
    else:
        num_warps = 1
    return block_d, num_warps


# Pass 1: compute only last-step bias per chunk (for inter-chunk scan input)
@triton.jit
def chunk_last_bias_kernel(
    u_ptr,           # [B, nc, cs, d]
    last_b0_ptr,     # [B, nc, d] output
    last_b1_ptr,
    m00_ptr, m01_ptr, m10_ptr, m11_ptr,  # [d]
    dt, dt2,
    cs, d: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bc = tl.program_id(0)  # indexes (B * nc)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offs < d
    base = pid_bc * cs * d

    m00v = tl.load(m00_ptr + d_offs, mask=mask)
    m01v = tl.load(m01_ptr + d_offs, mask=mask)
    m10v = tl.load(m10_ptr + d_offs, mask=mask)
    m11v = tl.load(m11_ptr + d_offs, mask=mask)

    rb0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    rb1 = tl.zeros([BLOCK_D], dtype=tl.float32)

    for s in range(cs):
        uv = tl.load(u_ptr + base + s * d + d_offs, mask=mask)
        nb0 = m00v * rb0 + m01v * rb1 + dt2 * uv
        nb1 = m10v * rb0 + m11v * rb1 + dt  * uv
        rb0 = nb0
        rb1 = nb1

    tl.store(last_b0_ptr + pid_bc * d + d_offs, rb0, mask=mask)
    tl.store(last_b1_ptr + pid_bc * d + d_offs, rb1, mask=mask)


# Inter-chunk prefix scan: one program per batch element
@triton.jit
def inter_chunk_scan_kernel(
    last_b0_ptr,   # [B, nc, d]
    last_b1_ptr,
    chunk_y_ptr,   # [B, nc, d] output
    chunk_z_ptr,
    y0_ptr,        # [B, d]
    z0_ptr,
    cm00_ptr, cm01_ptr, cm10_ptr, cm11_ptr,  # [d]
    nc, d: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offs < d

    y0 = tl.load(y0_ptr + b * d + d_offs, mask=mask)
    z0 = tl.load(z0_ptr + b * d + d_offs, mask=mask)
    cm00 = tl.load(cm00_ptr + d_offs, mask=mask)
    cm01 = tl.load(cm01_ptr + d_offs, mask=mask)
    cm10 = tl.load(cm10_ptr + d_offs, mask=mask)
    cm11 = tl.load(cm11_ptr + d_offs, mask=mask)

    pm00 = tl.full([BLOCK_D], 1.0, dtype=tl.float32)
    pm01 = tl.zeros([BLOCK_D], dtype=tl.float32)
    pm10 = tl.zeros([BLOCK_D], dtype=tl.float32)
    pm11 = tl.full([BLOCK_D], 1.0, dtype=tl.float32)
    pb0  = tl.zeros([BLOCK_D], dtype=tl.float32)
    pb1  = tl.zeros([BLOCK_D], dtype=tl.float32)

    for c in range(nc):
        cy = pm00 * y0 + pm01 * z0 + pb0
        cz = pm10 * y0 + pm11 * z0 + pb1
        tl.store(chunk_y_ptr + (b * nc + c) * d + d_offs, cy, mask=mask)
        tl.store(chunk_z_ptr + (b * nc + c) * d + d_offs, cz, mask=mask)

        cc0 = tl.load(last_b0_ptr + (b * nc + c) * d + d_offs, mask=mask)
        cc1 = tl.load(last_b1_ptr + (b * nc + c) * d + d_offs, mask=mask)

        nm00 = cm00 * pm00 + cm01 * pm10
        nm01 = cm00 * pm01 + cm01 * pm11
        nm10 = cm10 * pm00 + cm11 * pm10
        nm11 = cm10 * pm01 + cm11 * pm11
        nb0  = cm00 * pb0  + cm01 * pb1 + cc0
        nb1  = cm10 * pb0  + cm11 * pb1 + cc1

        pm00 = nm00; pm01 = nm01
        pm10 = nm10; pm11 = nm11
        pb0 = nb0; pb1 = nb1


# Pass 2: fused intra-chunk bias recompute + output expansion
# Compute M^(s+1) iteratively instead of loading from table
@triton.jit
def fused_expand_kernel(
    u_ptr,           # [B, nc, cs, d]
    chunk_y_ptr,     # [B, nc, d]
    chunk_z_ptr,
    m00_ptr, m01_ptr, m10_ptr, m11_ptr,  # [d]
    y_out_ptr,       # [B, nc, cs, d]
    z_out_ptr,
    dt, dt2,
    nc, cs, d: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_d = tl.program_id(1)
    b = pid_bc // nc
    c = pid_bc % nc
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offs < d

    m00v = tl.load(m00_ptr + d_offs, mask=mask)
    m01v = tl.load(m01_ptr + d_offs, mask=mask)
    m10v = tl.load(m10_ptr + d_offs, mask=mask)
    m11v = tl.load(m11_ptr + d_offs, mask=mask)

    cy = tl.load(chunk_y_ptr + (b * nc + c) * d + d_offs, mask=mask)
    cz = tl.load(chunk_z_ptr + (b * nc + c) * d + d_offs, mask=mask)

    base = (b * nc + c) * cs * d
    rb0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    rb1 = tl.zeros([BLOCK_D], dtype=tl.float32)

    # M^s matrix (starts at M^1 = M after first step)
    pm00 = m00v; pm01 = m01v; pm10 = m10v; pm11 = m11v

    for s in range(cs):
        uv = tl.load(u_ptr + base + s * d + d_offs, mask=mask)
        nb0 = m00v * rb0 + m01v * rb1 + dt2 * uv
        nb1 = m10v * rb0 + m11v * rb1 + dt  * uv
        rb0 = nb0
        rb1 = nb1

        # pm** = M^(s+1) at this point (initialized to M^1 = M, updated each step)
        y = pm00 * cy + pm01 * cz + rb0
        z = pm10 * cy + pm11 * cz + rb1

        tl.store(y_out_ptr + base + s * d + d_offs, y, mask=mask)
        tl.store(z_out_ptr + base + s * d + d_offs, z, mask=mask)

        # Update pm** = M^(s+2) = M @ M^(s+1)
        npm00 = m00v * pm00 + m01v * pm10
        npm01 = m00v * pm01 + m01v * pm11
        npm10 = m10v * pm00 + m11v * pm10
        npm11 = m10v * pm01 + m11v * pm11
        pm00 = npm00; pm01 = npm01; pm10 = npm10; pm11 = npm11

@triton.jit
def _compute_matrix_power_kernel(
    a_ptr, b_ptr,
    m00_ptr, m01_ptr, m10_ptr, m11_ptr,
    cm00_ptr, cm01_ptr, cm10_ptr, cm11_ptr,
    dt, d: tl.constexpr, cs: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offs < d
    a = tl.load(a_ptr + d_offs, mask=mask)
    bv = tl.load(b_ptr + d_offs, mask=mask)

    dt2 = dt * dt
    m00 = 1.0 - dt2 * a
    m01 = dt * (1.0 - dt * bv)
    m10 = -dt * a
    m11 = 1.0 - dt * bv

    tl.store(m00_ptr + d_offs, m00, mask=mask)
    tl.store(m01_ptr + d_offs, m01, mask=mask)
    tl.store(m10_ptr + d_offs, m10, mask=mask)
    tl.store(m11_ptr + d_offs, m11, mask=mask)

    # Compute M^cs
    c00 = tl.full([BLOCK_D], 1.0, dtype=tl.float32)
    c01 = tl.zeros([BLOCK_D], dtype=tl.float32)
    c10 = tl.zeros([BLOCK_D], dtype=tl.float32)
    c11 = tl.full([BLOCK_D], 1.0, dtype=tl.float32)
    for _ in range(cs):
        n00 = c00 * m00 + c01 * m10
        n01 = c00 * m01 + c01 * m11
        n10 = c10 * m00 + c11 * m10
        n11 = c10 * m01 + c11 * m11
        c00 = n00; c01 = n01; c10 = n10; c11 = n11

    tl.store(cm00_ptr + d_offs, c00, mask=mask)
    tl.store(cm01_ptr + d_offs, c01, mask=mask)
    tl.store(cm10_ptr + d_offs, c10, mask=mask)
    tl.store(cm11_ptr + d_offs, c11, mask=mask)


def _triton_forward(u, y0, z0, A, B, dt, chunk_size):
    batch_size, T, d = u.shape
    device = u.device
    dtype = torch.float32

    dt_f = float(dt)
    dt2_f = float(dt * dt)

    num_chunks = (T + chunk_size - 1) // chunk_size
    T_pad = num_chunks * chunk_size

    if T_pad > T:
        pad = torch.zeros(batch_size, T_pad - T, d, device=device, dtype=u.dtype)
        u_pad = torch.cat([u, pad], dim=1)
    else:
        u_pad = u

    u_chunks = u_pad.view(batch_size, num_chunks, chunk_size, d).contiguous()

    BLOCK_D, num_warps = _launch_config(d)
    num_d_tiles = triton.cdiv(d, BLOCK_D)

    # Pack M and M^cs into single buffer [8, d]
    mat_buf = torch.empty(8, d, device=device, dtype=dtype)
    m00, m01, m10, m11 = mat_buf[0], mat_buf[1], mat_buf[2], mat_buf[3]
    cm00, cm01, cm10, cm11 = mat_buf[4], mat_buf[5], mat_buf[6], mat_buf[7]

    _compute_matrix_power_kernel[(num_d_tiles,)](
        A, B, m00, m01, m10, m11, cm00, cm01, cm10, cm11,
        dt_f, d, chunk_size, BLOCK_D=BLOCK_D,
    )
    grid = (batch_size * num_chunks, num_d_tiles)

    # Pack last_b and chunk state into single buffer [4, B, nc, d]
    work_buf = torch.empty(4, batch_size, num_chunks, d, device=device, dtype=dtype)
    last_b0, last_b1 = work_buf[0], work_buf[1]
    chunk_y, chunk_z = work_buf[2], work_buf[3]

    chunk_last_bias_kernel[grid](
        u_chunks, last_b0, last_b1,
        m00, m01, m10, m11,
        dt_f, dt2_f,
        chunk_size, d,
        BLOCK_D=BLOCK_D, num_warps=num_warps,
    )

    inter_chunk_scan_kernel[(batch_size, num_d_tiles)](
        last_b0, last_b1,
        chunk_y, chunk_z,
        y0, z0,
        cm00, cm01, cm10, cm11,
        num_chunks, d,
        BLOCK_D=BLOCK_D, num_warps=num_warps,
    )

    # Pass 2: fused intra-chunk recompute + expansion
    out_buf = torch.empty(2, batch_size, num_chunks, chunk_size, d, device=device, dtype=dtype)
    y_out, z_out = out_buf[0], out_buf[1]

    fused_expand_kernel[grid](
        u_chunks, chunk_y, chunk_z,
        m00, m01, m10, m11,
        y_out, z_out,
        dt_f, dt2_f,
        num_chunks, chunk_size, d,
        BLOCK_D=BLOCK_D, num_warps=num_warps,
    )

    y_out = y_out.reshape(batch_size, T_pad, d)[:, :T, :]
    z_out = z_out.reshape(batch_size, T_pad, d)[:, :T, :]
    return y_out, z_out, mat_buf


# ============ Chunked backward pass ============

# Backward pass 1: compute gradient bias per chunk (reverse scan within chunk)
# The backward recurrence uses M^T: mt00=m00, mt01=m10, mt10=m01, mt11=m11
@triton.jit
def chunk_last_bias_bwd_kernel(
    grad_y_ptr,      # [B, nc, cs, d]
    grad_z_ptr,
    last_gb0_ptr,    # [B, nc, d] output
    last_gb1_ptr,
    a_ptr, b_ptr,    # [d]
    dt,
    cs, d: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bc = tl.program_id(0)  # indexes (B * nc)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offs < d
    base = pid_bc * cs * d

    a = tl.load(a_ptr + d_offs, mask=mask)
    bv = tl.load(b_ptr + d_offs, mask=mask)
    dt2 = dt * dt
    # M^T entries
    mt00 = 1.0 - dt2 * a
    mt01 = -dt * a
    mt10 = dt * (1.0 - dt * bv)
    mt11 = 1.0 - dt * bv

    rb0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    rb1 = tl.zeros([BLOCK_D], dtype=tl.float32)

    for s_rev in range(cs):
        s = cs - 1 - s_rev
        off = base + s * d + d_offs
        gy = tl.load(grad_y_ptr + off, mask=mask)
        gz = tl.load(grad_z_ptr + off, mask=mask)

        gy_total = gy + rb0
        gz_total = gz + rb1
        nb0 = mt00 * gy_total + mt01 * gz_total
        nb1 = mt10 * gy_total + mt11 * gz_total
        rb0 = nb0
        rb1 = nb1

    tl.store(last_gb0_ptr + pid_bc * d + d_offs, rb0, mask=mask)
    tl.store(last_gb1_ptr + pid_bc * d + d_offs, rb1, mask=mask)


# Backward inter-chunk scan: propagate gradients across chunks in reverse
@triton.jit
def inter_chunk_scan_bwd_kernel(
    last_gb0_ptr,   # [B, nc, d]
    last_gb1_ptr,
    chunk_gy_ptr,   # [B, nc, d] output
    chunk_gz_ptr,
    grad_y0_ptr,    # [B, d] output
    grad_z0_ptr,
    cm00_ptr, cm01_ptr, cm10_ptr, cm11_ptr,  # [d] -- M^cs (forward)
    nc, d: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offs < d

    # (M^T)^cs = (M^cs)^T: swap cm01 and cm10
    cmt00 = tl.load(cm00_ptr + d_offs, mask=mask)
    cmt01 = tl.load(cm10_ptr + d_offs, mask=mask)  # transposed
    cmt10 = tl.load(cm01_ptr + d_offs, mask=mask)  # transposed
    cmt11 = tl.load(cm11_ptr + d_offs, mask=mask)

    pb0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    pb1 = tl.zeros([BLOCK_D], dtype=tl.float32)

    for c_rev in range(nc):
        c = nc - 1 - c_rev
        tl.store(chunk_gy_ptr + (b * nc + c) * d + d_offs, pb0, mask=mask)
        tl.store(chunk_gz_ptr + (b * nc + c) * d + d_offs, pb1, mask=mask)

        cc0 = tl.load(last_gb0_ptr + (b * nc + c) * d + d_offs, mask=mask)
        cc1 = tl.load(last_gb1_ptr + (b * nc + c) * d + d_offs, mask=mask)

        nb0 = cmt00 * pb0 + cmt01 * pb1 + cc0
        nb1 = cmt10 * pb0 + cmt11 * pb1 + cc1
        pb0 = nb0
        pb1 = nb1

    # After processing all chunks, pb0/pb1 = grad w.r.t. y0/z0
    tl.store(grad_y0_ptr + b * d + d_offs, pb0, mask=mask)
    tl.store(grad_z0_ptr + b * d + d_offs, pb1, mask=mask)


# Backward pass 2: fused intra-chunk expansion + grad_u/grad_A/grad_B computation
@triton.jit
def fused_expand_bwd_kernel(
    grad_y_ptr,      # [B, nc, cs, d]
    grad_z_ptr,
    y_out_ptr,       # [B, nc, cs, d]
    z_out_ptr,
    y0_ptr,          # [B, d]
    z0_ptr,
    chunk_gy_ptr,    # [B, nc, d]
    chunk_gz_ptr,
    a_ptr, b_ptr,    # [d]
    grad_u_ptr,      # [B, nc, cs, d] output
    grad_a_ptr,      # [B, nc, d] output (partial sum within chunk)
    grad_b_ptr,      # [B, nc, d] output
    dt,
    nc, cs, d: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_d = tl.program_id(1)
    b = pid_bc // nc
    c = pid_bc % nc
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offs < d

    a = tl.load(a_ptr + d_offs, mask=mask)
    bv = tl.load(b_ptr + d_offs, mask=mask)

    # Load chunk initial gradient (from inter-chunk scan)
    gy_next = tl.load(chunk_gy_ptr + (b * nc + c) * d + d_offs, mask=mask)
    gz_next = tl.load(chunk_gz_ptr + (b * nc + c) * d + d_offs, mask=mask)

    grad_a_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    grad_b_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    base = (b * nc + c) * cs * d

    for s_rev in range(cs):
        s = cs - 1 - s_rev
        off = base + s * d + d_offs

        gy = tl.load(grad_y_ptr + off, mask=mask) + gy_next
        gz = tl.load(grad_z_ptr + off, mask=mask) + gz_next

        gz_combined = gz + dt * gy
        gy_next = gy - dt * a * gz_combined
        gz_next = (1.0 - dt * bv) * gz_combined

        tl.store(grad_u_ptr + off, dt * gz_combined, mask=mask)

        # Load y_prev, z_prev for grad_A/grad_B
        if s > 0:
            prev_off = base + (s - 1) * d + d_offs
            y_prev = tl.load(y_out_ptr + prev_off, mask=mask)
            z_prev = tl.load(z_out_ptr + prev_off, mask=mask)
        elif c > 0:
            prev_off = (b * nc + c - 1) * cs * d + (cs - 1) * d + d_offs
            y_prev = tl.load(y_out_ptr + prev_off, mask=mask)
            z_prev = tl.load(z_out_ptr + prev_off, mask=mask)
        else:
            y_prev = tl.load(y0_ptr + b * d + d_offs, mask=mask)
            z_prev = tl.load(z0_ptr + b * d + d_offs, mask=mask)

        grad_a_acc += (-dt * y_prev) * gz_combined
        grad_b_acc += (-dt * z_prev) * gz_combined

    tl.store(grad_a_ptr + (b * nc + c) * d + d_offs, grad_a_acc, mask=mask)
    tl.store(grad_b_ptr + (b * nc + c) * d + d_offs, grad_b_acc, mask=mask)


def _triton_backward(y0, z0, A, B, dt, chunk_size, mat_buf, y_out, z_out, grad_y_out, grad_z_out, u_dtype):
    batch_size, T, d = y_out.shape
    BLOCK_D, num_warps = _launch_config(d)
    num_d_tiles = triton.cdiv(d, BLOCK_D)
    device = y_out.device
    dtype = y_out.dtype
    dt_f = float(dt)

    num_chunks = (T + chunk_size - 1) // chunk_size
    T_pad = num_chunks * chunk_size

    # Reshape grads to chunked layout
    grad_y_out = grad_y_out.contiguous()
    grad_z_out = grad_z_out.contiguous()
    if T_pad > T:
        pad = torch.zeros(batch_size, T_pad - T, d, device=device, dtype=dtype)
        grad_y_out = torch.cat([grad_y_out, pad], dim=1)
        grad_z_out = torch.cat([grad_z_out, pad], dim=1)
    grad_y_chunks = grad_y_out.view(batch_size, num_chunks, chunk_size, d)
    grad_z_chunks = grad_z_out.view(batch_size, num_chunks, chunk_size, d)

    # Reshape y_out, z_out to chunked layout
    y_out_c = y_out.contiguous()
    z_out_c = z_out.contiguous()
    if T_pad > T:
        pad_yz = torch.zeros(batch_size, T_pad - T, d, device=device, dtype=dtype)
        y_out_c = torch.cat([y_out_c, pad_yz], dim=1)
        z_out_c = torch.cat([z_out_c, pad_yz], dim=1)
    y_out_chunks = y_out_c.view(batch_size, num_chunks, chunk_size, d)
    z_out_chunks = z_out_c.view(batch_size, num_chunks, chunk_size, d)

    grid = (batch_size * num_chunks, num_d_tiles)

    # Reuse M^cs from forward
    cm00, cm01, cm10, cm11 = mat_buf[4], mat_buf[5], mat_buf[6], mat_buf[7]

    # Pass 1: chunk-level backward bias + inter-chunk scan workspace
    work_buf = torch.empty(4, batch_size, num_chunks, d, device=device, dtype=dtype)
    last_gb0, last_gb1 = work_buf[0], work_buf[1]
    chunk_gy, chunk_gz = work_buf[2], work_buf[3]

    chunk_last_bias_bwd_kernel[grid](
        grad_y_chunks, grad_z_chunks,
        last_gb0, last_gb1,
        A, B, dt_f,
        chunk_size, d, BLOCK_D=BLOCK_D, num_warps=num_warps,
    )

    # Inter-chunk backward scan
    grad_y0 = torch.empty(batch_size, d, device=device, dtype=dtype)
    grad_z0 = torch.empty(batch_size, d, device=device, dtype=dtype)

    inter_chunk_scan_bwd_kernel[(batch_size, num_d_tiles)](
        last_gb0, last_gb1,
        chunk_gy, chunk_gz,
        grad_y0, grad_z0,
        cm00, cm01, cm10, cm11,
        num_chunks, d, BLOCK_D=BLOCK_D, num_warps=num_warps,
    )

    # Pass 2: fused intra-chunk expansion
    grad_u_chunks = torch.empty(batch_size, num_chunks, chunk_size, d, device=device, dtype=u_dtype)
    grad_ab_buf = torch.empty(2, batch_size, num_chunks, d, device=device, dtype=dtype)
    grad_a_partial, grad_b_partial = grad_ab_buf[0], grad_ab_buf[1]

    fused_expand_bwd_kernel[grid](
        grad_y_chunks, grad_z_chunks,
        y_out_chunks, z_out_chunks,
        y0, z0,
        chunk_gy, chunk_gz,
        A, B,
        grad_u_chunks, grad_a_partial, grad_b_partial,
        dt_f,
        num_chunks, chunk_size, d, BLOCK_D=BLOCK_D, num_warps=num_warps,
    )

    grad_u = grad_u_chunks.reshape(batch_size, T_pad, d)[:, :T, :]
    grad_A = grad_a_partial.sum(dim=(0, 1))
    grad_B = grad_b_partial.sum(dim=(0, 1))
    return grad_u, grad_y0, grad_z0, grad_A, grad_B


class Kernel_RNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, y0, z0, A, B, dt, chunk_size):
        y, z, mat_buf = _triton_forward(u, y0, z0, A, B, dt, chunk_size)
        ctx.save_for_backward(y0, z0, A, B, y, z, mat_buf)
        ctx.dt = dt
        ctx.chunk_size = chunk_size
        ctx.u_dtype = u.dtype
        return y, z

    @staticmethod
    def backward(ctx, grad_y_out, grad_z_out):
        y0, z0, A, B, y_out, z_out, mat_buf = ctx.saved_tensors
        dt = ctx.dt
        chunk_size = ctx.chunk_size
        u_dtype = ctx.u_dtype

        grad_u, grad_y0, grad_z0, grad_A, grad_B = _triton_backward(
            y0, z0, A, B, dt, chunk_size, mat_buf,
            y_out, z_out, grad_y_out, grad_z_out, u_dtype,
        )
        return (
            grad_u,
            grad_y0,
            grad_z0,
            grad_A,
            grad_B,
            None,
            None,
        )

def fused_chunk_linoss(q, k, v, beta, y0, z0, A, B, dt, chunk_size):
    # TODO: fuse the einsums into the kernel
    batch_size, T, h, d = q.shape
    orig_dtype = q.dtype
    scale = k.shape[-1] ** -0.5
    q = q * scale

    beta_scale = beta.unsqueeze(-1)

    u = torch.einsum('bthd,bthe->bthde', k, v)
    u = u * beta_scale.unsqueeze(-1)

    u = u.reshape(batch_size, T, h * d * d)

    y0 = y0.reshape(batch_size, h * d * d)
    z0 = z0.reshape(batch_size, h * d * d)
    A = A.reshape(h * d * d)
    B = B.reshape(h * d * d)
    y, z = Kernel_RNN.apply(u, y0, z0, A, B, dt, chunk_size)
    y = y.view(batch_size, T, h, d, d)
    z = z.view(batch_size, T, h, d, d)
    output = torch.einsum('bthd, bthde -> bthe', q, y)
    y_last = y[:, -1]
    z_last = z[:, -1]
    return output, y_last, z_last
