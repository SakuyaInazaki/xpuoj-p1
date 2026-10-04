"""CPU reference plus a draft Triton packer for row-major FP6 weight planes."""

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU verification does not require the judge runtime.
    triton = None


def encode_raw(raw):
    raw = raw.to(torch.uint8)
    mag = raw & 0x7f
    low = torch.where(mag <= 72, 0,
          torch.where(mag < 84, 1,
          torch.where(mag <= 90, 2,
          torch.where(mag < 94, 3,
          torch.where(mag <= 97, 4,
          torch.where(mag < 99, 5,
          torch.where(mag <= 101, 6,
          torch.where(mag < 103, 7, 8)))))))).to(torch.uint8)
    return ((raw & 0x80) >> 2) | torch.where(mag >= 104, mag - 96, low)


def decode_raw(code):
    code = code.to(torch.uint8)
    ab = code & 31
    low_raw = torch.tensor([0, 80, 88, 92, 96, 98, 100, 102], dtype=torch.uint8)
    mag = torch.where(ab < 8, low_raw[torch.clamp(ab.long(), max=7)], ab + 96)
    return ((code & 0x20) << 2) | mag


def pack_raw(raw):
    assert raw.shape[-1] % 4 == 0
    code = encode_raw(raw)
    lo = code & 15
    hi = code >> 4
    q4 = (lo[..., 0::2] | (lo[..., 1::2] << 4)).contiguous()
    q2 = (hi[..., 0::4] | (hi[..., 1::4] << 2)
          | (hi[..., 2::4] << 4) | (hi[..., 3::4] << 6)).contiguous()
    return q4, q2


def unpack_raw(q4, q2):
    assert q4.shape[-1] == 2 * q2.shape[-1]
    lo = torch.stack((q4 & 15, q4 >> 4), dim=-1).reshape(*q4.shape[:-1], -1)
    hi = torch.stack(tuple((q2 >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1)
    code = lo | (hi.reshape(*q2.shape[:-1], -1) << 4)
    return decode_raw(code)


if triton is not None:
    @triton.jit
    def _fp6_code(raw):
        mag = raw & 0x7f
        low = tl.where(mag <= 72, 0,
              tl.where(mag < 84, 1,
              tl.where(mag <= 90, 2,
              tl.where(mag < 94, 3,
              tl.where(mag <= 97, 4,
              tl.where(mag < 99, 5,
              tl.where(mag <= 101, 6,
              tl.where(mag < 103, 7, 8))))))))
        return ((raw & 0x80) >> 2) | tl.where(mag >= 104, mag - 96, low)


    @triton.jit
    def fp6_pack_two_plane_kernel(Q, Q4, Q2, GROUPS, BLOCK: tl.constexpr):
        group = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = group < GROUPS
        base = group * 4
        c0 = _fp6_code(tl.load(Q + base, mask=mask, other=0).to(tl.int32))
        c1 = _fp6_code(tl.load(Q + base + 1, mask=mask, other=0).to(tl.int32))
        c2 = _fp6_code(tl.load(Q + base + 2, mask=mask, other=0).to(tl.int32))
        c3 = _fp6_code(tl.load(Q + base + 3, mask=mask, other=0).to(tl.int32))
        tl.store(Q4 + 2 * group, (c0 & 15) | ((c1 & 15) << 4), mask=mask)
        tl.store(Q4 + 2 * group + 1, (c2 & 15) | ((c3 & 15) << 4), mask=mask)
        tl.store(Q2 + group, (c0 >> 4) | ((c1 >> 4) << 2)
                 | ((c2 >> 4) << 4) | ((c3 >> 4) << 6), mask=mask)


def pack_fp6_weights(q_fp8):
    """Pack contiguous [E, 2I, K] FP8; caller retains its [E, 2I] FP32 scale."""
    if triton is None:
        raise RuntimeError("Triton runtime unavailable")
    assert q_fp8.dtype == torch.float8_e4m3fn and q_fp8.is_contiguous()
    E, N, K = q_fp8.shape
    assert K % 4 == 0
    b4 = torch.empty((E, N, K // 2), dtype=torch.uint8, device=q_fp8.device)
    b2 = torch.empty((E, N, K // 4), dtype=torch.uint8, device=q_fp8.device)
    groups = q_fp8.numel() // 4
    block = 1024
    fp6_pack_two_plane_kernel[(triton.cdiv(groups, block),)](
        q_fp8.view(torch.uint8), b4, b2, groups, BLOCK=block, num_warps=8)
    return b4, b2


def main():
    raw = torch.arange(256, dtype=torch.uint8)
    q4, q2 = pack_raw(raw)
    rebuilt = unpack_raw(q4, q2)
    expected = decode_raw(encode_raw(raw))
    assert torch.equal(rebuilt, expected)
    assert torch.equal(encode_raw(raw), encode_raw(expected))
    assert encode_raw(raw)[[0x7f, 0xff]].tolist() == [31, 63]
    assert rebuilt[[0x7f, 0xff]].tolist() == [0x7f, 0xff]

    codes = torch.tensor([1, 18, 35, 52], dtype=torch.uint8)
    order_q4, order_q2 = pack_raw(decode_raw(codes))
    assert order_q4.tolist() == [0x21, 0x43] and order_q2.tolist() == [0xe4]

    row = torch.arange(4096, dtype=torch.int64).to(torch.uint8)
    row_q4, row_q2 = pack_raw(row)
    row_code = encode_raw(row)
    assert (row_q4.numel(), row_q2.numel()) == (2048, 1024)
    assert row_q4.numel() + row_q2.numel() == 3 * row.numel() // 4
    assert torch.equal(unpack_raw(row_q4, row_q2), decode_raw(row_code))
    assert torch.equal(row_q4, row_code[0::2] & 15 | ((row_code[1::2] & 15) << 4))
    print("PASS: all 256 raw codes, K=4096 address order, q4/q2=2048/1024 bytes")


if __name__ == "__main__":
    main()
