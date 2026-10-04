"""PRMT FP6 decode draft; CPU proof only, with no GPU JIT claim.

PTX 8.1 §9.7.8.7: generic selector nibbles choose source bytes 0..7;
c[3:0] controls d.b0. Triton v3.4 core.py says each asm invocation
processes `pack` elements, so pack=1 gives one code per PRMT invocation.
"""

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None


if triton is not None:
    @triton.jit
    def fp6_decode_prmt(code):
        abs_code = (code & 31).to(tl.uint32)
        table_a = tl.full(code.shape, 0x5c585000, tl.uint32)
        table_b = tl.full(code.shape, 0x66646260, tl.uint32)
        small = tl.inline_asm_elementwise(
            "prmt.b32 $0, $1, $2, $3;", "=r,r,r,r",
            [table_a, table_b, abs_code & 7],
            dtype=tl.uint32, is_pure=True, pack=1,
        )
        mag = tl.where(abs_code >= 8, abs_code + 96, small)
        raw = (((code.to(tl.uint32) & 32) << 2) | mag).to(tl.uint8)
        return raw.to(tl.float8e4nv, bitcast=True)


def prmt_b32(a, b, selector):
    source = a.to_bytes(4, "little") + b.to_bytes(4, "little")
    out = 0
    for byte in range(4):
        nibble = (selector >> (4 * byte)) & 15
        value = source[nibble & 7]
        if nibble & 8:
            value = 0xff if value & 0x80 else 0
        out |= value << (8 * byte)
    return out


def decode_prmt(code):
    abs_code = code & 31
    small = prmt_b32(0x5c585000, 0x66646260, abs_code & 7)
    mag = abs_code + 96 if abs_code >= 8 else small
    return ((code & 32) << 2) | mag


def decode_canonical(code):
    low = (0, 80, 88, 92, 96, 98, 100, 102)
    abs_code = code & 31
    mag = low[abs_code] if abs_code < 8 else abs_code + 96
    return ((code & 32) << 2) | mag


def main():
    got = [decode_prmt(code) for code in range(64)]
    expected = [decode_canonical(code) for code in range(64)]
    assert got == expected
    assert got[31] == 0x7f and got[63] == 0xff
    assert [prmt_b32(0x5c585000, 0x66646260, i) for i in range(8)] == [
        0, 80, 88, 92, 96, 98, 100, 102,
    ]
    print("PASS: PRMT selector simulation matches canonical decode for all 64 FP6 codes")


if __name__ == "__main__":
    main()
