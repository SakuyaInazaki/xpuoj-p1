"""CPU bit-exact checks for the proposed four-lane FP6 SWAR decoder.

Primary references:
- Triton v3.4.0 core.py:2720-2734, 2782-2791 (pack contract/example)
- Triton v3.4.0 ElementwiseOpToLLVM.cpp:204-303 (i8 input/output packing)
- Triton v3.4.0 semantic.py:46-58, 375-386, 414-423 (integer promotion/shifts)
- NVIDIA PTX ISA, prmt: 9.7.9.7 (generic selector and sign replication)

This is a CPU/ABI reference only.  It does not compile or run a Triton kernel.
"""

from itertools import permutations
import random


U32 = 0xFFFFFFFF
SMALL_A = 0x5C585000
SMALL_B = 0x66646260


def pack_bytes(values):
    assert len(values) == 4
    return sum((value & 0xFF) << (8 * lane) for lane, value in enumerate(values))


def unpack_bytes(word):
    return tuple((word >> (8 * lane)) & 0xFF for lane in range(4))


def prmt(a, b, selector):
    """Generic prmt.b32 semantics, including selector-msb sign replication."""
    sources = unpack_bytes(a) + unpack_bytes(b)
    out = 0
    for lane in range(4):
        control = (selector >> (4 * lane)) & 0xF
        byte = sources[control & 7]
        if control & 8:
            byte = 0xFF if byte & 0x80 else 0
        out |= byte << (8 * lane)
    return out


def canonical_decode(code):
    assert 0 <= code < 64
    magnitude = code & 31
    small = (0, 80, 88, 92, 96, 98, 100, 102)
    raw_magnitude = small[magnitude] if magnitude < 8 else magnitude + 96
    return raw_magnitude | ((code & 32) << 2)


def selector_expanded(a):
    return ((a & 7) | ((a >> 4) & 0x70) | ((a >> 8) & 0x700)
            | ((a >> 12) & 0x7000))


def selector_compressed(a):
    selector = a & 0x07070707
    selector = (selector | (selector >> 4)) & 0x00770077
    return (selector | (selector >> 8)) & 0x00007777


def swar_decode(word, compressed_selector=False):
    a = word & 0x1F1F1F1F
    selector = selector_compressed(a) if compressed_selector else selector_expanded(a)
    small = prmt(SMALL_A, SMALL_B, selector)
    large = (a + 0x60606060) & U32  # Each lane is <=31+96, so no byte carry.
    flags = (((a | (a >> 1)) & 0x08080808) << 4) & U32
    mask = prmt(flags, 0, 0xBA98)
    magnitude = small ^ ((small ^ large) & mask)
    return (magnitude | ((word & 0x20202020) << 2)) & U32


def expected_word(codes):
    return pack_bytes([canonical_decode(code) for code in codes])


def check_word(codes):
    word = pack_bytes(codes)
    expected = expected_word(codes)
    assert swar_decode(word) == expected, (codes, hex(swar_decode(word)), hex(expected))
    assert swar_decode(word, compressed_selector=True) == expected


def main():
    # 0xBA98 selects each byte from the first source with sign replication.
    for bits in range(16):
        flags = pack_bytes([0x80 if bits & (1 << lane) else 0 for lane in range(4)])
        expected_mask = pack_bytes([0xFF if bits & (1 << lane) else 0 for lane in range(4)])
        assert prmt(flags, 0, 0xBA98) == expected_mask

    # Each of the 64 codes is checked independently in every packed byte position.
    sentinels = (0, 7, 8, 31)
    for code in range(64):
        for lane in range(4):
            codes = list(sentinels)
            codes[lane] = code
            check_word(codes)

    # Positive/negative zero and NaN payloads, plus branch boundaries.
    boundary_words = (
        (0, 32, 31, 63),
        (7, 8, 39, 40),
        (30, 31, 62, 63),
        (0, 7, 8, 31),
        (32, 39, 40, 63),
    )
    for codes in boundary_words:
        check_word(codes)
        for permuted in permutations(codes):
            check_word(permuted)
    assert unpack_bytes(expected_word((0, 32, 31, 63))) == (0x00, 0x80, 0x7F, 0xFF)

    rng = random.Random(0xF6A4)
    random_words = 200_000
    for _ in range(random_words):
        check_word(tuple(rng.randrange(64) for _ in range(4)))

    print(
        "PASS: 64 codes x 4 byte positions; 0xBA98 masks; zero/NaN/boundaries; "
        f"{random_words} random words; both selector constructions"
    )


if __name__ == "__main__":
    main()
