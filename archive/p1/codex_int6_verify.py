"""CPU-only structural and bit-exact checks for codex_int6_c9_g64_bk64.py.

This does not compile Triton or claim GPU/JIT validation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "codex_int6_c9_g64_bk64.py"
DRAFT128 = HERE / "codex_int6_c9_g64_bk128_draft.py"


def fp8_first(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    wf = w.float()
    amax = wf.abs().amax(dim=2, keepdim=True)
    scale = torch.maximum(amax / 448.0, torch.full_like(amax, 1e-12))
    return (wf / scale).to(torch.float8_e4m3fn), scale.squeeze(2)


def accepted_q6round(q: torch.Tensor, group: int = 64) -> torch.Tensor:
    e, n, k = q.shape
    out = q.clone()
    for expert in range(e):
        w = q[expert].float().view(n, k // group, group)
        scale = w.abs().amax(dim=2, keepdim=True) / 31.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        rounded = torch.clamp(torch.round(w / scale), -31, 31) * scale
        out[expert].copy_(rounded.view(n, k).to(torch.float8_e4m3fn))
    return out


def pack_exact(q: torch.Tensor, group: int = 64):
    e, n, k = q.shape
    w = q.float().view(e, n, k // group, group)
    scale = w.abs().amax(dim=3, keepdim=True) / 31.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qi = torch.clamp(torch.round(w / scale), -31, 31)
    code = (qi.to(torch.int16) + 32).to(torch.uint8).view(e, n, k)
    hi, lo = code >> 2, code & 3
    hp = hi.view(e, n, k // 2, 2)
    lp = lo.view(e, n, k // 4, 4)
    q4 = hp[..., 0] | (hp[..., 1] << 4)
    q2 = lp[..., 0] | (lp[..., 1] << 2) | (lp[..., 2] << 4) | (lp[..., 3] << 6)
    return q4, q2, scale.squeeze(3)


def unpack_as_kernel(q4: torch.Tensor, q2: torch.Tensor, scale: torch.Tensor,
                     group: int = 64) -> torch.Tensor:
    e, n, k2 = q4.shape
    k = k2 * 2
    lane = torch.arange(k)
    # This is the kernel address/shift expression, rather than stack-based
    # inversion, so it checks every byte index and sub-field independently.
    hi = (q4[..., lane // 2] >> ((lane % 2) * 4)) & 15
    lo = (q2[..., lane // 4] >> ((lane % 4) * 2)) & 3
    code = hi.to(torch.int16) * 4 + lo.to(torch.int16)
    rebuilt = ((code - 32).float().view(e, n, k // group, group)
               * scale.float().unsqueeze(3))
    return rebuilt.view(e, n, k).to(torch.float8_e4m3fn)


def check_roundtrip() -> None:
    torch.manual_seed(20260905)
    # Include zero groups, endpoints, several rows/experts, and two G64 groups.
    w = torch.randn(3, 7, 128, dtype=torch.float32).to(torch.bfloat16)
    w[0, 0, :64] = 0
    w[1, 2, 64] = 50
    w[2, 6, 127] = -50
    q, _ = fp8_first(w)
    q4, q2, scale = pack_exact(q)
    got = unpack_as_kernel(q4, q2, scale)
    want = accepted_q6round(q)
    # int6 has one zero code, so it cannot retain the sign bit of FP8 -0.
    # The accepted probe can emit -0 after rounding a tiny negative value;
    # +0 is numerically exact and produces identical dot products.
    assert torch.equal(got.float(), want.float())
    byte_diff = got.view(torch.uint8) != want.view(torch.uint8)
    assert torch.all(got.float()[byte_diff] == 0)
    assert torch.all(want.float()[byte_diff] == 0)
    assert q4.shape == (3, 7, 64)
    assert q2.shape == (3, 7, 32)
    assert scale.shape == (3, 7, 2)


def check_c9_capacity() -> None:
    e, n2, k = 256, 4096, 4096
    dense = e * n2 * k
    q4 = dense // 2
    q2 = dense // 4
    scales = e * n2 * (k // 64) * 4
    packed = q4 + q2 + scales
    assert dense == 4 * 1024**3
    assert packed == int(3.25 * 1024**3)
    assert dense - packed == int(0.75 * 1024**3)
    assert packed / dense == 0.8125
    # Largest q4 linear element index is valid and int64 row bases cover the
    # 2^31-byte boundary exactly; q2 and scale planes are smaller.
    assert q4 == 2**31
    last_row = e * n2 - 1
    assert last_row * (k // 2) + (k // 2 - 1) == q4 - 1


def check_ast() -> None:
    source = CANDIDATE.read_text()
    tree = ast.parse(source)
    names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required = {
        "_pack_fp8_q6_exact", "_int6_source_key",
        "_get_full_int6_weights_lowmem", "_fgs_int6_g64_kernel",
        "_fgs_int6_g64_host", "run_kernel",
    }
    assert required <= names
    assert "data_ptr(" not in source
    assert "use_case9_int6 = (E == 256 and I == 2048 and H == 4096)" in source
    assert "(id(t), tuple(t.shape), tuple(t.stride())" in source
    assert "(cg - 32).to(tl.float32) * sg[:, None]).to(tl.float8e4nv)" in source
    assert "BLOCK_M=128, BLOCK_N=128, BLOCK_K=64" in source


def check_candidate_packer() -> None:
    """Execute the packer AST alone; importing the file needs judge-only libs."""
    tree = ast.parse(CANDIDATE.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_pack_fp8_q6_exact")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(CANDIDATE), "exec"), scope)
    torch.manual_seed(64)
    q = torch.randn(2, 5, 128).to(torch.float8_e4m3fn)
    q[0, 0, :64] = 0
    got4, got2, got_s, got_norm = scope["_pack_fp8_q6_exact"](q, 64)
    want4, want2, want_s = pack_exact(q, 64)
    assert torch.equal(got4, want4)
    assert torch.equal(got2, want2)
    assert torch.equal(got_s, want_s)
    rebuilt = unpack_as_kernel(got4, got2, got_s)
    assert torch.equal(rebuilt.float(), accepted_q6round(q).float())
    assert torch.equal(got_norm, torch.sqrt((rebuilt.float() ** 2).sum(dim=2)))


def check_bk128_draft() -> None:
    """Validate the packed-four PTX mask algebra and draft structure."""
    for h0 in range(256):
        for h1 in range(256):
            hp = h0 | (h0 << 8) | (h1 << 16) | (h1 << 24)
            hi = ((hp & 0x000F000F) << 2) | ((hp & 0xF000F000) >> 2)
            for q2 in (0, 1, 2, 3, 17, 85, 170, 255):
                lp = q2 * 0x01010101
                lo = ((lp & 0x00000003)
                      | ((lp & 0x00000C00) >> 2)
                      | ((lp & 0x00300000) >> 4)
                      | ((lp & 0xC0000000) >> 6))
                got = [(hi | lo) >> (8 * i) & 255 for i in range(4)]
                want = [
                    ((h0 & 15) << 2) | (q2 & 3),
                    ((h0 >> 4) << 2) | ((q2 >> 2) & 3),
                    ((h1 & 15) << 2) | ((q2 >> 4) & 3),
                    ((h1 >> 4) << 2) | ((q2 >> 6) & 3),
                ]
                assert got == want
    source = DRAFT128.read_text()
    ast.parse(source)
    assert "BLOCK_M=128, BLOCK_N=128, BLOCK_K=128" in source
    assert "row_g * (K // 64)" in source
    assert source.count("is_pure=True, pack=4") == 2
    assert source.count("tl.inline_asm_elementwise(") >= 2


if __name__ == "__main__":
    check_roundtrip()
    check_c9_capacity()
    check_ast()
    check_candidate_packer()
    check_bk128_draft()
    print("PASS: numerical-exact FP8->G64-int6->FP8 roundtrip, c9 layout/capacity, AST")
