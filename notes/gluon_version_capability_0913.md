# Triton Gluon Hopper capability audit (v3.4.0 vs v3.6.0)

Scope: exact official GitHub tags only. For each tag, this audit read the recursive tree once and the matched Hopper package entry file once. It did not inspect implementations in `mbarrier.py` or `tma.py`, and it makes no claim about the P1 remote runtime version.

| Version | Hopper Gluon paths confirmed by tag tree | Hopper package entry-point evidence |
|---|---|---|
| `v3.4.0` | `python/triton/experimental/gluon/language/nvidia/hopper/{__init__.py,mbarrier.py,tma.py}` and `python/triton/experimental/gluon/nvidia/hopper.py` | `__all__ = ["fence_async_shared", "mbarrier", "tma"]`. The entry file contains no `warpgroup_mma`, `warpgroup_mma_wait`, or `warpgroup_mma_init` definition/import. Therefore explicit Gluon WGMMA is absent from this Hopper language entry point. |
| `v3.6.0` | The same five paths | `__all__` adds `warpgroup_mma` and `warpgroup_mma_wait`; the entry file also defines `warpgroup_mma_init` (although it is omitted from `__all__`). Explicit imports from the module are therefore available for all three names. |

Exact `v3.6.0` definitions:

- `warpgroup_mma_init(value, _semantic)`
- `warpgroup_mma(a, b, acc, *, use_acc=True, precision=None, max_num_imprecise_acc=None, is_async=False, _semantic=None)`
- `warpgroup_mma_wait(num_outstanding=0, deps=None, _semantic=None)`; `deps=None` raises `ValueError`.
- `fence_async_shared(cluster=False, _semantic=None)` exists in both tag entry files.

Both trees prove the presence of separate `mbarrier.py` and `tma.py` modules, but this bounded read does not establish their per-function signatures. Neither inspected `__init__.py` contains a Python-side SM90 capability check; the `hopper` namespace and WGMMA implementation placement are not by themselves proof that a particular remote environment targets or accepts SM90.

`main` is deliberately excluded from version conclusions. Any API or tutorial observed on `main` is later-state evidence and must not be attributed to either tag.

Official sources: [v3.4.0 tree](https://api.github.com/repos/triton-lang/triton/git/trees/v3.4.0?recursive=1), [v3.4.0 Hopper entry](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/experimental/gluon/language/nvidia/hopper/__init__.py), [v3.6.0 tree](https://api.github.com/repos/triton-lang/triton/git/trees/v3.6.0?recursive=1), [v3.6.0 Hopper entry](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/experimental/gluon/language/nvidia/hopper/__init__.py).
