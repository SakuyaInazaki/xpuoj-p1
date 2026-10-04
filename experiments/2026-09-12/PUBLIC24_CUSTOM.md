# Public problem 24 Triton H800 diagnostic channel

This directory contains a review-gated client for one self-authored, single-GPU
Triton diagnostic.  It does not validate P1 distributed communication.

Frozen request contract:

- Endpoint: `POST customTest/createCustomTest`
- Proof-of-work and captcha action: `custom_test`
- `problemId`: `24` (public **FP8 GeMM NT**)
- `mode`: `GeneratedWorkload`
- `content.language`: `triton-h800`
- `content.code`: exact UTF-8 contents of the reviewed single Python file
- `content.compileAndRunOptions`: `{}`
- Authentication: existing web `sessionToken`
- Captcha source: `/Users/sakimi/Desktop/xpuoj-turnstile-pool`, action
  `custom_test`; it is accessed only by the explicit `submit` subcommand

Local review, with no network or token-pool access:

```bash
python3 experiments/2026-09-12/public24_triton_custom.py review PROBE.py
```

Read-only platform preflight:

```bash
python3 experiments/2026-09-12/public24_triton_custom.py preflight
```

After review of the exact source hash, create one custom test:

```bash
python3 experiments/2026-09-12/public24_triton_custom.py submit PROBE.py \
  --expect-sha256 REVIEWED_SHA256
```

Read a known result without printing arbitrary judge output:

```bash
python3 experiments/2026-09-12/public24_triton_custom.py status CUSTOM_TEST_ID
```

The source must use a stable `P1MD ` prefix for its own diagnostic lines.  The
status command returns only those lines from `userError`/`userOutput`, plus
compile and case status.  The historical public-24 recipe reliably preserves
function-local Python `print` output when the generated-workload script ends in
an intentional Python exception.  A diagnostic source should therefore:

1. define its own `@triton.jit` kernels and a `run_kernel` entry;
2. construct only self-generated CUDA tensors with allowed `torch` operations;
3. benchmark a callable with `triton.testing.do_bench`;
4. print bounded `P1MD ...` timing/resource lines from inside a function; and
5. intentionally raise `RuntimeError`, or call the undefined
   `p1md_force_user_error()` marker at module scope, after printing so the text
   is retained in `progress.cases[].result.userError`.  The historical public-24
   recipe used the undefined-name form, so `RuntimeError` is not mandatory.

Keep total compilation and execution below the historical 15-second per-case
limit.  Do not print generated tensor contents or inspect hidden workload data.
The live public problem metadata checked on 2026-09-12 does not expose this
limit; 15 seconds is the conservative 2026-09-04 observed GeneratedWorkload
budget and included Triton compilation.

The first payload should import only `torch`, `triton`,
`triton.language as tl`, and the already exercised
`triton.tools.tensor_descriptor.TensorDescriptor`.  Direct `torch.cuda.*`,
dunder attributes such as `triton.__version__`, and `torch.tensor` are known
sandbox failures.  `numpy`, `json`, CPU tensor generation, and broad host-side
Torch helpers have no successful public-24 custom-test evidence.  The client
rejects archived validator/TorchProxy failures before any network or token-pool
access and reports APIs that merely lack public-24 evidence as non-blocking
warnings for the reviewer.

On 2026-09-12 the current frontend bundles still referenced both
`customTest/createCustomTest` and `contest/play/createCustomTest`; the public
route in this file has not disappeared.  This was a read-only frontend check,
not a create request.
