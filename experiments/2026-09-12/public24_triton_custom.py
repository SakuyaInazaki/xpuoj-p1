#!/usr/bin/env python3
"""Review and submit one public-problem-24 Triton H800 custom test.

The default workflow is local-only::

    python3 public24_triton_custom.py review probe.py

The only write operation is the explicit ``submit`` subcommand.  It requires
the caller to repeat the reviewed source SHA-256 before this program contacts
the token pool or creates a custom test::

    python3 public24_triton_custom.py submit probe.py \
      --expect-sha256 <reviewed-sha256>

This tool never prints credentials, PoW material, captcha tokens, source code,
or arbitrary judge output.  ``status`` emits compile state and only diagnostic
lines beginning with ``P1MD `` from the submitter's own stdout/stderr.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import uuid

import requests


API = "https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com/api/"
SECRET = Path("/Users/sakimi/Desktop/xpuoj-p1/.secrets/xpuoj.json")
POOL_DIR = Path("/Users/sakimi/Desktop/xpuoj-turnstile-pool")

PROBLEM_ID = 24
LANGUAGE = "triton-h800"
MODE = "GeneratedWorkload"
ACTION = "custom_test"
CREATE_ENDPOINT = "customTest/createCustomTest"
DETAIL_ENDPOINT = "customTest/getCustomTestDetail"
MAX_SOURCE_BYTES = 1_048_576
DIAGNOSTIC_PREFIX = "P1MD "

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)


def source_record(path_text: str) -> tuple[Path, str, bytes, str, list[str]]:
    path = Path(path_text).expanduser().resolve()
    raw = path.read_bytes()
    if len(raw) > MAX_SOURCE_BYTES:
        raise RuntimeError(f"source exceeds {MAX_SOURCE_BYTES} bytes")
    code = raw.decode("utf-8")
    required = {
        "import triton": "Triton import",
        "@triton.jit": "a Triton JIT kernel",
        "def run_kernel": "run_kernel entry",
        "triton.testing.do_bench": "the proven benchmark helper",
    }
    missing = [label for marker, label in required.items() if marker not in code]
    if missing:
        raise RuntimeError("source is missing: " + ", ".join(missing))
    if "raise RuntimeError" not in code and "p1md_force_user_error()" not in code:
        raise RuntimeError(
            "source is missing an intentional RuntimeError or the reviewed "
            "p1md_force_user_error() NameError marker"
        )
    # Reproduce the rules backed by archived validator/TorchProxy failures.
    # Calls that merely lack public-24 evidence are warnings, not blockers.
    tree = ast.parse(code, filename=str(path))

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = dotted(node.value)
            return (parent + "." if parent else "") + node.attr
        return ""

    names = {dotted(node) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names.update(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    found = []
    warnings = []
    if "json" in imports or "json" in names:
        found.append("json is outside the proven payload imports")
    if any(name == "numpy" or name.startswith("numpy.") for name in imports | names):
        found.append("numpy is outside the proven payload imports")
    blocked_names = {
        "torch.tensor": "torch.tensor is blocked",
        "torch.full": "torch.full is blocked",
        "torch.full_like": "torch.full_like is blocked",
        "torch.manual_seed": "torch.manual_seed is blocked",
        "torch.matmul": "host torch.matmul is recorded as blocked",
    }
    unverified_names = {
        "torch.Generator": "torch.Generator has no successful public-24 evidence",
        "torch.rand": "torch.rand has no successful public-24 evidence",
        "torch.randn": "torch.randn has no successful public-24 evidence",
        "torch.topk": "host torch.topk is unverified",
        "torch.sort": "host torch.sort is unverified",
        "torch.bincount": "host torch.bincount is unverified",
        "torch.repeat_interleave": "host repeat_interleave is unverified",
        "torch.cumsum": "host torch.cumsum is unverified",
        "torch.sum": "host torch.sum has no successful public-24 evidence",
        "torch.mean": "host torch.mean has no successful public-24 evidence",
        "torch.log10": "host torch.log10 has no successful public-24 evidence",
        "torch.linalg.vector_norm": (
            "torch.linalg.vector_norm has no successful public-24 evidence"
        ),
    }
    if any(name == "torch.cuda" or name.startswith("torch.cuda.") for name in names):
        found.append("direct torch.cuda access is blocked")
    found.extend(reason for name, reason in blocked_names.items() if name in names)
    warnings.extend(
        reason for name, reason in unverified_names.items() if name in names
    )

    def literal(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return all(literal(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return all(literal(key) for key in node.keys if key is not None) and all(
                literal(value) for value in node.values
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return literal(node.operand)
        return False

    jit_nodes: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_jit = any(dotted(dec.func if isinstance(dec, ast.Call) else dec).endswith(".jit")
                         for dec in node.decorator_list)
            if is_jit:
                jit_nodes.update(id(child) for child in ast.walk(node))
        if isinstance(node, ast.Assign) and not literal(node.value):
            found.append(f"line {node.lineno}: top-level assignment RHS is not literal")
        if isinstance(node, ast.AnnAssign):
            found.append(f"line {node.lineno}: top-level annotated assignment is blocked")

    blocked_builtins = {
        "dir", "getattr", "hasattr", "min", "max", "setattr", "delattr",
        "eval", "exec", "compile", "globals", "locals", "vars", "open",
        "__import__",
    }
    blocked_tensor_methods = {
        "max", "min", "all", "any", "sum", "amax", "amin", "argmax",
        "argmin", "norm", "mean", "std", "abs",
    }
    safe_receivers = {"tl", "triton", "torch", "math"}
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.AnnAssign) and id(node) not in jit_nodes:
            found.append(f"line {line}: annotated assignment outside top-level JIT")
        if isinstance(node, ast.Lambda):
            found.append(f"line {line}: lambda is blocked")
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(
            node.op, ast.MatMult
        ):
            found.append(f"line {line}: '@' matrix operator is blocked")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in blocked_builtins:
                found.append(f"line {line}: builtin/name {node.id} is blocked")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            found.append(f"line {line}: dunder attribute {node.attr} is blocked")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in blocked_tensor_methods:
                receiver = dotted(node.func.value)
                receiver_root = receiver.split(".", 1)[0]
                if receiver_root not in safe_receivers:
                    found.append(
                        f"line {line}: tensor method .{node.func.attr}() is blocked"
                    )
    found = list(dict.fromkeys(found))
    warnings = list(dict.fromkeys(warnings))
    if found:
        raise RuntimeError("source uses risky sandbox APIs: " + "; ".join(found))
    return path, code, raw, hashlib.sha256(raw).hexdigest(), warnings


def payload(code: str) -> dict:
    return {
        "problemId": PROBLEM_ID,
        "mode": MODE,
        "content": {
            "language": LANGUAGE,
            "code": code,
            "compileAndRunOptions": {},
        },
    }


def review(path_text: str) -> str:
    path, _code, raw, digest, warnings = source_record(path_text)
    shape = payload(f"<source sha256={digest} bytes={len(raw)}>")
    print(
        json.dumps(
            {
                "network": False,
                "tokenPoolAccess": False,
                "sourcePath": str(path),
                "sourceBytes": len(raw),
                "sourceSha256": digest,
                "endpoint": CREATE_ENDPOINT,
                "action": ACTION,
                "unverifiedApiWarnings": warnings,
                "payload": shape,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return digest


def session() -> requests.Session:
    token = json.loads(SECRET.read_text(encoding="utf-8"))["sessionToken"]
    s = requests.Session()
    s.headers.update(
        {
            "accept": "application/json, text/plain, */*",
            "accept-language": "zh-CN,zh;q=0.9",
            "authorization": "Bearer " + token,
            "content-type": "application/json",
            "origin": "https://xpuoj.com",
            "referer": "https://xpuoj.com/",
            "user-agent": UA,
        }
    )
    return s


def request_json(s: requests.Session, endpoint: str, body: dict) -> dict:
    response = s.post(API + endpoint, json=body, timeout=60)
    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"{endpoint} returned non-JSON HTTP {response.status_code}") from exc
    if response.status_code >= 400:
        safe = {k: data.get(k) for k in ("error", "message", "code") if k in data}
        raise RuntimeError(f"{endpoint} HTTP {response.status_code}: {safe}")
    return data


def preflight(s: requests.Session) -> dict:
    info_response = s.get(API + "auth/getSessionInfo", timeout=60)
    info_response.raise_for_status()
    info = info_response.json()
    available = info.get("availableLanguages") or {}
    modes = (available.get("customTestModes") or {}).get(LANGUAGE) or []

    problem = request_json(
        s,
        "problem/getProblem",
        {"id": PROBLEM_ID, "localizedContentsOfLocale": "zh_CN"},
    )
    availability = request_json(
        s,
        "judgeClient/checkAvailability",
        {"language": LANGUAGE, "requiredFlags": ["custom-test"]},
    )
    credit_response = request_json(s, "customTest/getCustomTestCreditStatus", {})
    credit = credit_response.get("data") or credit_response
    out = {
        "network": True,
        "writeOperation": False,
        "problem24Readable": "error" not in problem,
        "language": LANGUAGE,
        "advertisedModes": modes,
        "generatedWorkloadAdvertised": MODE in modes,
        "judgeAvailable": availability.get("available"),
        "judgeReason": availability.get("reason"),
        "credit": {
            k: credit.get(k)
            for k in ("state", "available", "capacity", "nextCreditAt", "fullyRefilledAt")
            if k in credit
        },
        "createEndpoint": CREATE_ENDPOINT,
        "createEndpointCheckedByWrite": False,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not out["problem24Readable"]:
        raise RuntimeError("public problem 24 is no longer readable")
    if MODE not in modes:
        raise RuntimeError(f"{MODE} is no longer advertised for {LANGUAGE}")
    if availability.get("available") is not True:
        raise RuntimeError(f"{LANGUAGE} custom-test judge is unavailable")
    return out


def pow_ok(digest: bytes, difficulty: int) -> bool:
    whole_bytes = difficulty // 2
    if any(digest[i] != 0 for i in range(whole_bytes)):
        return False
    return difficulty % 2 == 0 or digest[whole_bytes] >> 4 == 0


def solve_pow(random_data: str, difficulty: int) -> tuple[int, str]:
    prefix = random_data.encode()
    nonce = 0
    while True:
        digest = hashlib.sha256(prefix + str(nonce).encode()).digest()
        if pow_ok(digest, difficulty):
            return nonce, digest.hex()
        nonce += 1


def fresh_pow(s: requests.Session) -> str:
    challenge = request_json(
        s, "proofOfWork/issueChallenge", {"action": ACTION}
    )
    nonce, response = solve_pow(challenge["randomData"], challenge["difficulty"])
    return json.dumps(
        {"id": challenge["id"], "nonce": nonce, "response": response},
        separators=(",", ":"),
    )


def take_custom_token() -> str:
    sys.path.insert(0, str(POOL_DIR))
    from pool_client import take  # type: ignore

    # The pool client logs token fragments on stderr.  Keep them out of logs.
    with contextlib.redirect_stderr(io.StringIO()):
        return take(ACTION, timeout=180)


def submit(path_text: str, expected_sha256: str) -> None:
    path, code, raw, digest, warnings = source_record(path_text)
    if digest != expected_sha256.lower():
        raise RuntimeError(
            f"reviewed hash mismatch: expected {expected_sha256.lower()}, actual {digest}"
        )

    # Complete every reversible/read-only check before consuming a token.
    s = session()
    preflight(s)
    print(
        json.dumps(
            {
                "sourcePath": str(path),
                "sourceBytes": len(raw),
                "sourceSha256": digest,
                "endpoint": CREATE_ENDPOINT,
                "problemId": PROBLEM_ID,
                "language": LANGUAGE,
                "mode": MODE,
                "status": "preflight-passed; acquiring one custom_test token",
                "unverifiedApiWarnings": warnings,
            },
            ensure_ascii=False,
        )
    )

    captcha = take_custom_token()
    headers = {
        "x-captcha-result": json.dumps(
            {"turnstile": {"token": captcha}}, separators=(",", ":")
        ),
        "x-proof-of-work": fresh_pow(s),
        "x-request-id": str(uuid.uuid4()),
    }
    response = s.post(
        API + CREATE_ENDPOINT,
        json=payload(code),
        headers=headers,
        timeout=180,
    )
    try:
        result = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"create returned non-JSON HTTP {response.status_code}; token consumed"
        ) from exc
    if response.status_code not in (200, 201) or result.get("error"):
        safe = {k: result.get(k) for k in ("error", "message", "code") if k in result}
        raise RuntimeError(
            f"create failed HTTP {response.status_code}: {safe}; token may be consumed"
        )
    data = result.get("data") if isinstance(result.get("data"), dict) else result
    custom_test_id = data.get("id") or data.get("customTestId")
    if not custom_test_id:
        raise RuntimeError(
            f"create returned no id; response keys={sorted(result)}; token consumed"
        )
    print(
        json.dumps(
            {
                "customTestId": str(custom_test_id),
                "sourceSha256": digest,
                "initialState": data.get("state"),
            },
            ensure_ascii=False,
        )
    )


def own_diagnostic_lines(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [line for line in value.splitlines() if line.startswith(DIAGNOSTIC_PREFIX)]


def status(custom_test_id: str) -> None:
    s = session()
    response = request_json(s, DETAIL_ENDPOINT, {"id": str(custom_test_id)})
    data = response.get("data") if isinstance(response.get("data"), dict) else response
    progress = data.get("progress") if isinstance(data.get("progress"), dict) else {}
    compile_result = (
        progress.get("compile") if isinstance(progress.get("compile"), dict) else {}
    )
    cases = []
    for index, case in enumerate(progress.get("cases") or []):
        result = case.get("result") if isinstance(case, dict) else {}
        if not isinstance(result, dict):
            result = {}
        cases.append(
            {
                "index": index,
                "status": result.get("status") or case.get("status"),
                "time": result.get("time"),
                "memory": result.get("memory"),
                "diagnostics": own_diagnostic_lines(result.get("userError"))
                + own_diagnostic_lines(result.get("userOutput")),
            }
        )
    out = {
        "customTestId": str(custom_test_id),
        "error": response.get("error"),
        "state": data.get("state"),
        "status": progress.get("status") or data.get("status"),
        "compileSuccess": compile_result.get("success"),
        "compileMessage": str(compile_result.get("message") or "")[:2000],
        "totalOccupiedTime": progress.get("totalOccupiedTime"),
        "cases": cases,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    review_parser = sub.add_parser("review", help="local-only payload review")
    review_parser.add_argument("source")

    sub.add_parser("preflight", help="read-only platform preflight")

    submit_parser = sub.add_parser("submit", help="create exactly one custom test")
    submit_parser.add_argument("source")
    submit_parser.add_argument("--expect-sha256", required=True)

    status_parser = sub.add_parser("status", help="read one existing custom-test result")
    status_parser.add_argument("custom_test_id")

    args = parser.parse_args()
    if args.command == "review":
        review(args.source)
    elif args.command == "preflight":
        preflight(session())
    elif args.command == "submit":
        submit(args.source, args.expect_sha256)
    elif args.command == "status":
        status(args.custom_test_id)


if __name__ == "__main__":
    main()
