#!/usr/bin/env python3
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from xpuoj_web import submit  # noqa: E402


def main():
    source = Path(sys.argv[1])
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        sid = submit(source.read_text())
    print(f"SID = {sid}")


if __name__ == "__main__":
    main()
