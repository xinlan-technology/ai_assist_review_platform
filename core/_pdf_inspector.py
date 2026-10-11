"""Standalone PDF metadata worker with no app, storage, or network imports.

The caller enforces the deadline; CPU and memory limits apply where supported.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import BytesIO
import json
import sys


def _limits() -> None:
    if sys.platform == "win32":
        return
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)


def _inspect(data: bytes) -> dict:
    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except MemoryError:
                raise
            except Exception:
                unlocked = 0
            if not unlocked:
                return {"error": "password"}
        return {"pages": len(reader.pages)}
    except MemoryError:
        return {"error": "resource"}
    except Exception:
        # A parser failure may leave the page count unknown.
        return {"pages": None}


def main() -> None:
    _limits()
    data = sys.stdin.buffer.read(50 * 1024 * 1024 + 1)
    if len(data) > 50 * 1024 * 1024:
        report = {"error": "size"}
    else:
        # Only our tiny protocol response reaches the parent's stdout pipe.
        with redirect_stdout(sys.stderr):
            report = _inspect(data)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
