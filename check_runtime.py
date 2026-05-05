#!/usr/bin/env python3
"""Runtime pre-flight checks for ChatTrader.KPai.

Checks:
1) Python version is 3.12.x
2) CUDA is available in PyTorch
3) Parquet engine compatibility (pandas + pyarrow roundtrip)
"""

from __future__ import annotations

import platform
import sys
import tempfile
from pathlib import Path


def _check_python_version() -> tuple[bool, str]:
    major, minor = sys.version_info[:2]
    ok = (major, minor) == (3, 12)
    msg = f"python_version={platform.python_version()} expected=3.12.x"
    return ok, msg


def _check_cuda() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure path
        return False, f"torch_import_error={exc}"

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        return True, f"cuda_available=True device={device_name}"
    return False, "cuda_available=False"


def _check_parquet() -> tuple[bool, str]:
    try:
        import pandas as pd
        import pyarrow  # noqa: F401 - validates importability
    except Exception as exc:  # pragma: no cover - import failure path
        return False, f"parquet_dependency_error={exc}"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "runtime_check.parquet"
            df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30]})
            df.to_parquet(path, engine="pyarrow", index=False)
            loaded = pd.read_parquet(path, engine="pyarrow")
            ok = bool(df.equals(loaded))
            return ok, f"parquet_roundtrip_ok={ok} rows={len(loaded)}"
    except Exception as exc:  # pragma: no cover - parquet runtime path
        return False, f"parquet_roundtrip_error={exc}"


def main() -> int:
    checks = [
        ("PYTHON_3_12", _check_python_version),
        ("CUDA_AVAILABLE", _check_cuda),
        ("PARQUET_ENGINE", _check_parquet),
    ]

    failures = []
    print("=== ChatTrader Runtime Validation ===")
    for label, fn in checks:
        ok, detail = fn()
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: {detail}")
        if not ok:
            failures.append(label)

    if failures:
        print(f"RUNTIME_STATUS=FAIL failed_checks={','.join(failures)}")
        return 1

    print("RUNTIME_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
