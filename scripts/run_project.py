"""Jalankan source notebook yang diekspor, dengan cwd dan seed startup yang tetap."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    if sys.version_info < (3,12):
        raise SystemExit("Gunakan Python 3.12 atau lebih baru.")
    env = dict(os.environ, PYTHONHASHSEED="42", MPLBACKEND="Agg")
    return subprocess.run([sys.executable, str(ROOT / "src/uts_ml.py")], cwd=ROOT, env=env).returncode

if __name__ == "__main__":
    raise SystemExit(main())
