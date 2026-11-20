"""Run the published Python examples against the installed framework."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDES = (
    "README.md",
    "docs/python.md",
    "docs/measurements.md",
    "docs/impact-contracts.md",
    "docs/javascript.md",
    "docs/migration-v1.md",
)


def main() -> None:
    for guide in GUIDES:
        examples = re.findall(
            r"^\x60\x60\x60python\n(.*?)^\x60\x60\x60$",
            (ROOT / guide).read_text(encoding="utf-8"),
            flags=re.MULTILINE | re.DOTALL,
        )
        if not examples:
            raise SystemExit(f"{guide}: expected Python examples")
        print(f"Running {len(examples)} examples from {guide}", flush=True)
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "example.py"
            script.write_text("\n\n".join(examples), encoding="utf-8")
            subprocess.run(
                [sys.executable, "-I", str(script)], cwd=directory, check=True
            )


if __name__ == "__main__":
    main()
