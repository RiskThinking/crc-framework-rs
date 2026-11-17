"""Copy SDK-owned data contracts into the native consumer's resource tree."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = {
    "schemas/dataset-2.0.schema.json": "dataset/dataset-2.0.schema.json",
    "fixtures/dataset-2.0.json": "dataset/fixtures.json",
    "taxonomy/taxonomy-1.0.schema.json": "taxonomy/taxonomy-1.0.schema.json",
    "taxonomy/fixtures.json": "taxonomy/fixtures.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True, help="SDK repository root")
    parser.add_argument("--check", action="store_true", help="check without writing")
    args = parser.parse_args()
    source_root = args.sdk / "crc_sdk/contracts"
    destination_root = ROOT / "python/crc_framework/contracts"
    for source, destination in RESOURCES.items():
        data = (source_root / source).read_bytes()
        target = destination_root / destination
        if args.check:
            if not target.exists() or target.read_bytes() != data:
                raise SystemExit(f"{target} differs from {source_root / source}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


if __name__ == "__main__":
    main()
