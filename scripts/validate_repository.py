"""Run fast, dependency-free checks against the checked-in GenRec artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _name(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _check_json(path: Path) -> CheckResult:
    try:
        value = _load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return CheckResult(_name(path), False, str(exc))
    size = len(value) if hasattr(value, "__len__") else 1
    return CheckResult(_name(path), True, f"valid JSON ({size} records)")


def _check_jsonl(path: Path) -> CheckResult:
    count = 0
    line_number = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                json.loads(line)
                count += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return CheckResult(
            _name(path), False, f"line {line_number}: {exc}"
        )
    return CheckResult(_name(path), True, f"valid JSONL ({count} records)")


def _check_csv(path: Path) -> CheckResult:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
    except (OSError, UnicodeError, StopIteration, csv.Error) as exc:
        return CheckResult(_name(path), False, str(exc))
    if not header:
        return CheckResult(_name(path), False, "missing header")
    return CheckResult(
        _name(path), True, f"valid CSV header ({len(header)} columns)"
    )


def _check_nonempty(path: Path) -> CheckResult:
    size = path.stat().st_size
    return CheckResult(_name(path), size > 0, f"{size:,} bytes")


CHECKS: tuple[tuple[str, Callable[[Path], CheckResult]], ...] = (
    ("data/clusters/cluster_to_label.json", _check_json),
    ("data/clusters/product_info.json", _check_json),
    ("data/clusters/user_cluster_map.json", _check_nonempty),
    ("data/processed/user_embeddings.json", _check_nonempty),
    ("data/processed/rlhf_preference_dataset.jsonl", _check_jsonl),
    ("data/processed/asin_mapping.csv", _check_csv),
)


def validate(root: Path = ROOT) -> list[CheckResult]:
    results: list[CheckResult] = []
    for relative_path, checker in CHECKS:
        path = root / relative_path
        if not path.is_file():
            results.append(CheckResult(relative_path, False, "file is missing"))
            continue
        results.append(checker(path))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository root to validate"
    )
    args = parser.parse_args()

    results = validate(args.root.resolve())
    for result in results:
        marker = "PASS" if result.ok else "FAIL"
        print(f"[{marker}] {result.name}: {result.detail}")

    failures = sum(not result.ok for result in results)
    print(f"\n{len(results) - failures}/{len(results)} checks passed")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
