import csv
import json
from pathlib import Path

from scripts.validate_repository import _check_json, validate


def _write_fixture(root: Path) -> None:
    values = {
        "data/clusters/cluster_to_label.json": {"0": "general"},
        "data/clusters/product_info.json": {"item": {"title": "Example"}},
        "data/clusters/user_cluster_map.json": {"user": 0},
        "data/processed/user_embeddings.json": {"user": [0.1, 0.2]},
    }
    for relative_path, value in values.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    preference_path = root / "data/processed/rlhf_preference_dataset.jsonl"
    preference_path.write_text('{"chosen": "A", "rejected": "B"}\n', encoding="utf-8")

    mapping_path = root / "data/processed/asin_mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["asin", "index"])
        writer.writerow(["B000000001", 0])


def test_validate_accepts_complete_fixture(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    results = validate(tmp_path)

    assert len(results) == 6
    assert all(result.ok for result in results)


def test_validate_reports_missing_artifacts(tmp_path: Path) -> None:
    results = validate(tmp_path)

    assert len(results) == 6
    assert all(not result.ok for result in results)
    assert all(result.detail == "file is missing" for result in results)


def test_json_check_reports_invalid_content(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not-json}", encoding="utf-8")

    result = _check_json(path)

    assert not result.ok
    assert "Expecting property name" in result.detail
