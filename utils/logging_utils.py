from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict


def append_metrics_row(csv_path: str | Path, row: Dict[str, Any]) -> None:
    """Append a metrics row to a CSV file, creating headers on first write."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(json_path: str | Path, payload: Dict[str, Any]) -> None:
    """Write a JSON file with indentation."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_jsonl(jsonl_path: str | Path, payload: Dict[str, Any]) -> None:
    """Append one JSON object per line to a JSONL file."""
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
