#!/usr/bin/env python
"""
Agentic-V2XShield Dataset Audit Script

Purpose:
    Scan VeReMi-NextGen zip/json files, extract metadata, inspect file structures,
    summarize ground-truth labels, estimate attack distribution, and create reports.

Usage:
    python dataset_audit.py --data-dir "D:\other\Agentic-V2XShield\Dataset" --out-dir "outputs_dataset_audit"

Outputs:
    outputs_dataset_audit/
        dataset_inventory.csv
        zip_file_inventory.csv
        groundtruth_summary.csv
        groundtruth_attack_distribution.csv
        inferred_dataset_report.md
        sample_records/
"""

import argparse
import csv
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def safe_read_json(path: Path) -> Any:
    """Read JSON safely."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_dict(d: Dict[str, Any], prefix: str = "", max_depth: int = 3) -> Dict[str, Any]:
    """Flatten nested dictionaries for compact inspection."""
    out = {}
    if max_depth <= 0:
        out[prefix.rstrip(".")] = str(type(d).__name__)
        return out

    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key + ".", max_depth=max_depth - 1))
        elif isinstance(v, list):
            out[key] = f"list(len={len(v)})"
            if v and isinstance(v[0], dict):
                out.update(flatten_dict(v[0], key + "[0].", max_depth=max_depth - 1))
        else:
            out[key] = v
    return out


def infer_asset_info(filename: str) -> Dict[str, str]:
    """
    Infer scenario, density, split, and attack type from VeReMi-NextGen asset names.
    Example:
        InTAS_urban_2_randomPositionOffset.zip
        InTAS_urban_2_Train_groundTruth.json
    """
    stem = Path(filename).stem
    info = {
        "filename": filename,
        "scenario": "",
        "density": "",
        "asset_kind": "",
        "attack_type": "",
        "split": "",
    }

    m = re.match(r"InTAS_(highway|urban)_(\d+)(?:_(.*))?$", stem)
    if not m:
        info["asset_kind"] = "unknown"
        return info

    info["scenario"] = m.group(1)
    info["density"] = m.group(2)
    suffix = m.group(3) or ""

    if suffix == "":
        info["asset_kind"] = "base_dataset"
    elif "groundTruth" in suffix or "ground_truth" in suffix:
        info["asset_kind"] = "ground_truth"
        if "Train" in suffix:
            info["split"] = "train"
        elif "Validation" in suffix:
            info["split"] = "validation"
        elif "Test" in suffix:
            info["split"] = "test"
        else:
            info["split"] = "unknown"
    else:
        info["asset_kind"] = "attack_specific_dataset"
        info["attack_type"] = suffix

    return info


def list_zip_members(zip_path: Path, max_sample: int = 10) -> Dict[str, Any]:
    """Inspect a zip without extracting it."""
    result = {
        "zip_name": zip_path.name,
        "num_members": 0,
        "total_uncompressed_mb": 0.0,
        "extensions": {},
        "sample_members": [],
        "has_json": False,
        "has_csv": False,
        "has_xml": False,
        "has_txt": False,
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = zf.infolist()
            result["num_members"] = len(infos)
            result["total_uncompressed_mb"] = round(sum(i.file_size for i in infos) / (1024 * 1024), 3)
            exts = Counter()
            samples = []
            for i in infos:
                if i.is_dir():
                    continue
                ext = Path(i.filename).suffix.lower() or "[no_ext]"
                exts[ext] += 1
                if len(samples) < max_sample:
                    samples.append(i.filename)
            result["extensions"] = dict(exts)
            result["sample_members"] = samples
            result["has_json"] = ".json" in exts
            result["has_csv"] = ".csv" in exts
            result["has_xml"] = ".xml" in exts
            result["has_txt"] = ".txt" in exts
    except zipfile.BadZipFile:
        result["error"] = "BadZipFile"
    except Exception as e:
        result["error"] = str(e)

    return result


def summarize_groundtruth_json(path: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Summarize ground truth JSON.

    This function is intentionally flexible because VeReMi-NextGen JSON structures
    may contain lists, dictionaries, nested records, or split-specific formats.
    """
    summary = {
        "file": path.name,
        "status": "ok",
        "top_level_type": "",
        "num_top_level_items": 0,
        "candidate_record_count": 0,
        "detected_key_columns": "",
        "possible_label_keys": "",
        "possible_attack_keys": "",
    }

    try:
        data = safe_read_json(path)
    except Exception as e:
        summary["status"] = f"read_error: {e}"
        return summary, pd.DataFrame()

    summary["top_level_type"] = type(data).__name__

    records = []

    if isinstance(data, list):
        summary["num_top_level_items"] = len(data)
        for x in data:
            if isinstance(x, dict):
                records.append(x)

    elif isinstance(data, dict):
        summary["num_top_level_items"] = len(data)

        # Case 1: dict of records
        dict_values = list(data.values())
        if dict_values and all(isinstance(v, dict) for v in dict_values[: min(20, len(dict_values))]):
            for k, v in data.items():
                rec = dict(v)
                rec["_record_id"] = k
                records.append(rec)

        # Case 2: dict containing lists of records
        for k, v in data.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        rec = dict(item)
                        rec["_parent_key"] = k
                        records.append(rec)

        # Fallback: use one flattened top-level row
        if not records:
            records = [flatten_dict(data)]

    else:
        summary["status"] = "unsupported_json_structure"

    summary["candidate_record_count"] = len(records)

    if not records:
        return summary, pd.DataFrame()

    flat_records = []
    for r in records[:200000]:  # safety cap
        if isinstance(r, dict):
            flat_records.append(flatten_dict(r, max_depth=4))

    df = pd.DataFrame(flat_records)

    cols = list(df.columns)
    summary["detected_key_columns"] = ", ".join(cols[:40])

    label_keys = [c for c in cols if re.search(r"label|class|malicious|attack|attacker|benign|type", c, re.I)]
    attack_keys = [c for c in cols if re.search(r"attack|type|class|attacker", c, re.I)]

    summary["possible_label_keys"] = ", ".join(label_keys[:20])
    summary["possible_attack_keys"] = ", ".join(attack_keys[:20])

    return summary, df


def value_distribution(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Create compact distributions for likely label/attack columns."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    candidate_cols = [
        c for c in df.columns
        if re.search(r"label|class|malicious|attack|attacker|benign|type", c, re.I)
    ]

    for col in candidate_cols[:25]:
        vc = df[col].astype(str).value_counts(dropna=False).head(50)
        for value, count in vc.items():
            rows.append({
                "source_file": source_file,
                "column": col,
                "value": value,
                "count": int(count),
                "ratio": float(count / max(len(df), 1)),
            })

    return pd.DataFrame(rows)


def write_markdown_report(
    out_path: Path,
    inventory_df: pd.DataFrame,
    zip_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    dist_df: pd.DataFrame,
) -> None:
    """Write human-readable report."""
    lines = []
    lines.append("# VeReMi-NextGen Dataset Audit Report\n")
    lines.append("## 1. Dataset Inventory\n")
    lines.append(f"- Total files found: **{len(inventory_df)}**")
    if not inventory_df.empty:
        lines.append(f"- ZIP files: **{int((inventory_df['extension'] == '.zip').sum())}**")
        lines.append(f"- JSON files: **{int((inventory_df['extension'] == '.json').sum())}**")
        lines.append("")

        scenario_counts = inventory_df["scenario"].replace("", "unknown").value_counts().to_dict()
        kind_counts = inventory_df["asset_kind"].replace("", "unknown").value_counts().to_dict()
        lines.append("### Scenario Counts\n")
        for k, v in scenario_counts.items():
            lines.append(f"- {k}: {v}")
        lines.append("\n### Asset Kinds\n")
        for k, v in kind_counts.items():
            lines.append(f"- {k}: {v}")

    lines.append("\n## 2. ZIP Structure Summary\n")
    if not zip_df.empty:
        show_cols = ["zip_name", "num_members", "total_uncompressed_mb", "extensions", "sample_members"]
        lines.append(zip_df[show_cols].to_markdown(index=False))
    else:
        lines.append("No ZIP files found.")

    lines.append("\n## 3. Ground-Truth Summary\n")
    if not gt_df.empty:
        show_cols = [
            "file", "top_level_type", "num_top_level_items", "candidate_record_count",
            "possible_label_keys", "possible_attack_keys"
        ]
        lines.append(gt_df[show_cols].to_markdown(index=False))
    else:
        lines.append("No ground-truth JSON summary available.")

    lines.append("\n## 4. Candidate Label / Attack Distributions\n")
    if not dist_df.empty:
        lines.append(dist_df.head(120).to_markdown(index=False))
    else:
        lines.append("No candidate label/attack distributions detected.")

    lines.append("\n## 5. Recommended Next Step\n")
    lines.append(
        "Use the detected ground-truth columns to build one unified table with these minimum fields: "
        "`timestamp`, `sender_id`, `pos_x`, `pos_y`, `speed`, `acceleration`, `heading`, "
        "`attack_type`, and `label`. After this table is created, train ML baselines and then add the "
        "agentic edge-cloud decision layer."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to folder containing VeReMi-NextGen zip/json files.")
    parser.add_argument("--out-dir", default="outputs_dataset_audit", help="Output folder.")
    parser.add_argument("--extract-samples", action="store_true", help="Extract small sample files from zips if supported.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    sample_dir = out_dir / "sample_records"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)

    all_files = [p for p in data_dir.iterdir() if p.is_file()]
    if not all_files:
        print(f"[ERROR] No files found in: {data_dir}")
        sys.exit(1)

    inventory_rows = []
    zip_rows = []
    gt_rows = []
    dist_frames = []

    print(f"[INFO] Scanning: {data_dir}")
    print(f"[INFO] Files found: {len(all_files)}")

    for p in sorted(all_files):
        info = infer_asset_info(p.name)
        row = {
            **info,
            "path": str(p),
            "extension": p.suffix.lower(),
            "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
        }
        inventory_rows.append(row)

        if p.suffix.lower() == ".zip":
            print(f"[ZIP] {p.name}")
            zip_info = list_zip_members(p)
            zip_rows.append(zip_info)

        elif p.suffix.lower() == ".json":
            print(f"[JSON] {p.name}")
            summary, df = summarize_groundtruth_json(p)
            gt_rows.append(summary)

            if not df.empty:
                # Save compact sample for manual inspection
                sample_file = sample_dir / f"{p.stem}_sample.csv"
                df.head(200).to_csv(sample_file, index=False)

                dist = value_distribution(df, p.name)
                if not dist.empty:
                    dist_frames.append(dist)

    inventory_df = pd.DataFrame(inventory_rows)
    zip_df = pd.DataFrame(zip_rows)
    gt_df = pd.DataFrame(gt_rows)
    dist_df = pd.concat(dist_frames, ignore_index=True) if dist_frames else pd.DataFrame()

    inventory_df.to_csv(out_dir / "dataset_inventory.csv", index=False)
    zip_df.to_csv(out_dir / "zip_file_inventory.csv", index=False)
    gt_df.to_csv(out_dir / "groundtruth_summary.csv", index=False)
    if not dist_df.empty:
        dist_df.to_csv(out_dir / "groundtruth_attack_distribution.csv", index=False)
    else:
        pd.DataFrame(columns=["source_file", "column", "value", "count", "ratio"]).to_csv(
            out_dir / "groundtruth_attack_distribution.csv", index=False
        )

    write_markdown_report(
        out_dir / "inferred_dataset_report.md",
        inventory_df,
        zip_df,
        gt_df,
        dist_df,
    )

    print("\n[DONE] Dataset audit complete.")
    print(f"[OUT] {out_dir.resolve()}")
    print("\nCreated:")
    print(" - dataset_inventory.csv")
    print(" - zip_file_inventory.csv")
    print(" - groundtruth_summary.csv")
    print(" - groundtruth_attack_distribution.csv")
    print(" - inferred_dataset_report.md")
    print(" - sample_records/*.csv")


if __name__ == "__main__":
    main()
