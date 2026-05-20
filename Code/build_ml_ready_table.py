#!/usr/bin/env python
"""
Build ML-ready VeReMi-NextGen table for Agentic-V2XShield.

This script:
1. Recursively extracts nested ZIP files.
2. Reads JSON / JSONL / CSV records.
3. Infers scenario, density, split, and attack type from file/folder names.
4. Normalizes sender/receiver fields.
5. Creates one ML-ready CSV table for baseline IDS and agentic edge-cloud experiments.

Usage:
    python build_ml_ready_table.py ^
      --data-dir "D:\other\Agentic-V2XShield\Dataset" ^
      --out-dir "outputs_ml_ready" ^
      --scenario urban ^
      --density 2

Main output:
    outputs_ml_ready/veremi_ml_ready_all.csv

Optional smaller output:
    outputs_ml_ready/veremi_ml_ready_sample_100k.csv
"""

import argparse
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


ATTACK_NAMES = [
    "accelerationMultiplication",
    "constantPositionOffset",
    "constantSpeedOffset",
    "dataReplay",
    "dosAttack",
    "feignedBraking",
    "positionMirroring",
    "randomPositionOffset",
    "randomSpeedOffset",
    "reversedHeading",
    "suddenConstantSpeed",
    "suddenStop",
    "timeDelayAttack",
    "trafficCongestionSybil",
    "zeroSpeedReport",
]


def infer_context(path: Path) -> Dict[str, str]:
    """Infer scenario, density, split, and attack_type from path text."""
    s = str(path).replace("\\", "/")

    scenario = ""
    density = ""
    split = ""
    attack_type = "normal"

    m = re.search(r"InTAS_(urban|highway)_(\d+)", s)
    if m:
        scenario = m.group(1)
        density = m.group(2)

    for sp in ["Train", "Validation", "Test", "train", "validation", "test"]:
        if f"/{sp}/" in s or f"_{sp}_" in s:
            split = sp.lower()
            break

    for attack in ATTACK_NAMES:
        if attack in s:
            attack_type = attack
            break

    if "groundTruth" in s or "ground_truth" in s:
        if "Train" in s:
            split = "train"
        elif "Validation" in s:
            split = "validation"
        elif "Test" in s:
            split = "test"

    return {
        "scenario": scenario,
        "density": density,
        "split": split,
        "attack_type": attack_type,
        "label": 0 if attack_type == "normal" else 1,
    }


def recursive_extract_zip(zip_path: Path, extract_root: Path, max_rounds: int = 8) -> None:
    """
    Extract a zip and any nested zips found inside it.
    Uses marker files to avoid repeated extraction.
    """
    extract_root.mkdir(parents=True, exist_ok=True)

    first_target = extract_root / zip_path.stem
    marker = first_target / ".extracted_ok"

    if not marker.exists():
        first_target.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(first_target)
            marker.write_text("ok", encoding="utf-8")
        except zipfile.BadZipFile:
            print(f"[WARN] Bad zip skipped: {zip_path}")
            return

    for _ in range(max_rounds):
        nested = list(extract_root.rglob("*.zip"))
        new_extracted = 0

        for nz in nested:
            target = nz.parent / nz.stem
            marker = target / ".extracted_ok"
            if marker.exists():
                continue

            target.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(nz, "r") as zf:
                    zf.extractall(target)
                marker.write_text("ok", encoding="utf-8")
                new_extracted += 1
            except zipfile.BadZipFile:
                print(f"[WARN] Bad nested zip skipped: {nz}")

        if new_extracted == 0:
            break


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    """Load JSON list/dict records robustly."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="latin-1") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] JSON read failed: {path} | {e}")
        return []

    records = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                records.append(item)

    elif isinstance(data, dict):
        # dict of records
        if data and all(isinstance(v, dict) for v in list(data.values())[:20]):
            for k, v in data.items():
                rec = dict(v)
                rec["_record_id"] = k
                records.append(rec)
        else:
            # dict containing list of records
            for k, v in data.items():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            rec = dict(item)
                            rec["_parent_key"] = k
                            records.append(rec)

    return records


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append(item)
                except Exception:
                    pass
    except Exception as e:
        print(f"[WARN] JSONL read failed: {path} | {e}")
    return records


def parse_pos(value: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse 'x,y,z' position strings or list-like positions."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None, None

    if isinstance(value, str):
        parts = value.split(",")
        try:
            vals = [float(p.strip()) for p in parts[:3]]
            while len(vals) < 3:
                vals.append(None)
            return vals[0], vals[1], vals[2]
        except Exception:
            return None, None, None

    if isinstance(value, (list, tuple)):
        try:
            vals = [float(x) for x in value[:3]]
            while len(vals) < 3:
                vals.append(None)
            return vals[0], vals[1], vals[2]
        except Exception:
            return None, None, None

    return None, None, None


def get_first(row: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in row:
            return row[k]
    return None


def normalize_record(row: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    """Convert different record layouts into one flat schema."""
    ctx = infer_context(source_path)

    sender_pos = get_first(row, ["sender.pos", "pos", "sender_position", "position"])
    receiver_pos = get_first(row, ["receiver.pos", "receiver_position"])

    sx, sy, sz = parse_pos(sender_pos)
    rx, ry, rz = parse_pos(receiver_pos)

    out = {
        "scenario": ctx["scenario"],
        "density": ctx["density"],
        "split": ctx["split"],
        "attack_type": ctx["attack_type"],
        "label": ctx["label"],
        "source_file": str(source_path),

        "rcvTime": get_first(row, ["rcvTime", "receiveTime", "receiverTime"]),
        "sendTime": get_first(row, ["sendTime", "timestamp", "time"]),
        "sender_id": get_first(row, ["sender_id", "sender.id", "id"]),
        "sender_alias": get_first(row, ["sender_alias", "sender.alias"]),
        "messageID": get_first(row, ["messageID", "message_id", "msg_id"]),

        "sender_pos_x": sx,
        "sender_pos_y": sy,
        "sender_pos_z": sz,
        "sender_pos_noise": get_first(row, ["sender.pos_noise", "pos_noise"]),

        "sender_spd": get_first(row, ["sender.spd", "spd", "speed"]),
        "sender_spd_noise": get_first(row, ["sender.spd_noise", "spd_noise"]),
        "sender_acl": get_first(row, ["sender.acl", "acl", "acceleration"]),
        "sender_acl_noise": get_first(row, ["sender.acl_noise", "acl_noise"]),
        "sender_hed": get_first(row, ["sender.hed", "hed", "heading"]),
        "sender_hed_noise": get_first(row, ["sender.hed_noise", "hed_noise"]),
        "sender_driver_profile": get_first(row, ["sender.driversProfile", "driversProfile"]),

        "receiver_pos_x": rx,
        "receiver_pos_y": ry,
        "receiver_pos_z": rz,
        "receiver_pos_noise": get_first(row, ["receiver.pos_noise"]),
        "receiver_spd": get_first(row, ["receiver.spd"]),
        "receiver_spd_noise": get_first(row, ["receiver.spd_noise"]),
        "receiver_acl": get_first(row, ["receiver.acl"]),
        "receiver_acl_noise": get_first(row, ["receiver.acl_noise"]),
        "receiver_hed": get_first(row, ["receiver.hed"]),
        "receiver_hed_noise": get_first(row, ["receiver.hed_noise"]),
        "receiver_driver_profile": get_first(row, ["receiver.driversProfile"]),
    }

    # Time-derived features
    try:
        out["delay"] = float(out["rcvTime"]) - float(out["sendTime"])
    except Exception:
        out["delay"] = None

    # Kinematic residual features useful for V2X attack detection
    try:
        out["speed_abs"] = abs(float(out["sender_spd"]))
    except Exception:
        out["speed_abs"] = None

    try:
        out["accel_abs"] = abs(float(out["sender_acl"]))
    except Exception:
        out["accel_abs"] = None

    return out


def read_records_from_file(path: Path) -> List[Dict[str, Any]]:
    ext = path.suffix.lower()

    if ext == ".json":
        return load_json_records(path)

    if ext in [".jsonl", ".ndjson"]:
        return load_jsonl_records(path)

    if ext == ".csv":
        try:
            return pd.read_csv(path).to_dict("records")
        except Exception as e:
            print(f"[WARN] CSV read failed: {path} | {e}")
            return []

    return []


def should_keep(path: Path, scenario: str, density: str) -> bool:
    ctx = infer_context(path)
    if scenario and ctx["scenario"] and ctx["scenario"] != scenario:
        return False
    if density and ctx["density"] and ctx["density"] != density:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="outputs_ml_ready")
    parser.add_argument("--scenario", default="urban", choices=["", "urban", "highway"])
    parser.add_argument("--density", default="2")
    parser.add_argument("--max-records-per-file", type=int, default=0,
                        help="0 means no limit. Use e.g. 50000 for quick testing.")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Use existing extracted folder if already extracted.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    extract_dir = out_dir / "_extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    if not args.skip_extract:
        zip_files = sorted(data_dir.glob("*.zip"))
        print(f"[INFO] Found {len(zip_files)} top-level ZIP files.")
        for zp in tqdm(zip_files, desc="Extracting nested ZIPs"):
            if should_keep(zp, args.scenario, args.density):
                recursive_extract_zip(zp, extract_dir)

    # Search both extracted data and direct top-level JSON ground truth
    search_roots = [extract_dir, data_dir]
    candidate_files = []
    for root in search_roots:
        if root.exists():
            for ext in ["*.json", "*.jsonl", "*.ndjson", "*.csv"]:
                candidate_files.extend(root.rglob(ext))

    # Remove generated output CSVs if rerun
    candidate_files = [
        p for p in candidate_files
        if "outputs_ml_ready" not in str(p) and should_keep(p, args.scenario, args.density)
    ]

    print(f"[INFO] Candidate readable files: {len(candidate_files)}")

    all_rows = []
    source_summary = []

    for fp in tqdm(candidate_files, desc="Reading records"):
        records = read_records_from_file(fp)
        if args.max_records_per_file and len(records) > args.max_records_per_file:
            records = records[:args.max_records_per_file]

        if not records:
            continue

        norm_rows = [normalize_record(r, fp) for r in records]
        all_rows.extend(norm_rows)

        ctx = infer_context(fp)
        source_summary.append({
            "source_file": str(fp),
            "records": len(norm_rows),
            **ctx,
        })

    if not all_rows:
        print("[ERROR] No records were loaded. Check whether nested ZIP extraction produced JSON/CSV files.")
        return

    df = pd.DataFrame(all_rows)

    # Basic cleanup
    numeric_cols = [
        "rcvTime", "sendTime", "sender_alias", "messageID",
        "sender_pos_x", "sender_pos_y", "sender_pos_z",
        "sender_spd", "sender_acl", "sender_hed",
        "receiver_pos_x", "receiver_pos_y", "receiver_pos_z",
        "receiver_spd", "receiver_acl", "receiver_hed",
        "delay", "speed_abs", "accel_abs",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save outputs
    full_csv = out_dir / "veremi_ml_ready_all.csv"
    df.to_csv(full_csv, index=False)

    sample_csv = out_dir / "veremi_ml_ready_sample_100k.csv"
    df.sample(min(100000, len(df)), random_state=42).to_csv(sample_csv, index=False)

    summary_df = pd.DataFrame(source_summary)
    summary_df.to_csv(out_dir / "source_record_summary.csv", index=False)

    dist = (
        df.groupby(["scenario", "density", "split", "attack_type", "label"], dropna=False)
          .size()
          .reset_index(name="records")
          .sort_values(["scenario", "density", "split", "attack_type"])
    )
    dist.to_csv(out_dir / "ml_ready_distribution.csv", index=False)

    feature_report = {
        "total_records": int(len(df)),
        "total_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "label_counts": df["label"].value_counts(dropna=False).to_dict(),
        "attack_counts": df["attack_type"].value_counts(dropna=False).to_dict(),
        "split_counts": df["split"].value_counts(dropna=False).to_dict(),
    }

    with (out_dir / "ml_ready_report.json").open("w", encoding="utf-8") as f:
        json.dump(feature_report, f, indent=2)

    md = []
    md.append("# ML-Ready VeReMi Dataset Report\n")
    md.append(f"- Total records: **{len(df):,}**")
    md.append(f"- Total columns: **{df.shape[1]}**")
    md.append("\n## Label Counts\n")
    md.append(df["label"].value_counts(dropna=False).to_markdown())
    md.append("\n## Attack Counts\n")
    md.append(df["attack_type"].value_counts(dropna=False).to_markdown())
    md.append("\n## Split Counts\n")
    md.append(df["split"].value_counts(dropna=False).to_markdown())
    md.append("\n## Distribution\n")
    md.append(dist.to_markdown(index=False))
    (out_dir / "ml_ready_report.md").write_text("\n".join(md), encoding="utf-8")

    print("\n[DONE] ML-ready dataset created.")
    print(f"[OUT] {out_dir.resolve()}")
    print("Created:")
    print(" - veremi_ml_ready_all.csv")
    print(" - veremi_ml_ready_sample_100k.csv")
    print(" - source_record_summary.csv")
    print(" - ml_ready_distribution.csv")
    print(" - ml_ready_report.md")
    print(" - ml_ready_report.json")


if __name__ == "__main__":
    main()
