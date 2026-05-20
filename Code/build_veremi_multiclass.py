#!/usr/bin/env python
"""
VeReMi-NextGen multiclass dataset builder.

This script builds a multiclass ML table from multiple attack folders, where each
folder name becomes the class label.

Example class mapping:
    normal                    -> 0
    constantPositionOffset    -> 1
    randomPositionOffset      -> 2
    trafficCongestionSybil    -> 3
    timeDelayAttack           -> 4
    dataReplay                -> 5
    dosAttack                 -> 6

Important:
    - Binary label uses the message-level "attacker" field when available.
    - Multiclass label uses the attack folder name.
    - For benign base folder InTAS_urban_2, class_name = normal.

Usage:
    python build_veremi_multiclass.py ^
      --data-dir "D:\other\Agentic-V2XShield\Dataset" ^
      --out-dir "outputs_multiclass" ^
      --scenario urban ^
      --density 2
"""

import argparse
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    text = str(path).replace("\\", "/")

    scenario = ""
    density = ""
    split = ""
    class_name = "normal"

    m = re.search(r"InTAS_(urban|highway)_(\d+)", text)
    if m:
        scenario = m.group(1)
        density = m.group(2)

    for sp in ["Train", "Validation", "Test", "train", "validation", "test"]:
        if f"/{sp}/" in text or f"_{sp}_" in text:
            split = sp.lower()
            break

    # Folder name decides multiclass label.
    for attack in ATTACK_NAMES:
        if attack in text:
            class_name = attack
            break

    return {
        "scenario": scenario,
        "density": density,
        "split": split,
        "class_name": class_name,
    }


def recursive_extract_zip(zip_path: Path, extract_root: Path, max_rounds: int = 10) -> None:
    target = extract_root / zip_path.stem
    marker = target / ".extracted_ok"

    if not marker.exists():
        target.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target)
            marker.write_text("ok", encoding="utf-8")
        except zipfile.BadZipFile:
            print(f"[WARN] Bad zip skipped: {zip_path}")
            return

    for _ in range(max_rounds):
        zips = list(target.rglob("*.zip"))
        new_count = 0

        for zp in zips:
            nested_target = zp.parent / zp.stem
            nested_marker = nested_target / ".extracted_ok"
            if nested_marker.exists():
                continue

            nested_target.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(nested_target)
                nested_marker.write_text("ok", encoding="utf-8")
                new_count += 1
            except zipfile.BadZipFile:
                print(f"[WARN] Bad nested zip skipped: {zp}")

        if new_count == 0:
            break


def parse_position(value: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if value is None:
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
            vals = [float(v) for v in value[:3]]
            while len(vals) < 3:
                vals.append(None)
            return vals[0], vals[1], vals[2]
        except Exception:
            return None, None, None

    return None, None, None


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def heading_diff_deg(a: Any, b: Any) -> Optional[float]:
    a = to_float(a)
    b = to_float(b)
    if a is None or b is None:
        return None
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def euclidean_2d(x1, y1, x2, y2) -> Optional[float]:
    vals = [to_float(v) for v in [x1, y1, x2, y2]]
    if any(v is None for v in vals):
        return None
    return math.sqrt((vals[0] - vals[2]) ** 2 + (vals[1] - vals[3]) ** 2)


def load_json_file(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="latin-1") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return []

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        rows = []
        for k, v in data.items():
            if isinstance(v, dict):
                r = dict(v)
                r["_record_id"] = k
                rows.append(r)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        r = dict(item)
                        r["_parent_key"] = k
                        rows.append(r)
        return rows

    return []


def normalize_record(rec: Dict[str, Any], path: Path) -> Dict[str, Any]:
    ctx = infer_context(path)

    sender = rec.get("sender", {}) if isinstance(rec.get("sender", {}), dict) else {}
    receiver = rec.get("receiver", {}) if isinstance(rec.get("receiver", {}), dict) else {}

    sx, sy, sz = parse_position(sender.get("pos"))
    rx, ry, rz = parse_position(receiver.get("pos"))
    snx, sny, snz = parse_position(sender.get("pos_noise"))
    rnx, rny, rnz = parse_position(receiver.get("pos_noise"))

    attacker = rec.get("attacker", None)
    attacker_val = to_float(attacker)

    # Message-level binary label. If attacker is missing in normal records, set normal as 0.
    if attacker_val is None:
        binary_label = 0 if ctx["class_name"] == "normal" else 1
    else:
        binary_label = int(attacker_val)

    send_time = to_float(rec.get("sendTime"))
    rcv_time = to_float(rec.get("rcvTime"))
    delay = None if send_time is None or rcv_time is None else rcv_time - send_time

    sender_spd = to_float(sender.get("spd"))
    receiver_spd = to_float(receiver.get("spd"))
    sender_acl = to_float(sender.get("acl"))
    receiver_acl = to_float(receiver.get("acl"))
    sender_hed = to_float(sender.get("hed"))
    receiver_hed = to_float(receiver.get("hed"))

    out = {
        "scenario": ctx["scenario"],
        "density": ctx["density"],
        "split": ctx["split"],
        "class_name": ctx["class_name"],
        "binary_label": binary_label,
        "source_file": str(path),

        "rcvTime": rcv_time,
        "sendTime": send_time,
        "delay": delay,

        "sender_id": rec.get("sender_id"),
        "sender_alias": rec.get("sender_alias"),
        "messageID": rec.get("messageID"),
        "attacker_raw": attacker,

        "sender_pos_x": sx,
        "sender_pos_y": sy,
        "sender_pos_z": sz,
        "receiver_pos_x": rx,
        "receiver_pos_y": ry,
        "receiver_pos_z": rz,

        "sender_pos_noise_x": snx,
        "sender_pos_noise_y": sny,
        "sender_pos_noise_z": snz,
        "receiver_pos_noise_x": rnx,
        "receiver_pos_noise_y": rny,
        "receiver_pos_noise_z": rnz,

        "sender_spd": sender_spd,
        "receiver_spd": receiver_spd,
        "sender_spd_noise": to_float(sender.get("spd_noise")),
        "receiver_spd_noise": to_float(receiver.get("spd_noise")),

        "sender_acl": sender_acl,
        "receiver_acl": receiver_acl,
        "sender_acl_noise": to_float(sender.get("acl_noise")),
        "receiver_acl_noise": to_float(receiver.get("acl_noise")),

        "sender_hed": sender_hed,
        "receiver_hed": receiver_hed,
        "sender_hed_noise": to_float(sender.get("hed_noise")),
        "receiver_hed_noise": to_float(receiver.get("hed_noise")),

        "sender_driver_profile": sender.get("driversProfile"),
        "receiver_driver_profile": receiver.get("driversProfile"),
        "distance_to_road_edge": to_float(sender.get("distance_to_road_edge")),
    }

    # Engineered features
    out["sender_receiver_distance"] = euclidean_2d(sx, sy, rx, ry)
    out["speed_delta"] = None if sender_spd is None or receiver_spd is None else abs(sender_spd - receiver_spd)
    out["accel_delta"] = None if sender_acl is None or receiver_acl is None else abs(sender_acl - receiver_acl)
    out["heading_delta"] = heading_diff_deg(sender_hed, receiver_hed)
    out["edge_violation"] = None if out["distance_to_road_edge"] is None else int(out["distance_to_road_edge"] < 0)
    out["abs_sender_speed"] = None if sender_spd is None else abs(sender_spd)
    out["abs_sender_accel"] = None if sender_acl is None else abs(sender_acl)

    return out


def keep_top_level_zip(zp: Path, scenario: str, density: str) -> bool:
    ctx = infer_context(zp)
    if scenario and ctx["scenario"] and ctx["scenario"] != scenario:
        return False
    if density and ctx["density"] and ctx["density"] != density:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="outputs_multiclass")
    parser.add_argument("--scenario", default="urban", choices=["", "urban", "highway"])
    parser.add_argument("--density", default="2")
    parser.add_argument("--max-files-per-class", type=int, default=0,
                        help="0 means all files. Use 20 for quick debugging.")
    parser.add_argument("--max-records-per-file", type=int, default=0,
                        help="0 means all records. Use 2000 for quick debugging.")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    extract_dir = out_dir / "_extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        zip_files = [z for z in sorted(data_dir.glob("*.zip")) if keep_top_level_zip(z, args.scenario, args.density)]
        print(f"[INFO] ZIP files selected: {len(zip_files)}")
        for zp in tqdm(zip_files, desc="Extracting ZIPs"):
            recursive_extract_zip(zp, extract_dir)

    json_files = []
    for root in [extract_dir]:
        json_files.extend(root.rglob("veh_*.json"))

    json_files = [p for p in json_files if keep_top_level_zip(p, args.scenario, args.density)]

    # Group by class folder.
    grouped = {}
    for p in json_files:
        cls = infer_context(p)["class_name"]
        grouped.setdefault(cls, []).append(p)

    print("[INFO] JSON vehicle files by class:")
    for cls, files in sorted(grouped.items()):
        print(f"  {cls:30s}: {len(files)}")

    selected_files = []
    for cls, files in sorted(grouped.items()):
        files = sorted(files)
        if args.max_files_per_class and len(files) > args.max_files_per_class:
            files = files[:args.max_files_per_class]
        selected_files.extend(files)

    class_names = sorted(grouped.keys())
    if "normal" in class_names:
        class_names.remove("normal")
        class_names = ["normal"] + class_names

    class_to_id = {c: i for i, c in enumerate(class_names)}

    rows = []
    source_rows = []

    for fp in tqdm(selected_files, desc="Reading vehicle JSON files"):
        records = load_json_file(fp)
        if args.max_records_per_file and len(records) > args.max_records_per_file:
            records = records[:args.max_records_per_file]

        ctx = infer_context(fp)
        loaded = 0

        for rec in records:
            row = normalize_record(rec, fp)
            row["class_id"] = class_to_id[row["class_name"]]
            rows.append(row)
            loaded += 1

        source_rows.append({
            "source_file": str(fp),
            "class_name": ctx["class_name"],
            "class_id": class_to_id.get(ctx["class_name"], -1),
            "split": ctx["split"],
            "records": loaded,
        })

    if not rows:
        print("[ERROR] No rows loaded.")
        return

    df = pd.DataFrame(rows)

    # Save class mapping
    mapping_df = pd.DataFrame(
        [{"class_name": k, "class_id": v} for k, v in class_to_id.items()]
    ).sort_values("class_id")
    mapping_df.to_csv(out_dir / "class_mapping.csv", index=False)

    # Save complete data
    df.to_csv(out_dir / "veremi_multiclass_all.csv", index=False)

    # Save balanced dataset by class for fair ML experiments
    min_count = int(df["class_name"].value_counts().min())
    balanced = (
        df.groupby("class_name", group_keys=False)
          .apply(lambda x: x.sample(min_count, random_state=42))
          .reset_index(drop=True)
    )
    balanced.to_csv(out_dir / "veremi_multiclass_balanced.csv", index=False)

    # Save reports
    pd.DataFrame(source_rows).to_csv(out_dir / "source_file_summary.csv", index=False)

    dist = (
        df.groupby(["scenario", "density", "split", "class_name", "class_id", "binary_label"], dropna=False)
          .size()
          .reset_index(name="records")
          .sort_values(["class_id", "split", "binary_label"])
    )
    dist.to_csv(out_dir / "multiclass_distribution.csv", index=False)

    md = []
    md.append("# VeReMi-NextGen Multiclass Dataset Report\n")
    md.append(f"- Total records: **{len(df):,}**")
    md.append(f"- Total features/columns: **{df.shape[1]}**")
    md.append(f"- Number of classes: **{len(class_to_id)}**")
    md.append(f"- Balanced records per class: **{min_count:,}**")
    md.append(f"- Balanced total records: **{len(balanced):,}**")

    md.append("\n## Class Mapping\n")
    md.append(mapping_df.to_markdown(index=False))

    md.append("\n## Class Counts\n")
    md.append(df["class_name"].value_counts().to_frame("records").to_markdown())

    md.append("\n## Binary Label Counts\n")
    md.append(df["binary_label"].value_counts().to_frame("records").to_markdown())

    md.append("\n## Split Counts\n")
    md.append(df["split"].value_counts().to_frame("records").to_markdown())

    md.append("\n## Full Distribution\n")
    md.append(dist.to_markdown(index=False))

    (out_dir / "multiclass_report.md").write_text("\n".join(md), encoding="utf-8")

    print("\n[DONE] Multiclass dataset created.")
    print(f"[OUT] {out_dir.resolve()}")
    print("Created:")
    print(" - veremi_multiclass_all.csv")
    print(" - veremi_multiclass_balanced.csv")
    print(" - class_mapping.csv")
    print(" - multiclass_distribution.csv")
    print(" - multiclass_report.md")
    print(" - source_file_summary.csv")


if __name__ == "__main__":
    main()
