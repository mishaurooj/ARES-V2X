#!/usr/bin/env python
"""
Fix AECTE++ pipeline KeyError: 'setting'

Cause:
    In run_experiments(), the script accidentally used:
        metrics.append(m)
    instead of:
        metrics.append(met)

Usage:
    python fix_aectepp_script.py --script "D:\other\Agentic-V2XShield\aectepp_llm_agentic_pipeline.py"
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--script", required=True)
args = parser.parse_args()

p = Path(args.script)
if not p.exists():
    raise FileNotFoundError(p)

text = p.read_text(encoding="utf-8")

bad = "metrics.append(m)\n            responses.append(resp)"
good = "metrics.append(met)\n            responses.append(resp)"

if bad not in text:
    print("[INFO] Exact bug line not found. The script may already be fixed.")
else:
    text = text.replace(bad, good)
    backup = p.with_suffix(".backup_before_metrics_fix.py")
    backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    p.write_text(text, encoding="utf-8")
    print(f"[DONE] Fixed: {p}")
    print(f"[BACKUP] {backup}")
