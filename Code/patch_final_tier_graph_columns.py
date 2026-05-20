#!/usr/bin/env python
"""
Patch final_tier_agentic_v2xshield.py for:
KeyError: "['gnn_neighbor_rule_mean'] not in index"

Usage:
    python patch_final_tier_graph_columns.py --script "D:\\other\\Agentic-V2XShield\\final_tier_agentic_v2xshield.py"
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
backup = p.with_suffix(".backup_graph_columns.py")
backup.write_text(text, encoding="utf-8")

old = 'return df.merge(graph[keep], on=["split", "sender_id"], how="left")'

new = """    # Safety: some graph/GNN columns may be absent depending on receiver proxy structure.
    # Create missing columns so feature sets remain stable across VeReMi variants.
    for col in keep:
        if col not in graph.columns:
            graph[col] = 0.0

    return df.merge(graph[keep], on=["split", "sender_id"], how="left")"""

if old not in text:
    raise RuntimeError("Could not find the graph merge line to patch.")

text = text.replace(old, new)
p.write_text(text, encoding="utf-8")

print("[DONE] Patched graph/GNN missing-column handling.")
print(f"[SCRIPT] {p}")
print(f"[BACKUP] {backup}")
