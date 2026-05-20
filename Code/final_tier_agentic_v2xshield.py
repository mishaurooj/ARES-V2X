#!/usr/bin/env python
"""
Agentic-V2XShield Final-Tier Wrapper

This script extends final_plus_agentic_v2xshield.py with the three missing
publication-level priorities:

P1. Stronger robustness stress tests:
    packet loss, feature dropout, GPS corruption, delay injection,
    timestamp desynchronization, stale trust, and edge-cloud outage.

P2. Lightweight GNN-style baseline:
    GraphSAGE-Tabular baseline using node features + neighbor aggregate
    graph features without requiring PyTorch/PyG.

P3. Explainability visualizations:
    trust propagation, graph disagreement, temporal risk evolution,
    robustness degradation, scalability, and LLM policy triage.

Place this file in the same folder as final_plus_agentic_v2xshield.py.

Run:
    python final_tier_agentic_v2xshield.py ^
      --csv "outputs_multiclass\\veremi_multiclass_balanced.csv" ^
      --out-dir "outputs_final_tier" ^
      --max-per-class 100000 ^
      --llm-provider ollama ^
      --llm-model llama3.2:3b

Fast test:
    python final_tier_agentic_v2xshield.py ^
      --csv "outputs_multiclass\\veremi_multiclass_balanced.csv" ^
      --out-dir "outputs_final_tier_test" ^
      --max-per-class 30000 ^
      --llm-provider none
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import final_plus_agentic_v2xshield as base


GNN_BASELINE_FEATURES = [
    "gnn_neighbor_speed_mean", "gnn_neighbor_accel_mean",
    "gnn_neighbor_heading_delta_mean", "gnn_neighbor_rule_mean",
    "gnn_neighbor_trust_mean", "gnn_neighbor_degree_mean",
    "gnn_feature_disagreement", "gnn_sender_degree_log",
]


def engineer_graph_trust_tier(df, out_dir):
    """Graph trust + GraphSAGE-style neighborhood aggregate features."""
    df = df.copy()
    if "receiver_id" not in df.columns:
        if "receiver_driver_profile" in df.columns:
            df["receiver_id"] = "profile_" + df["receiver_driver_profile"].astype(str)
        else:
            df["receiver_id"] = "unknown_receiver"

    rows = []
    for split, part in df.groupby("split", sort=False):
        sender_stats = (
            part.groupby("sender_id")
            .agg(
                sender_binary_rate=("binary_label", "mean"),
                sender_count=("binary_label", "size"),
                sender_rule_mean=("risk_rule_score", "mean"),
                sender_speed_mean=("sender_spd", "mean"),
                sender_accel_mean=("sender_acl", "mean"),
                sender_heading_delta_mean=("heading_delta", "mean"),
                sender_trust_mean=("sender_trust_prior", "mean"),
            )
            .reset_index()
        )

        edges = part.groupby(["sender_id", "receiver_id"]).size().reset_index(name="edge_weight")
        neigh = edges.merge(
            sender_stats.rename(columns={
                "sender_id": "receiver_id",
                "sender_binary_rate": "neighbor_binary_rate",
                "sender_rule_mean": "neighbor_rule_mean",
                "sender_speed_mean": "neighbor_speed_mean",
                "sender_accel_mean": "neighbor_accel_mean",
                "sender_heading_delta_mean": "neighbor_heading_delta_mean",
                "sender_trust_mean": "neighbor_trust_mean",
                "sender_count": "neighbor_count",
            }),
            on="receiver_id", how="left",
        )

        for c in [
            "neighbor_binary_rate", "neighbor_rule_mean", "neighbor_speed_mean",
            "neighbor_accel_mean", "neighbor_heading_delta_mean",
            "neighbor_trust_mean", "neighbor_count",
        ]:
            neigh[c] = neigh[c].fillna(0)
            neigh[f"weighted_{c}"] = neigh[c] * neigh["edge_weight"]

        agg = (
            neigh.groupby("sender_id")
            .agg(
                graph_sender_degree_prior=("receiver_id", "nunique"),
                graph_neighbor_risk_prior=("weighted_neighbor_binary_rate", "sum"),
                graph_neighbor_rule_prior=("weighted_neighbor_rule_mean", "sum"),
                gnn_neighbor_speed_mean=("weighted_neighbor_speed_mean", "sum"),
                gnn_neighbor_accel_mean=("weighted_neighbor_accel_mean", "sum"),
                gnn_neighbor_heading_delta_mean=("weighted_neighbor_heading_delta_mean", "sum"),
                gnn_neighbor_trust_mean=("weighted_neighbor_trust_mean", "sum"),
                gnn_neighbor_degree_mean=("weighted_neighbor_count", "sum"),
                graph_edge_weight_sum=("edge_weight", "sum"),
            )
            .reset_index()
        )

        for c in [
            "graph_neighbor_risk_prior", "graph_neighbor_rule_prior",
            "gnn_neighbor_speed_mean", "gnn_neighbor_accel_mean",
            "gnn_neighbor_heading_delta_mean", "gnn_neighbor_trust_mean",
            "gnn_neighbor_degree_mean",
        ]:
            agg[c] = agg[c] / agg["graph_edge_weight_sum"].replace(0, np.nan)

        sender_stats = sender_stats.merge(agg, on="sender_id", how="left")
        sender_stats["graph_trust_propagated"] = (
            1.0
            - 0.55 * sender_stats["sender_binary_rate"].fillna(0)
            - 0.25 * sender_stats["graph_neighbor_risk_prior"].fillna(0)
            - 0.20 * sender_stats["sender_rule_mean"].fillna(0)
        ).clip(0, 1)
        sender_stats["graph_local_disagreement"] = (
            sender_stats["sender_binary_rate"].fillna(0)
            - sender_stats["graph_neighbor_risk_prior"].fillna(0)
        ).abs()
        sender_stats["gnn_feature_disagreement"] = (
            (sender_stats["sender_speed_mean"].fillna(0) - sender_stats["gnn_neighbor_speed_mean"].fillna(0)).abs()
            + (sender_stats["sender_accel_mean"].fillna(0) - sender_stats["gnn_neighbor_accel_mean"].fillna(0)).abs()
            + (sender_stats["sender_heading_delta_mean"].fillna(0) - sender_stats["gnn_neighbor_heading_delta_mean"].fillna(0)).abs()
        )
        sender_stats["gnn_sender_degree_log"] = np.log1p(sender_stats["graph_sender_degree_prior"].fillna(0))
        sender_stats["split"] = split
        rows.append(sender_stats)

    graph = pd.concat(rows, ignore_index=True)
    graph.to_csv(out_dir / "csv" / "graph_trust_summary.csv", index=False)

    keep = ["split", "sender_id"] + base.GRAPH_TRUST_FEATURES + GNN_BASELINE_FEATURES
        # Safety: some graph/GNN columns may be absent depending on receiver proxy structure.
    # Create missing columns so feature sets remain stable across VeReMi variants.
    for col in keep:
        if col not in graph.columns:
            graph[col] = 0.0

    return df.merge(graph[keep], on=["split", "sender_id"], how="left")


def stress_df_tier(df, mode, level, seed=42):
    """Stronger, reviewer-facing stress tests."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    test_mask = out["split"].astype(str).str.lower() == "test"
    idx = out.index[test_mask].to_numpy()

    if mode == "packet_loss":
        drop_n = int(len(idx) * level)
        if drop_n > 0:
            drop_idx = rng.choice(idx, size=drop_n, replace=False)
            out = out.drop(index=drop_idx).reset_index(drop=True)

    elif mode == "feature_dropout":
        cols = base.EDGE_FEATURES + base.CLOUD_FEATURES + base.TEMPORAL_FEATURES
        for col in cols:
            if col in out.columns:
                miss = test_mask & (rng.random(len(out)) < level)
                out.loc[miss, col] = np.nan

    elif mode == "gps_corruption_m":
        for col in ["sender_pos_x", "sender_pos_y", "receiver_pos_x", "receiver_pos_y"]:
            if col in out.columns:
                out.loc[test_mask, col] = out.loc[test_mask, col] + rng.normal(0, level, size=test_mask.sum())
        if "sender_receiver_distance" in out.columns:
            out.loc[test_mask, "sender_receiver_distance"] = (
                out.loc[test_mask, "sender_receiver_distance"] + rng.normal(0, abs(level), size=test_mask.sum())
            )

    elif mode == "delay_injection":
        for col in ["delay", "delay_roll_mean_5", "sender_delay_mean_prior"]:
            if col in out.columns:
                out.loc[test_mask, col] = out.loc[test_mask, col].fillna(0) + level

    elif mode == "timestamp_desync":
        if "msg_time_gap" in out.columns:
            out.loc[test_mask, "msg_time_gap"] = out.loc[test_mask, "msg_time_gap"].fillna(0) + rng.normal(level, level / 2, size=test_mask.sum())
        if "sender_pos_step_dist" in out.columns:
            out.loc[test_mask, "sender_pos_step_dist"] = out.loc[test_mask, "sender_pos_step_dist"].fillna(0) * (1 + level / 100.0)

    elif mode == "stale_trust":
        trust_cols = base.TRUST_FEATURES + base.GRAPH_TRUST_FEATURES + GNN_BASELINE_FEATURES
        for col in trust_cols:
            if col in out.columns:
                shuffled = out.loc[test_mask, col].sample(frac=1.0, random_state=seed).values
                replace_mask = test_mask & (rng.random(len(out)) < level)
                out.loc[replace_mask, col] = shuffled[:replace_mask.sum()]

    elif mode == "edge_cloud_outage":
        outage_cols = base.CLOUD_FEATURES + base.GRAPH_TRUST_FEATURES + GNN_BASELINE_FEATURES
        for col in outage_cols:
            if col in out.columns:
                miss = test_mask & (rng.random(len(out)) < level)
                out.loc[miss, col] = np.nan
    else:
        raise ValueError(mode)

    return out


def run_all_tier(df, out_dir, llm_provider, llm_model):
    raw_features = list(dict.fromkeys(base.EDGE_FEATURES + base.CLOUD_FEATURES))
    enhanced_features = list(dict.fromkeys(base.EDGE_FEATURES + base.CLOUD_FEATURES + base.TEMPORAL_FEATURES + base.TRUST_FEATURES + base.GRAPH_TRUST_FEATURES))
    gnn_features = list(dict.fromkeys(base.EDGE_FEATURES + base.CLOUD_FEATURES + base.GRAPH_TRUST_FEATURES + GNN_BASELINE_FEATURES))
    proposed_features = list(dict.fromkeys(enhanced_features + GNN_BASELINE_FEATURES))

    five_baselines = ["LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting"]
    if base.HAS_XGB:
        five_baselines.append("XGBoost")

    detection_rows, response_rows, saved_models = [], [], {}

    for task, target in [("binary", "binary_label"), ("multiclass", "class_id")]:
        for model_name in five_baselines:
            print(f"[BASELINE RAW] {task} | {model_name}")
            det, resp, _ = base.train_eval(df, target, task, "S0_raw_edge_cloud", model_name, raw_features, out_dir)
            detection_rows.append(det); response_rows.append(resp)

        for model_name in five_baselines:
            print(f"[BASELINE ENHANCED] {task} | {model_name}")
            det, resp, _ = base.train_eval(df, target, task, "S1_enhanced_temporal_trust_graph", model_name, enhanced_features, out_dir)
            detection_rows.append(det); response_rows.append(resp)

        print(f"[GNN BASELINE] {task} | GraphSAGE-Tabular")
        det, resp, _ = base.train_eval(df, target, task, "S2_graphsage_tabular", "RandomForest", gnn_features, out_dir)
        det["model"] = "GraphSAGE-Tabular"; resp["model"] = "GraphSAGE-Tabular"
        detection_rows.append(det); response_rows.append(resp)

        ablations = {
            "A1_edge_only": base.EDGE_FEATURES,
            "A2_cloud_only": base.CLOUD_FEATURES,
            "A3_temporal_only": base.TEMPORAL_FEATURES,
            "A4_trust_only": base.TRUST_FEATURES,
            "A5_graph_trust_only": base.GRAPH_TRUST_FEATURES + GNN_BASELINE_FEATURES,
            "A6_edge_cloud_temporal": list(dict.fromkeys(base.EDGE_FEATURES + base.CLOUD_FEATURES + base.TEMPORAL_FEATURES)),
            "A7_full_AECTEpp": proposed_features,
        }
        for setting, feats in ablations.items():
            print(f"[PROPOSED] {task} | {setting}")
            det, resp, model = base.train_eval(df, target, task, setting, "AECTE++", feats, out_dir, proposed=True)
            detection_rows.append(det); response_rows.append(resp)
            if setting == "A7_full_AECTEpp":
                saved_models[(task, "AECTE++")] = model
                joblib.dump(model, out_dir / "models" / f"{task}_AECTEpp.joblib")

    det_df = pd.DataFrame(detection_rows)
    resp_df = pd.DataFrame(response_rows)
    det_df.to_csv(out_dir / "csv" / "all_detection_metrics.csv", index=False)
    resp_df.to_csv(out_dir / "csv" / "response_resilience_metrics.csv", index=False)

    robust_rows = []
    proposed_model = saved_models.get(("binary", "AECTE++"))
    if proposed_model is not None:
        stress_plan = {
            "packet_loss": [0.10, 0.20, 0.30, 0.50],
            "feature_dropout": [0.10, 0.20, 0.30, 0.50],
            "gps_corruption_m": [25.0, 50.0, 100.0, 250.0],
            "delay_injection": [100.0, 250.0, 500.0, 1000.0],
            "timestamp_desync": [25.0, 50.0, 100.0, 200.0],
            "stale_trust": [0.10, 0.20, 0.30, 0.50],
            "edge_cloud_outage": [0.10, 0.20, 0.30, 0.50],
        }
        for mode, levels in stress_plan.items():
            for level in levels:
                print(f"[ROBUSTNESS] {mode} | {level}")
                sdf = stress_df_tier(df, mode, level)
                _, _, _, _, X_test, y_test = base.split_xy(sdf, "binary_label", proposed_features)
                t0 = time.perf_counter(); pred = proposed_model.predict(X_test); infer_s = time.perf_counter() - t0
                row = base.evaluate(y_test, pred, "binary", f"{mode}_{level}", "AECTE++", 0.0, infer_s)
                row["stress_type"] = mode; row["stress_level"] = level
                robust_rows.append(row)
    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(out_dir / "csv" / "robustness_metrics.csv", index=False)

    scale_rows = []
    if proposed_model is not None:
        _, _, _, _, X_test_full, y_test_full = base.split_xy(df, "binary_label", proposed_features)
        for n in [10000, 25000, 50000, 100000, min(197270, len(X_test_full))]:
            Xs, ys = X_test_full.head(n), y_test_full.head(n)
            t0 = time.perf_counter(); pred = proposed_model.predict(Xs); infer_s = time.perf_counter() - t0
            row = base.evaluate(ys, pred, "binary", f"N={n}", "AECTE++", 0.0, infer_s)
            row["sample_size"] = n
            scale_rows.append(row)
    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(out_dir / "csv" / "scalability_metrics.csv", index=False)

    multiclass_model = saved_models.get(("multiclass", "AECTE++"))
    if multiclass_model is not None:
        base.run_llm_incident_agent(df, multiclass_model, proposed_features, out_dir, llm_provider, llm_model, top_k=50)

    return det_df, resp_df, robust_df, scale_df, proposed_features


def make_tables_tier(det_df, resp_df, robust_df, scale_df, out_dir):
    comp = det_df[det_df["setting"].isin(["S0_raw_edge_cloud", "S1_enhanced_temporal_trust_graph", "S2_graphsage_tabular", "A7_full_AECTEpp"])].copy()
    comp = comp[["task", "setting", "model", "accuracy", "macro_f1", "weighted_f1", "mcc", "latency_ms_per_msg"]].sort_values(["task", "macro_f1"], ascending=[True, False])
    comp.to_csv(out_dir / "tables" / "baseline_comparison_table.csv", index=False)
    base.create_latex_table(comp, out_dir / "tables" / "baseline_comparison_table.tex", "Comparative detection performance of baselines, graph baseline, and AECTE++.", "tab:baseline_comparison")

    ablation = det_df[det_df["model"].eq("AECTE++")][["task", "setting", "accuracy", "macro_f1", "weighted_f1", "mcc", "latency_ms_per_msg"]].sort_values(["task", "setting"])
    ablation.to_csv(out_dir / "tables" / "ablation_table.csv", index=False)
    base.create_latex_table(ablation, out_dir / "tables" / "ablation_table.tex", "Ablation analysis of AECTE++ evidence groups.", "tab:aectepp_ablation")

    response = resp_df[["task", "setting", "model", "attack_coverage", "false_isolation_rate", "response_precision", "resilience_utility"]].sort_values(["task", "resilience_utility"], ascending=[True, False])
    response.to_csv(out_dir / "tables" / "response_resilience_table.csv", index=False)
    base.create_latex_table(response.head(35), out_dir / "tables" / "response_resilience_table.tex", "Cyber-response and resilience metrics.", "tab:response_resilience")

    if not robust_df.empty:
        robust_df.to_csv(out_dir / "tables" / "robustness_table.csv", index=False)
        base.create_latex_table(robust_df[["stress_type", "stress_level", "accuracy", "macro_f1", "mcc", "latency_ms_per_msg"]], out_dir / "tables" / "robustness_table.tex", "Robustness of AECTE++ under degraded V2X conditions.", "tab:robustness")
    if not scale_df.empty:
        scale_df.to_csv(out_dir / "tables" / "scalability_table.csv", index=False)
        base.create_latex_table(scale_df[["sample_size", "accuracy", "macro_f1", "mcc", "latency_ms_per_msg", "inference_s"]], out_dir / "tables" / "scalability_table.tex", "Scalability of AECTE++ under increasing V2X message volume.", "tab:scalability")


def make_figures_tier(det_df, resp_df, robust_df, scale_df, df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for ax, task, panel in zip(axes, ["binary", "multiclass"], ["(a)", "(b)"]):
        sub = det_df[(det_df["task"].eq(task)) & (det_df["setting"].isin(["S0_raw_edge_cloud", "S1_enhanced_temporal_trust_graph", "S2_graphsage_tabular", "A7_full_AECTEpp"]))].copy()
        sub["name"] = sub["model"] + "\n" + sub["setting"].replace({"S0_raw_edge_cloud":"raw", "S1_enhanced_temporal_trust_graph":"enh.", "S2_graphsage_tabular":"graph", "A7_full_AECTEpp":"prop."})
        sub = sub.sort_values("macro_f1", ascending=False).head(8)
        bars = ax.bar(sub["name"], sub["macro_f1"])
        ax.set_ylabel("Macro F1-score"); ax.set_xlabel("Method"); ax.set_title(f"{task.capitalize()} Comparative Analysis")
        ax.tick_params(axis="x", rotation=18); base.add_panel(ax, panel)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    base.savefig(fig, out_dir / "figures" / "fig1_comparative_baseline_graph_gnn")

    fig, ax = plt.subplots(figsize=(16, 9))
    ab = det_df[det_df["model"].eq("AECTE++")].copy()
    order = ["A1_edge_only", "A2_cloud_only", "A3_temporal_only", "A4_trust_only", "A5_graph_trust_only", "A6_edge_cloud_temporal", "A7_full_AECTEpp"]
    labels = ["Edge", "Cloud", "Temporal", "Trust", "Graph", "E+C+T", "Full"]
    pivot = ab.pivot_table(index="setting", columns="task", values="macro_f1", aggfunc="max").reindex(order)
    x = np.arange(len(order)); w = 0.36
    ax.bar(x-w/2, pivot["binary"], width=w, label="Binary"); ax.bar(x+w/2, pivot["multiclass"], width=w, label="Multiclass")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10); ax.set_ylabel("Macro F1-score"); ax.set_xlabel("AECTE++ configuration"); ax.set_title("AECTE++ Ablation Analysis"); ax.legend(); base.add_panel(ax, "(a)")
    base.savefig(fig, out_dir / "figures" / "fig2_aectepp_ablation")

    if not robust_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        for stress, part in robust_df.groupby("stress_type"):
            part = part.sort_values("stress_level")
            axes[0].plot(part["stress_level"], part["macro_f1"], marker="o", linewidth=2.5, label=stress)
            axes[1].plot(part["stress_level"], part["mcc"], marker="s", linewidth=2.5, label=stress)
        axes[0].set_xlabel("Stress level"); axes[0].set_ylabel("Binary macro F1-score"); axes[0].set_title("Robustness Degradation"); axes[0].legend(ncol=2); base.add_panel(axes[0], "(a)")
        axes[1].set_xlabel("Stress level"); axes[1].set_ylabel("MCC"); axes[1].set_title("Reliability Under Stress"); axes[1].legend(ncol=2); base.add_panel(axes[1], "(b)")
        base.savefig(fig, out_dir / "figures" / "fig3_robustness_degradation")

    if not scale_df.empty:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(scale_df["sample_size"], scale_df["latency_ms_per_msg"], marker="o", linewidth=3)
        ax.set_xlabel("Number of test messages"); ax.set_ylabel("Latency per message (ms)"); ax.set_title("AECTE++ Scalability"); base.add_panel(ax, "(a)")
        base.savefig(fig, out_dir / "figures" / "fig4_scalability")

    sample = df[df["split"].astype(str).str.lower().eq("test")].sample(min(50000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    axes[0].scatter(sample["sender_trust_prior"], sample["graph_trust_propagated"], s=8, alpha=0.25)
    axes[0].set_xlabel("Sender prior trust"); axes[0].set_ylabel("Graph-propagated trust"); axes[0].set_title("Trust Propagation Consistency"); base.add_panel(axes[0], "(a)")
    axes[1].scatter(sample["risk_rule_score"], sample["graph_local_disagreement"], s=8, alpha=0.25)
    axes[1].set_xlabel("Rule-based risk score"); axes[1].set_ylabel("Graph local disagreement"); axes[1].set_title("Graph Disagreement Explanation"); base.add_panel(axes[1], "(b)")
    base.savefig(fig, out_dir / "fig5_trust_graph_explainability")

    fig, ax = plt.subplots(figsize=(16, 9))
    top_senders = df[df["split"].astype(str).str.lower().eq("test")].groupby("sender_id")["risk_rule_score"].mean().sort_values(ascending=False).head(5).index
    for sid in top_senders:
        part = df[(df["split"].astype(str).str.lower().eq("test")) & (df["sender_id"].eq(sid))].head(200)
        ax.plot(range(len(part)), part["risk_rule_score"].rolling(10, min_periods=1).mean(), linewidth=2, label=str(sid))
    ax.set_xlabel("Message index"); ax.set_ylabel("Rolling risk score"); ax.set_title("Temporal Risk Evolution for High-Risk Senders"); ax.legend(); base.add_panel(ax, "(a)")
    base.savefig(fig, out_dir / "fig6_temporal_risk_evolution")

    llm_path = out_dir / "csv" / "llm_incident_reports.csv"
    if llm_path.exists():
        llm = pd.read_csv(llm_path)
        if not llm.empty:
            fig, axes = plt.subplots(1, 2, figsize=(16, 9))
            counts = llm["risk_level"].value_counts().reindex(["low", "medium", "high", "critical"]).fillna(0)
            bars = axes[0].bar(counts.index, counts.values)
            axes[0].set_xlabel("LLM-assigned risk level"); axes[0].set_ylabel("Incident count"); axes[0].set_title("LLM Policy Agent Triage"); base.add_panel(axes[0], "(a)")
            for b in bars:
                axes[0].text(b.get_x()+b.get_width()/2, b.get_height(), f"{int(b.get_height())}", ha="center", va="bottom")
            atk = llm["predicted_attack_type"].value_counts()
            axes[1].bar(atk.index, atk.values); axes[1].set_xlabel("Predicted attack type"); axes[1].set_ylabel("Incident count"); axes[1].set_title("LLM Incident Attack Composition"); axes[1].tick_params(axis="x", rotation=15); base.add_panel(axes[1], "(b)")
            base.savefig(fig, out_dir / "fig7_llm_policy_agent")


def write_report_tier(det_df, resp_df, robust_df, scale_df, out_dir):
    lines = []
    lines.append("# Agentic-V2XShield Final-Tier Report\n")
    lines.append("## Added Publication-Level Components\n")
    lines.append("- Stronger robustness stress testing.")
    lines.append("- Lightweight GraphSAGE-style tabular graph baseline.")
    lines.append("- Explainability figures for trust propagation, graph disagreement, risk evolution, and LLM policy triage.")
    lines.append("- CSV and LaTeX comparative tables.\n")
    lines.append("## Top Binary Results\n")
    lines.append(det_df[det_df["task"].eq("binary")].sort_values("macro_f1", ascending=False).head(12).to_markdown(index=False))
    lines.append("\n## Top Multiclass Results\n")
    lines.append(det_df[det_df["task"].eq("multiclass")].sort_values("macro_f1", ascending=False).head(12).to_markdown(index=False))
    lines.append("\n## Top Response Results\n")
    lines.append(resp_df.sort_values("resilience_utility", ascending=False).head(12).to_markdown(index=False))
    if not robust_df.empty:
        lines.append("\n## Robustness Summary\n")
        lines.append(robust_df[["stress_type", "stress_level", "accuracy", "macro_f1", "mcc"]].to_markdown(index=False))
    if not scale_df.empty:
        lines.append("\n## Scalability Summary\n")
        lines.append(scale_df[["sample_size", "macro_f1", "latency_ms_per_msg", "inference_s"]].to_markdown(index=False))
    llm_path = out_dir / "csv" / "llm_incident_reports.csv"
    if llm_path.exists():
        llm = pd.read_csv(llm_path)
        lines.append("\n## LLM Policy Agent Samples\n")
        lines.append(llm[["risk_level", "predicted_attack_type", "attack_probability", "policy_action"]].head(10).to_markdown(index=False))
    lines.append("\n## Recommended Claim\n")
    lines.append("Agentic-V2XShield should be positioned as a deployment-aware V2X cyber-resilience framework integrating edge-cloud evidence, temporal consistency, behavioral trust, graph-neighborhood reasoning, robustness testing, and LLM-assisted response orchestration. The LLM performs policy reasoning and explanation, while ML performs real-time detection.")
    (out_dir / "reports" / "final_tier_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default="outputs_final_tier")
    parser.add_argument("--max-per-class", type=int, default=100000)
    parser.add_argument("--keep-leakage-risk-features", action="store_true")
    parser.add_argument("--llm-provider", choices=["none", "ollama"], default="none")
    parser.add_argument("--llm-model", default="llama3.2:3b")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    base.ensure_dirs(out_dir)

    print("[INFO] Loading dataset.")
    df = base.load_dataset(Path(args.csv), args.max_per_class, args.keep_leakage_risk_features)
    print("[INFO] Engineering temporal and trust features.")
    df = base.engineer_temporal_trust(df)
    print("[INFO] Engineering graph + GraphSAGE-style features.")
    df = engineer_graph_trust_tier(df, out_dir)
    df.to_csv(out_dir / "csv" / "engineered_dataset_snapshot.csv", index=False)

    print("[INFO] Running comparative experiments, graph baseline, robustness, scalability, and LLM agent.")
    det_df, resp_df, robust_df, scale_df, _ = run_all_tier(df, out_dir, args.llm_provider, args.llm_model)

    print("[INFO] Creating tables.")
    make_tables_tier(det_df, resp_df, robust_df, scale_df, out_dir)
    print("[INFO] Creating figures.")
    make_figures_tier(det_df, resp_df, robust_df, scale_df, df, out_dir)
    print("[INFO] Writing report.")
    write_report_tier(det_df, resp_df, robust_df, scale_df, out_dir)

    print("\n[DONE] Final-tier Agentic-V2XShield pipeline completed.")
    print(f"[OUT] {out_dir.resolve()}")
    print("Key outputs:")
    print(" - reports/final_tier_report.md")
    print(" - tables/baseline_comparison_table.csv/.tex")
    print(" - tables/ablation_table.csv/.tex")
    print(" - tables/robustness_table.csv/.tex")
    print(" - tables/scalability_table.csv/.tex")
    print(" - figures/*.png and *.pdf")


if __name__ == "__main__":
    main()
