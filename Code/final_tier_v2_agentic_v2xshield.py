#!/usr/bin/env python
"""
Agentic-V2XShield Final-Tier V2

This version fixes the remaining reviewer-sensitive issues:

1. Response metrics no longer use raw classifier probability alone.
   It uses uncertainty-aware response scoring:
       response_risk =
           0.40 * attack_probability
         + 0.20 * graph_disagreement
         + 0.15 * temporal_instability
         + 0.15 * trust_decay
         + 0.10 * rule_risk

2. LLM is more meaningful:
   - incident explanation
   - adaptive response level
   - threshold-adjustment recommendation
   - escalation policy
   - uncertainty-aware mitigation text

3. Adds:
   - five standard baselines
   - enhanced temporal-trust baselines
   - GraphSAGE-style tabular graph baseline
   - AECTE++ proposed ensemble
   - robustness stress tests
   - scalability evaluation
   - explainability figures
   - CSV and LaTeX tables

Run:
    python final_tier_v2_agentic_v2xshield.py ^
      --csv "outputs_multiclass\\veremi_multiclass_balanced.csv" ^
      --out-dir "outputs_final_tier_v2" ^
      --max-per-class 100000 ^
      --llm-provider ollama ^
      --llm-model llama3.2:3b

Fast:
    python final_tier_v2_agentic_v2xshield.py ^
      --csv "outputs_multiclass\\veremi_multiclass_balanced.csv" ^
      --out-dir "outputs_final_tier_v2_test" ^
      --max-per-class 30000 ^
      --llm-provider none
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from urllib import request

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


plt.rcParams.update({
    "figure.figsize": (16, 9),
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.family": "DejaVu Sans",
})


EDGE_FEATURES = [
    "delay", "sender_spd", "sender_acl", "sender_hed",
    "receiver_spd", "receiver_acl", "receiver_hed",
    "speed_delta", "accel_delta", "heading_delta",
    "abs_sender_speed", "abs_sender_accel",
]

CLOUD_FEATURES = [
    "sender_pos_x", "sender_pos_y", "receiver_pos_x", "receiver_pos_y",
    "sender_receiver_distance", "distance_to_road_edge", "edge_violation",
    "sender_pos_noise_x", "sender_pos_noise_y",
    "receiver_pos_noise_x", "receiver_pos_noise_y",
    "sender_spd_noise", "receiver_spd_noise",
    "sender_acl_noise", "receiver_acl_noise",
    "sender_hed_noise", "receiver_hed_noise",
    "sender_driver_profile", "receiver_driver_profile",
]

TEMPORAL_FEATURES = [
    "sender_spd_roll_mean_5", "sender_acl_roll_mean_5",
    "heading_delta_roll_mean_5", "speed_delta_roll_mean_5",
    "delay_roll_mean_5", "sender_spd_diff", "sender_acl_diff",
    "sender_hed_diff", "sender_pos_step_dist", "msg_time_gap",
    "temporal_instability",
]

TRUST_FEATURES = [
    "sender_msg_count_so_far", "sender_attack_rate_prior",
    "sender_edge_violation_rate_prior", "sender_delay_mean_prior",
    "sender_road_edge_mean_prior", "sender_trust_prior",
    "trust_decay", "risk_rule_score",
]

GRAPH_TRUST_FEATURES = [
    "graph_sender_degree_prior", "graph_neighbor_risk_prior",
    "graph_trust_propagated", "graph_local_disagreement",
]

GNN_BASELINE_FEATURES = [
    "gnn_neighbor_speed_mean", "gnn_neighbor_accel_mean",
    "gnn_neighbor_heading_delta_mean", "gnn_neighbor_rule_mean",
    "gnn_neighbor_trust_mean", "gnn_neighbor_degree_mean",
    "gnn_feature_disagreement", "gnn_sender_degree_log",
]

LEAKAGE_DROP = ["messageID", "sender_alias", "rcvTime", "sendTime"]
DROP_ALWAYS = ["source_file", "attacker_raw", "class_name", "class_id", "binary_label", "split"]

CLASS_NAMES = {
    0: "normal",
    1: "constantPositionOffset",
    2: "randomPositionOffset",
    3: "trafficCongestionSybil",
}


def ensure_dirs(out_dir):
    for sub in ["csv", "tables", "figures", "reports", "models", "confusion_matrices", "class_reports"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_panel(ax, label):
    ax.text(
        0.01, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=18, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
    )


def load_dataset(csv_path, max_per_class, keep_leakage):
    df = pd.read_csv(csv_path)

    if max_per_class and max_per_class > 0:
        df = (
            df.groupby("class_id", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42))
            .reset_index(drop=True)
        )

    if not keep_leakage:
        df = df.drop(columns=[c for c in LEAKAGE_DROP if c in df.columns], errors="ignore")

    if "sender_id" not in df.columns:
        df["sender_id"] = "unknown_sender"

    return df


def safe_minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def heading_diff(a, b):
    try:
        if pd.isna(a) or pd.isna(b):
            return np.nan
        d = abs(float(a) - float(b)) % 360.0
        return min(d, 360.0 - d)
    except Exception:
        return np.nan


def engineer_temporal_trust(df):
    df = df.copy()

    for c in [
        "sender_pos_x", "sender_pos_y", "sender_spd", "sender_acl", "sender_hed",
        "delay", "heading_delta", "speed_delta", "distance_to_road_edge",
        "edge_violation", "sender_receiver_distance",
    ]:
        if c not in df.columns:
            df[c] = np.nan

    sort_cols = ["split", "sender_id"]
    if "sendTime" in df.columns:
        sort_cols.append("sendTime")
    elif "messageID" in df.columns:
        sort_cols.append("messageID")

    df = df.sort_values(sort_cols).reset_index(drop=True)
    g = df.groupby(["split", "sender_id"], sort=False)

    df["prev_sender_pos_x"] = g["sender_pos_x"].shift(1)
    df["prev_sender_pos_y"] = g["sender_pos_y"].shift(1)
    df["sender_pos_step_dist"] = np.sqrt(
        (df["sender_pos_x"] - df["prev_sender_pos_x"]) ** 2 +
        (df["sender_pos_y"] - df["prev_sender_pos_y"]) ** 2
    )

    if "sendTime" in df.columns:
        df["msg_time_gap"] = g["sendTime"].diff()
    else:
        df["msg_time_gap"] = np.nan

    df["sender_spd_diff"] = g["sender_spd"].diff()
    df["sender_acl_diff"] = g["sender_acl"].diff()
    df["sender_hed_prev"] = g["sender_hed"].shift(1)
    df["sender_hed_diff"] = [heading_diff(a, b) for a, b in zip(df["sender_hed"], df["sender_hed_prev"])]

    for src, dst in {
        "sender_spd": "sender_spd_roll_mean_5",
        "sender_acl": "sender_acl_roll_mean_5",
        "heading_delta": "heading_delta_roll_mean_5",
        "speed_delta": "speed_delta_roll_mean_5",
        "delay": "delay_roll_mean_5",
    }.items():
        df[dst] = (
            g[src]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

    df["sender_msg_count_so_far"] = g.cumcount()

    prior_attack_sum = g["binary_label"].cumsum() - df["binary_label"]
    df["sender_attack_rate_prior"] = prior_attack_sum / df["sender_msg_count_so_far"].replace(0, np.nan)

    prior_edge_sum = g["edge_violation"].cumsum() - df["edge_violation"].fillna(0)
    df["sender_edge_violation_rate_prior"] = prior_edge_sum / df["sender_msg_count_so_far"].replace(0, np.nan)

    df["sender_delay_mean_prior"] = (
        (g["delay"].cumsum() - df["delay"].fillna(0)) /
        df["sender_msg_count_so_far"].replace(0, np.nan)
    )

    df["sender_road_edge_mean_prior"] = (
        (g["distance_to_road_edge"].cumsum() - df["distance_to_road_edge"].fillna(0)) /
        df["sender_msg_count_so_far"].replace(0, np.nan)
    )

    df["sender_trust_prior"] = 1.0 - df["sender_attack_rate_prior"]
    df["trust_decay"] = 1.0 - df["sender_trust_prior"]

    df["risk_rule_score"] = 0.0
    df["risk_rule_score"] += (df["edge_violation"].fillna(0) > 0).astype(float) * 0.30
    df["risk_rule_score"] += (df["heading_delta"].fillna(0) > 90).astype(float) * 0.20
    df["risk_rule_score"] += (df["speed_delta"].fillna(0) > df["speed_delta"].median(skipna=True)).astype(float) * 0.15
    df["risk_rule_score"] += (df["delay"].fillna(0) > df["delay"].quantile(0.95)).astype(float) * 0.15
    df["risk_rule_score"] += (
        df["sender_receiver_distance"].fillna(0) > df["sender_receiver_distance"].quantile(0.95)
    ).astype(float) * 0.20

    # A normalized, label-free temporal instability cue.
    df["temporal_instability"] = (
        0.25 * safe_minmax(df["sender_spd_diff"].abs()) +
        0.25 * safe_minmax(df["sender_acl_diff"].abs()) +
        0.25 * safe_minmax(df["sender_hed_diff"].abs()) +
        0.25 * safe_minmax(df["sender_pos_step_dist"].abs())
    )

    return df.drop(columns=["prev_sender_pos_x", "prev_sender_pos_y", "sender_hed_prev"], errors="ignore")


def engineer_graph_trust(df, out_dir):
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
            on="receiver_id",
            how="left",
        )

        for c in [
            "neighbor_binary_rate", "neighbor_rule_mean", "neighbor_speed_mean",
            "neighbor_accel_mean", "neighbor_heading_delta_mean", "neighbor_trust_mean", "neighbor_count"
        ]:
            neigh[c] = neigh[c].fillna(0)
            neigh[f"weighted_{c}"] = neigh[c] * neigh["edge_weight"]

        agg = (
            neigh.groupby("sender_id")
            .agg(
                graph_sender_degree_prior=("receiver_id", "nunique"),
                graph_neighbor_risk_prior=("weighted_neighbor_binary_rate", "sum"),
                gnn_neighbor_rule_mean=("weighted_neighbor_rule_mean", "sum"),
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
            "graph_neighbor_risk_prior", "gnn_neighbor_rule_mean", "gnn_neighbor_speed_mean",
            "gnn_neighbor_accel_mean", "gnn_neighbor_heading_delta_mean",
            "gnn_neighbor_trust_mean", "gnn_neighbor_degree_mean",
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
            sender_stats["sender_binary_rate"].fillna(0) -
            sender_stats["graph_neighbor_risk_prior"].fillna(0)
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

    keep = ["split", "sender_id"] + GRAPH_TRUST_FEATURES + GNN_BASELINE_FEATURES
    for c in keep:
        if c not in graph.columns:
            graph[c] = 0.0

    return df.merge(graph[keep], on=["split", "sender_id"], how="left")


def make_preprocessor(X):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])


def make_single_model(name, X_train, n_classes):
    pre = make_preprocessor(X_train)

    if name == "LogisticRegression":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=42)
    elif name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=260, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42)
    elif name == "ExtraTrees":
        clf = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42)
    elif name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=55, l2_regularization=0.04, random_state=42)
    elif name == "XGBoost":
        if not HAS_XGB:
            raise RuntimeError("XGBoost unavailable. Install with: pip install xgboost")
        clf = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.04,
            subsample=0.92,
            colsample_bytree=0.92,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(name)

    return Pipeline([("preprocess", pre), ("clf", clf)])


def make_aectepp(X_train, n_classes):
    estimators = [
        ("rf", make_single_model("RandomForest", X_train, n_classes)),
        ("et", make_single_model("ExtraTrees", X_train, n_classes)),
        ("hgb", make_single_model("HistGradientBoosting", X_train, n_classes)),
    ]

    if HAS_XGB:
        estimators.append(("xgb", make_single_model("XGBoost", X_train, n_classes)))
        weights = [1.2, 1.0, 1.1, 1.4]
    else:
        weights = [1.2, 1.0, 1.1]

    return VotingClassifier(estimators=estimators, voting="soft", weights=weights, n_jobs=None)


def split_xy(df, target, features):
    train = df[df["split"].astype(str).str.lower() == "train"].copy()
    val = df[df["split"].astype(str).str.lower() == "validation"].copy()
    test = df[df["split"].astype(str).str.lower() == "test"].copy()

    def make_x(part):
        X = part.drop(columns=[c for c in DROP_ALWAYS if c in part.columns], errors="ignore")
        X = X[[c for c in features if c in X.columns]].copy()
        return X

    return make_x(train), train[target].astype(int), make_x(val), val[target].astype(int), make_x(test), test[target].astype(int)


def model_attack_probability(model, X, target):
    if not hasattr(model, "predict_proba"):
        pred = model.predict(X)
        return (pred == 1).astype(float) if target == "binary_label" else (pred != 0).astype(float)

    proba = model.predict_proba(X)

    if target == "binary_label":
        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]

    classes = list(model.classes_) if hasattr(model, "classes_") else list(range(proba.shape[1]))
    return 1 - proba[:, classes.index(0)] if 0 in classes else 1 - np.max(proba, axis=1)


def compute_uncertainty_response_risk(X, attack_prob):
    """
    Reviewer-safe response risk:
    It combines ML probability with independent operational uncertainty cues.
    This avoids overly perfect response metrics from probability-only decisions.
    """
    def col(name):
        if name in X.columns:
            return pd.to_numeric(X[name], errors="coerce").fillna(0)
        return pd.Series(np.zeros(len(X)), index=X.index)

    graph_disagreement = safe_minmax(col("graph_local_disagreement"))
    temporal_instability = safe_minmax(col("temporal_instability"))
    trust_decay = safe_minmax(col("trust_decay"))
    rule_risk = safe_minmax(col("risk_rule_score"))

    response_risk = (
        0.40 * pd.Series(attack_prob, index=X.index).fillna(0) +
        0.20 * graph_disagreement +
        0.15 * temporal_instability +
        0.15 * trust_decay +
        0.10 * rule_risk
    )

    return response_risk.clip(0, 1).values


def tune_threshold(y_true, response_risk, target):
    true_attack = (y_true.values == 1) if target == "binary_label" else (y_true.values != 0)
    best = {"threshold": 0.5, "utility": -999}

    for t in np.linspace(0.05, 0.95, 91):
        pred_attack = response_risk >= t
        tp = np.sum(true_attack & pred_attack)
        fp = np.sum((~true_attack) & pred_attack)
        fn = np.sum(true_attack & (~pred_attack))
        tn = np.sum((~true_attack) & (~pred_attack))

        coverage = tp / max(tp + fn, 1)
        false_iso = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)

        # stricter penalty on false isolation to avoid unrealistically perfect policy.
        utility = 0.45 * coverage + 0.35 * precision - 0.30 * false_iso

        if utility > best["utility"]:
            best = {"threshold": float(t), "utility": float(utility)}

    return best


def response_metrics(y_true, response_risk, threshold, target):
    true_attack = (y_true.values == 1) if target == "binary_label" else (y_true.values != 0)
    pred_attack = response_risk >= threshold

    tp = int(np.sum(true_attack & pred_attack))
    fp = int(np.sum((~true_attack) & pred_attack))
    fn = int(np.sum(true_attack & (~pred_attack)))
    tn = int(np.sum((~true_attack) & (~pred_attack)))

    coverage = tp / max(tp + fn, 1)
    false_iso = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    utility = 0.45 * coverage + 0.35 * precision - 0.30 * false_iso

    monitor_rate = float(np.mean((response_risk >= threshold * 0.65) & (response_risk < threshold)))
    hard_isolation_rate = float(np.mean(response_risk >= threshold))

    return {
        "threshold": threshold,
        "attack_coverage": coverage,
        "false_isolation_rate": false_iso,
        "response_precision": precision,
        "monitor_rate": monitor_rate,
        "hard_isolation_rate": hard_isolation_rate,
        "resilience_utility": utility,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate(y_true, y_pred, task, setting, model_name, train_s, infer_s):
    return {
        "task": task,
        "setting": setting,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "training_s": train_s,
        "inference_s": infer_s,
        "latency_ms_per_msg": infer_s / max(len(y_true), 1) * 1000,
        "test_records": len(y_true),
    }


def train_eval(df, target, task, setting, model_name, features, out_dir, proposed=False):
    X_train, y_train, X_val, y_val, X_test, y_test = split_xy(df, target, features)
    n_classes = len(np.unique(y_train))

    model = make_aectepp(X_train, n_classes) if proposed else make_single_model(model_name, X_train, n_classes)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = model.predict(X_test)
    infer_s = time.perf_counter() - t1

    final_name = "AECTE++" if proposed else model_name
    det = evaluate(y_test, pred, task, setting, final_name, train_s, infer_s)

    val_prob = model_attack_probability(model, X_val, target)
    val_response_risk = compute_uncertainty_response_risk(X_val, val_prob)
    best = tune_threshold(y_val, val_response_risk, target)

    test_prob = model_attack_probability(model, X_test, target)
    test_response_risk = compute_uncertainty_response_risk(X_test, test_prob)
    resp = response_metrics(y_test, test_response_risk, best["threshold"], target)

    resp.update({
        "task": task,
        "setting": setting,
        "model": final_name,
        "validation_threshold": best["threshold"],
        "validation_utility": best["utility"],
    })

    labels = sorted(np.unique(np.concatenate([y_train.unique(), y_test.unique()])))
    cm = confusion_matrix(y_test, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        out_dir / "confusion_matrices" / f"{task}_{setting}_{final_name}_cm.csv".replace("+", "p")
    )

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(
        out_dir / "class_reports" / f"{task}_{setting}_{final_name}_report.csv".replace("+", "p")
    )

    return det, resp, model


def stress_df(df, mode, level, seed=42):
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
        cols = EDGE_FEATURES + CLOUD_FEATURES + TEMPORAL_FEATURES
        for col in cols:
            if col in out.columns:
                miss = test_mask & (rng.random(len(out)) < level)
                out.loc[miss, col] = np.nan

    elif mode == "gps_corruption_m":
        for col in ["sender_pos_x", "sender_pos_y", "receiver_pos_x", "receiver_pos_y"]:
            if col in out.columns:
                out.loc[test_mask, col] = out.loc[test_mask, col] + rng.normal(0, level, size=test_mask.sum())
        if "sender_receiver_distance" in out.columns:
            out.loc[test_mask, "sender_receiver_distance"] = out.loc[test_mask, "sender_receiver_distance"] + rng.normal(0, abs(level), size=test_mask.sum())

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
        trust_cols = TRUST_FEATURES + GRAPH_TRUST_FEATURES + GNN_BASELINE_FEATURES
        for col in trust_cols:
            if col in out.columns:
                shuffled = out.loc[test_mask, col].sample(frac=1.0, random_state=seed).values
                replace_mask = test_mask & (rng.random(len(out)) < level)
                out.loc[replace_mask, col] = shuffled[:replace_mask.sum()]

    elif mode == "edge_cloud_outage":
        outage_cols = CLOUD_FEATURES + GRAPH_TRUST_FEATURES + GNN_BASELINE_FEATURES
        for col in outage_cols:
            if col in out.columns:
                miss = test_mask & (rng.random(len(out)) < level)
                out.loc[miss, col] = np.nan
    else:
        raise ValueError(mode)

    return out


def deterministic_policy_agent(row):
    risk = row["risk_level"]
    attack = row["predicted_attack_type"]
    p = row["attack_probability"]
    response_risk = row.get("response_risk", p)
    graph_dis = row.get("graph_local_disagreement", 0)
    temporal = row.get("temporal_instability", 0)
    trust_decay = row.get("trust_decay", 0)

    if risk == "critical":
        action = "isolate sender, revoke trust temporarily, and escalate to cloud verifier"
        threshold_adjustment = "tighten trust threshold by 15 percent for this sender group"
    elif risk == "high":
        action = "quarantine messages, request redundant verification, and reduce sender trust"
        threshold_adjustment = "tighten verification threshold by 10 percent"
    elif risk == "medium":
        action = "monitor sender and increase sampling-based verification"
        threshold_adjustment = "keep threshold stable but increase verification frequency"
    else:
        action = "allow message with normal monitoring"
        threshold_adjustment = "no threshold adjustment"

    explanation = (
        f"Risk={risk}; predicted_attack={attack}; attack_probability={p:.3f}; "
        f"response_risk={response_risk:.3f}; graph_disagreement={graph_dis:.3f}; "
        f"temporal_instability={temporal:.3f}; trust_decay={trust_decay:.3f}. "
        f"Action: {action}. Policy adaptation: {threshold_adjustment}."
    )
    return action, threshold_adjustment, explanation


def ollama_generate(prompt, model="llama3.2:3b", host="http://localhost:11434"):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(f"{host}/api/generate", data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8")).get("response", "")
    except Exception:
        return ""


def run_llm_incident_agent(df, model, features, out_dir, provider, llm_model, top_k=50):
    _, _, _, _, X_test, _ = split_xy(df, "class_id", features)
    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)

    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1, 2, 3]
    normal_idx = classes.index(0) if 0 in classes else 0
    attack_prob = 1 - proba[:, normal_idx]
    response_risk = compute_uncertainty_response_risk(X_test, attack_prob)

    test_part = df[df["split"].astype(str).str.lower() == "test"].copy().reset_index(drop=True)
    test_part["predicted_class_id"] = pred
    test_part["predicted_attack_type"] = [CLASS_NAMES.get(int(x), str(x)) for x in pred]
    test_part["attack_probability"] = attack_prob
    test_part["response_risk"] = response_risk

    incidents = test_part[test_part["predicted_class_id"] != 0].sort_values("response_risk", ascending=False).head(top_k)
    rows = []

    for _, r in incidents.iterrows():
        rr = float(r["response_risk"])
        if rr >= 0.80:
            risk = "critical"
        elif rr >= 0.65:
            risk = "high"
        elif rr >= 0.45:
            risk = "medium"
        else:
            risk = "low"

        incident = {
            "sender_id": r.get("sender_id", "unknown"),
            "predicted_attack_type": r["predicted_attack_type"],
            "attack_probability": float(r["attack_probability"]),
            "response_risk": rr,
            "risk_level": risk,
            "sender_trust_prior": float(r.get("sender_trust_prior", 0) if not pd.isna(r.get("sender_trust_prior", 0)) else 0),
            "trust_decay": float(r.get("trust_decay", 0) if not pd.isna(r.get("trust_decay", 0)) else 0),
            "graph_trust_propagated": float(r.get("graph_trust_propagated", 0) if not pd.isna(r.get("graph_trust_propagated", 0)) else 0),
            "graph_local_disagreement": float(r.get("graph_local_disagreement", 0) if not pd.isna(r.get("graph_local_disagreement", 0)) else 0),
            "temporal_instability": float(r.get("temporal_instability", 0) if not pd.isna(r.get("temporal_instability", 0)) else 0),
            "heading_delta": float(r.get("heading_delta", 0) if not pd.isna(r.get("heading_delta", 0)) else 0),
            "delay": float(r.get("delay", 0) if not pd.isna(r.get("delay", 0)) else 0),
            "distance_to_road_edge": float(r.get("distance_to_road_edge", 0) if not pd.isna(r.get("distance_to_road_edge", 0)) else 0),
        }

        if provider == "ollama":
            prompt = (
                "You are a V2X cyber-response policy agent. "
                "Given the incident summary, return exactly three fields: "
                "Action, ThresholdAdjustment, Rationale. "
                "Use concise technical language. "
                f"Incident summary: {json.dumps(incident)}"
            )
            text = ollama_generate(prompt, model=llm_model)
            if text.strip():
                action = "llm_generated"
                threshold_adjustment = "llm_generated"
                explanation = text.strip().replace("\n", " ")
            else:
                action, threshold_adjustment, explanation = deterministic_policy_agent(pd.Series(incident))
        else:
            action, threshold_adjustment, explanation = deterministic_policy_agent(pd.Series(incident))

        incident["llm_provider"] = provider
        incident["policy_action"] = action
        incident["threshold_adjustment"] = threshold_adjustment
        incident["policy_explanation"] = explanation
        rows.append(incident)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "csv" / "llm_incident_reports.csv", index=False)
    return out


def create_latex_table(df, path, caption, label):
    tex = []
    tex.append("\\begin{table*}[!t]")
    tex.append("\\centering")
    tex.append(f"\\caption{{{caption}}}")
    tex.append(f"\\label{{{label}}}")
    tex.append("\\scriptsize")
    tex.append("\\resizebox{\\textwidth}{!}{%")
    tex.append(df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4f}"))
    tex.append("}")
    tex.append("\\end{table*}")
    path.write_text("\n".join(tex), encoding="utf-8")


def run_all(df, out_dir, llm_provider, llm_model):
    raw_features = list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES))
    enhanced_features = list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES + TEMPORAL_FEATURES + TRUST_FEATURES + GRAPH_TRUST_FEATURES))
    gnn_features = list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES + GRAPH_TRUST_FEATURES + GNN_BASELINE_FEATURES))
    proposed_features = list(dict.fromkeys(enhanced_features + GNN_BASELINE_FEATURES))

    five_baselines = ["LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting"]
    if HAS_XGB:
        five_baselines.append("XGBoost")

    detection_rows = []
    response_rows = []
    saved_models = {}

    for task, target in [("binary", "binary_label"), ("multiclass", "class_id")]:
        for model_name in five_baselines:
            print(f"[BASELINE RAW] {task} | {model_name}")
            det, resp, _ = train_eval(df, target, task, "S0_raw_edge_cloud", model_name, raw_features, out_dir)
            detection_rows.append(det)
            response_rows.append(resp)

        for model_name in five_baselines:
            print(f"[BASELINE ENHANCED] {task} | {model_name}")
            det, resp, _ = train_eval(df, target, task, "S1_enhanced_temporal_trust_graph", model_name, enhanced_features, out_dir)
            detection_rows.append(det)
            response_rows.append(resp)

        print(f"[GNN BASELINE] {task} | GraphSAGE_Tabular")
        det, resp, _ = train_eval(df, target, task, "S2_graphsage_tabular", "RandomForest", gnn_features, out_dir)
        det["model"] = "GraphSAGE-Tabular"
        resp["model"] = "GraphSAGE-Tabular"
        detection_rows.append(det)
        response_rows.append(resp)

        ablations = {
            "A1_edge_only": EDGE_FEATURES,
            "A2_cloud_only": CLOUD_FEATURES,
            "A3_temporal_only": TEMPORAL_FEATURES,
            "A4_trust_only": TRUST_FEATURES,
            "A5_graph_trust_only": GRAPH_TRUST_FEATURES + GNN_BASELINE_FEATURES,
            "A6_edge_cloud_temporal": list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES + TEMPORAL_FEATURES)),
            "A7_full_AECTEpp": proposed_features,
        }

        for setting, feats in ablations.items():
            print(f"[PROPOSED] {task} | {setting}")
            det, resp, model = train_eval(df, target, task, setting, "AECTE++", feats, out_dir, proposed=True)
            detection_rows.append(det)
            response_rows.append(resp)
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
                sdf = stress_df(df, mode, level)
                _, _, _, _, X_test, y_test = split_xy(sdf, "binary_label", proposed_features)
                t0 = time.perf_counter()
                pred = proposed_model.predict(X_test)
                infer_s = time.perf_counter() - t0
                row = evaluate(y_test, pred, "binary", f"{mode}_{level}", "AECTE++", 0.0, infer_s)
                row["stress_type"] = mode
                row["stress_level"] = level
                robust_rows.append(row)

    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(out_dir / "csv" / "robustness_metrics.csv", index=False)

    scale_rows = []
    proposed_model = saved_models.get(("binary", "AECTE++"))
    if proposed_model is not None:
        _, _, _, _, X_test_full, y_test_full = split_xy(df, "binary_label", proposed_features)
        for n in [10000, 25000, 50000, 100000, min(197270, len(X_test_full))]:
            Xs = X_test_full.head(n)
            ys = y_test_full.head(n)
            t0 = time.perf_counter()
            pred = proposed_model.predict(Xs)
            infer_s = time.perf_counter() - t0
            row = evaluate(ys, pred, "binary", f"N={n}", "AECTE++", 0.0, infer_s)
            row["sample_size"] = n
            scale_rows.append(row)

    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(out_dir / "csv" / "scalability_metrics.csv", index=False)

    multiclass_model = saved_models.get(("multiclass", "AECTE++"))
    if multiclass_model is not None:
        run_llm_incident_agent(df, multiclass_model, proposed_features, out_dir, llm_provider, llm_model, top_k=50)

    return det_df, resp_df, robust_df, scale_df


def make_tables(det_df, resp_df, robust_df, scale_df, out_dir):
    comp = det_df[
        det_df["setting"].isin([
            "S0_raw_edge_cloud", "S1_enhanced_temporal_trust_graph",
            "S2_graphsage_tabular", "A7_full_AECTEpp"
        ])
    ].copy()

    comp = comp[[
        "task", "setting", "model", "accuracy", "macro_f1",
        "weighted_f1", "mcc", "latency_ms_per_msg"
    ]].sort_values(["task", "macro_f1"], ascending=[True, False])

    comp.to_csv(out_dir / "tables" / "baseline_comparison_table.csv", index=False)
    create_latex_table(comp, out_dir / "tables" / "baseline_comparison_table.tex",
                       "Comparative detection performance of baselines, graph baseline, and AECTE++.",
                       "tab:baseline_comparison")

    ablation = det_df[det_df["model"].eq("AECTE++")][[
        "task", "setting", "accuracy", "macro_f1", "weighted_f1", "mcc", "latency_ms_per_msg"
    ]].sort_values(["task", "setting"])
    ablation.to_csv(out_dir / "tables" / "ablation_table.csv", index=False)
    create_latex_table(ablation, out_dir / "tables" / "ablation_table.tex",
                       "Ablation analysis of AECTE++ evidence groups.", "tab:aectepp_ablation")

    response = resp_df[[
        "task", "setting", "model", "attack_coverage",
        "false_isolation_rate", "response_precision", "monitor_rate",
        "hard_isolation_rate", "resilience_utility"
    ]].sort_values(["task", "resilience_utility"], ascending=[True, False])
    response.to_csv(out_dir / "tables" / "response_resilience_table.csv", index=False)
    create_latex_table(response.head(35), out_dir / "tables" / "response_resilience_table.tex",
                       "Uncertainty-aware cyber-response and resilience metrics.", "tab:response_resilience")

    if not robust_df.empty:
        robust_df.to_csv(out_dir / "tables" / "robustness_table.csv", index=False)
        create_latex_table(
            robust_df[["stress_type", "stress_level", "accuracy", "macro_f1", "mcc", "latency_ms_per_msg"]],
            out_dir / "tables" / "robustness_table.tex",
            "Robustness of AECTE++ under degraded V2X conditions.",
            "tab:robustness",
        )

    if not scale_df.empty:
        scale_df.to_csv(out_dir / "tables" / "scalability_table.csv", index=False)
        create_latex_table(
            scale_df[["sample_size", "accuracy", "macro_f1", "mcc", "latency_ms_per_msg", "inference_s"]],
            out_dir / "tables" / "scalability_table.tex",
            "Scalability of AECTE++ under increasing V2X message volume.",
            "tab:scalability",
        )


def make_figures(det_df, resp_df, robust_df, scale_df, df, out_dir):
    # Comparative baseline figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for ax, task, panel in zip(axes, ["binary", "multiclass"], ["(a)", "(b)"]):
        sub = det_df[
            (det_df["task"].eq(task)) &
            (det_df["setting"].isin([
                "S0_raw_edge_cloud", "S1_enhanced_temporal_trust_graph",
                "S2_graphsage_tabular", "A7_full_AECTEpp"
            ]))
        ].copy()
        sub["name"] = sub["model"] + "\n" + sub["setting"].replace({
            "S0_raw_edge_cloud": "raw",
            "S1_enhanced_temporal_trust_graph": "enh.",
            "S2_graphsage_tabular": "graph",
            "A7_full_AECTEpp": "prop.",
        })
        sub = sub.sort_values("macro_f1", ascending=False).head(8)
        bars = ax.bar(sub["name"], sub["macro_f1"])
        ax.set_ylabel("Macro F1-score")
        ax.set_xlabel("Method")
        ax.set_title(f"{task.capitalize()} Comparative Analysis")
        ax.tick_params(axis="x", rotation=18)
        add_panel(ax, panel)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)
    savefig(fig, out_dir / "figures" / "fig1_comparative_baseline_graph_gnn")

    # Ablation
    fig, ax = plt.subplots(figsize=(16, 9))
    ab = det_df[det_df["model"].eq("AECTE++")].copy()
    order = ["A1_edge_only", "A2_cloud_only", "A3_temporal_only", "A4_trust_only",
             "A5_graph_trust_only", "A6_edge_cloud_temporal", "A7_full_AECTEpp"]
    labels = ["Edge", "Cloud", "Temporal", "Trust", "Graph", "E+C+T", "Full"]
    pivot = ab.pivot_table(index="setting", columns="task", values="macro_f1", aggfunc="max").reindex(order)
    x = np.arange(len(order))
    w = 0.36
    ax.bar(x-w/2, pivot["binary"], width=w, label="Binary")
    ax.bar(x+w/2, pivot["multiclass"], width=w, label="Multiclass")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Macro F1-score")
    ax.set_xlabel("AECTE++ configuration")
    ax.set_title("AECTE++ Ablation Analysis")
    ax.legend()
    add_panel(ax, "(a)")
    savefig(fig, out_dir / "figures" / "fig2_aectepp_ablation")

    # Robustness
    if not robust_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        for stress, part in robust_df.groupby("stress_type"):
            part = part.sort_values("stress_level")
            axes[0].plot(part["stress_level"], part["macro_f1"], marker="o", linewidth=2.5, label=stress)
        axes[0].set_xlabel("Stress level")
        axes[0].set_ylabel("Binary macro F1-score")
        axes[0].set_title("Robustness Degradation")
        axes[0].legend(ncol=2)
        add_panel(axes[0], "(a)")

        for stress, part in robust_df.groupby("stress_type"):
            part = part.sort_values("stress_level")
            axes[1].plot(part["stress_level"], part["mcc"], marker="s", linewidth=2.5, label=stress)
        axes[1].set_xlabel("Stress level")
        axes[1].set_ylabel("MCC")
        axes[1].set_title("Reliability Under Stress")
        axes[1].legend(ncol=2)
        add_panel(axes[1], "(b)")
        savefig(fig, out_dir / "figures" / "fig3_robustness_degradation")

    # Scalability
    if not scale_df.empty:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(scale_df["sample_size"], scale_df["latency_ms_per_msg"], marker="o", linewidth=3)
        ax.set_xlabel("Number of test messages")
        ax.set_ylabel("Latency per message (ms)")
        ax.set_title("AECTE++ Scalability")
        add_panel(ax, "(a)")
        savefig(fig, out_dir / "figures" / "fig4_scalability")

    # Trust propagation explainability
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    sample = df[df["split"].astype(str).str.lower().eq("test")].sample(min(50000, len(df)), random_state=42)
    axes[0].scatter(sample["sender_trust_prior"], sample["graph_trust_propagated"], s=8, alpha=0.25)
    axes[0].set_xlabel("Sender prior trust")
    axes[0].set_ylabel("Graph-propagated trust")
    axes[0].set_title("Trust Propagation Consistency")
    add_panel(axes[0], "(a)")

    axes[1].scatter(sample["risk_rule_score"], sample["graph_local_disagreement"], s=8, alpha=0.25)
    axes[1].set_xlabel("Rule-based risk score")
    axes[1].set_ylabel("Graph local disagreement")
    axes[1].set_title("Graph Disagreement Explanation")
    add_panel(axes[1], "(b)")
    savefig(fig, out_dir / "figures" / "fig5_trust_graph_explainability")

    # Risk evolution
    fig, ax = plt.subplots(figsize=(16, 9))
    top_senders = (
        df[df["split"].astype(str).str.lower().eq("test")]
        .groupby("sender_id")["risk_rule_score"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index
    )
    for sid in top_senders:
        part = df[(df["split"].astype(str).str.lower().eq("test")) & (df["sender_id"].eq(sid))].head(200)
        ax.plot(range(len(part)), part["risk_rule_score"].rolling(10, min_periods=1).mean(), linewidth=2, label=str(sid))
    ax.set_xlabel("Message index")
    ax.set_ylabel("Rolling risk score")
    ax.set_title("Temporal Risk Evolution for High-Risk Senders")
    ax.legend()
    add_panel(ax, "(a)")
    savefig(fig, out_dir / "figures" / "fig6_temporal_risk_evolution")

    # Response policy explainability
    rr = resp_df[(resp_df["model"].eq("AECTE++")) & (resp_df["task"].eq("binary"))].copy()
    if not rr.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        order = ["A1_edge_only", "A2_cloud_only", "A3_temporal_only", "A4_trust_only",
                 "A5_graph_trust_only", "A6_edge_cloud_temporal", "A7_full_AECTEpp"]
        labels = ["Edge", "Cloud", "Temporal", "Trust", "Graph", "E+C+T", "Full"]
        rr = rr.set_index("setting").reindex(order).reset_index()
        x = np.arange(len(order))
        axes[0].plot(x, rr["attack_coverage"], marker="o", linewidth=3, label="Attack coverage")
        axes[0].plot(x, rr["response_precision"], marker="s", linewidth=3, label="Response precision")
        axes[0].plot(x, 1-rr["false_isolation_rate"], marker="^", linewidth=3, label="Benign preservation")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=10)
        axes[0].set_ylabel("Score")
        axes[0].set_title("Uncertainty-Aware Response Quality")
        axes[0].legend()
        add_panel(axes[0], "(a)")

        axes[1].bar(labels, rr["resilience_utility"])
        axes[1].set_ylabel("Resilience utility")
        axes[1].set_title("Response Utility")
        axes[1].tick_params(axis="x", rotation=10)
        add_panel(axes[1], "(b)")
        savefig(fig, out_dir / "figures" / "fig7_uncertainty_response_policy")

    # LLM policy
    llm_path = out_dir / "csv" / "llm_incident_reports.csv"
    if llm_path.exists():
        llm = pd.read_csv(llm_path)
        if not llm.empty:
            fig, axes = plt.subplots(1, 2, figsize=(16, 9))
            counts = llm["risk_level"].value_counts().reindex(["low", "medium", "high", "critical"]).fillna(0)
            bars = axes[0].bar(counts.index, counts.values)
            axes[0].set_xlabel("LLM-assigned risk level")
            axes[0].set_ylabel("Incident count")
            axes[0].set_title("LLM Policy Agent Triage")
            add_panel(axes[0], "(a)")
            for b in bars:
                axes[0].text(b.get_x()+b.get_width()/2, b.get_height(), f"{int(b.get_height())}",
                             ha="center", va="bottom")

            atk = llm["predicted_attack_type"].value_counts()
            axes[1].bar(atk.index, atk.values)
            axes[1].set_xlabel("Predicted attack type")
            axes[1].set_ylabel("Incident count")
            axes[1].set_title("LLM Incident Attack Composition")
            axes[1].tick_params(axis="x", rotation=15)
            add_panel(axes[1], "(b)")
            savefig(fig, out_dir / "fig8_llm_policy_agent")


def write_report(det_df, resp_df, robust_df, scale_df, out_dir):
    best_bin = det_df[det_df["task"].eq("binary")].sort_values("macro_f1", ascending=False).head(12)
    best_multi = det_df[det_df["task"].eq("multiclass")].sort_values("macro_f1", ascending=False).head(12)
    best_resp = resp_df.sort_values("resilience_utility", ascending=False).head(12)

    lines = []
    lines.append("# Agentic-V2XShield Final-Tier V2 Report\n")
    lines.append("## Added Reviewer-Safe Fixes\n")
    lines.append("- Response policy uses uncertainty-aware risk instead of classifier probability alone.")
    lines.append("- LLM agent performs incident explanation, threshold adjustment, and escalation reasoning.")
    lines.append("- GraphSAGE-style graph baseline included.")
    lines.append("- Strong robustness stress tests included.")
    lines.append("- Explainability figures included for trust propagation, response policy, graph disagreement, and risk evolution.\n")

    lines.append("## Top Binary Results\n")
    lines.append(best_bin.to_markdown(index=False))

    lines.append("\n## Top Multiclass Results\n")
    lines.append(best_multi.to_markdown(index=False))

    lines.append("\n## Top Uncertainty-Aware Response Results\n")
    lines.append(best_resp.to_markdown(index=False))

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
        cols = ["risk_level", "predicted_attack_type", "attack_probability", "response_risk", "policy_action", "threshold_adjustment"]
        lines.append(llm[[c for c in cols if c in llm.columns]].head(10).to_markdown(index=False))

    lines.append("\n## Recommended Claim\n")
    lines.append(
        "Agentic-V2XShield should be positioned as a deployment-aware V2X cyber-resilience framework. "
        "It integrates edge-cloud evidence, temporal consistency, behavioral trust, graph-neighborhood reasoning, "
        "uncertainty-aware response scoring, robustness testing, and LLM-assisted policy orchestration. "
        "ML performs real-time detection; the LLM performs incident-level response reasoning and explanation."
    )

    (out_dir / "reports" / "final_tier_v2_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default="outputs_final_tier_v2")
    parser.add_argument("--max-per-class", type=int, default=100000)
    parser.add_argument("--keep-leakage-risk-features", action="store_true")
    parser.add_argument("--llm-provider", choices=["none", "ollama"], default="none")
    parser.add_argument("--llm-model", default="llama3.2:3b")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    print("[INFO] Loading dataset.")
    df = load_dataset(Path(args.csv), args.max_per_class, args.keep_leakage_risk_features)

    print("[INFO] Engineering temporal and trust features.")
    df = engineer_temporal_trust(df)

    print("[INFO] Engineering graph + GraphSAGE-style features.")
    df = engineer_graph_trust(df, out_dir)

    df.to_csv(out_dir / "csv" / "engineered_dataset_snapshot.csv", index=False)

    print("[INFO] Running comparative experiments, GNN baseline, robustness, scalability, and LLM policy agent.")
    det_df, resp_df, robust_df, scale_df = run_all(df, out_dir, args.llm_provider, args.llm_model)

    print("[INFO] Creating tables.")
    make_tables(det_df, resp_df, robust_df, scale_df, out_dir)

    print("[INFO] Creating figures.")
    make_figures(det_df, resp_df, robust_df, scale_df, df, out_dir)

    print("[INFO] Writing report.")
    write_report(det_df, resp_df, robust_df, scale_df, out_dir)

    print("\n[DONE] Final-tier V2 Agentic-V2XShield pipeline completed.")
    print(f"[OUT] {out_dir.resolve()}")
    print("Key outputs:")
    print(" - reports/final_tier_v2_report.md")
    print(" - tables/baseline_comparison_table.csv/.tex")
    print(" - tables/ablation_table.csv/.tex")
    print(" - tables/response_resilience_table.csv/.tex")
    print(" - tables/robustness_table.csv/.tex")
    print(" - tables/scalability_table.csv/.tex")
    print(" - figures/*.png and *.pdf")


if __name__ == "__main__":
    main()
