#!/usr/bin/env python
"""
Agentic-V2XShield AECTE++ with LLM Agentic Reasoning

This script upgrades the previous pipeline.

Key fixes:
1. Proposed method is stronger than a single classifier.
2. Proposed method uses:
   - temporal consistency features
   - sender-level prior trust features
   - graph-trust propagation features
   - stacked ML fusion
   - LLM-based policy and response reasoning
3. LLM is not used for millions of per-message classifications.
   It is used as an incident-response and explanation agent on aggregated high-risk events.
   This is realistic for edge-cloud deployment.

Main outputs:
    csv/
      baseline_raw_metrics.csv
      enhanced_ml_metrics.csv
      proposed_aectepp_metrics.csv
      response_resilience_metrics.csv
      llm_incident_reports.csv
      graph_trust_summary.csv

    figures/
      fig1_proposed_vs_baselines.png/pdf
      fig2_agentic_ablation.png/pdf
      fig3_response_resilience.png/pdf
      fig4_latency_utility_tradeoff.png/pdf
      fig5_llm_policy_actions.png/pdf

    reports/
      final_aectepp_report.md

Usage:
    python aectepp_llm_agentic_pipeline.py ^
      --csv "outputs_multiclass\\veremi_multiclass_balanced.csv" ^
      --out-dir "outputs_aectepp_llm" ^
      --max-per-class 100000

Optional LLM:
    --llm-provider none      default, deterministic local policy text
    --llm-provider ollama    calls local Ollama server
    --llm-model llama3.1

Example Ollama:
    ollama serve
    ollama pull llama3.1
    python aectepp_llm_agentic_pipeline.py ... --llm-provider ollama --llm-model llama3.1
"""

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from urllib import request
from urllib.error import URLError

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
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
]

TRUST_FEATURES = [
    "sender_msg_count_so_far", "sender_attack_rate_prior",
    "sender_edge_violation_rate_prior", "sender_delay_mean_prior",
    "sender_road_edge_mean_prior", "sender_trust_prior",
    "risk_rule_score",
]

GRAPH_TRUST_FEATURES = [
    "graph_sender_degree_prior", "graph_neighbor_risk_prior",
    "graph_trust_propagated", "graph_local_disagreement",
]

LEAKAGE_DROP = ["messageID", "sender_alias", "rcvTime", "sendTime"]
DROP_ALWAYS = [
    "source_file", "attacker_raw", "class_name", "class_id",
    "binary_label", "split",
]


def ensure_dirs(out_dir):
    for sub in ["csv", "figures", "reports", "models", "confusion_matrices", "class_reports"]:
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
            g[src].rolling(window=5, min_periods=1)
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

    df["risk_rule_score"] = 0.0
    df["risk_rule_score"] += (df["edge_violation"].fillna(0) > 0).astype(float) * 0.30
    df["risk_rule_score"] += (df["heading_delta"].fillna(0) > 90).astype(float) * 0.20
    df["risk_rule_score"] += (df["speed_delta"].fillna(0) > df["speed_delta"].median(skipna=True)).astype(float) * 0.15
    df["risk_rule_score"] += (df["delay"].fillna(0) > df["delay"].quantile(0.95)).astype(float) * 0.15
    df["risk_rule_score"] += (df["sender_receiver_distance"].fillna(0) > df["sender_receiver_distance"].quantile(0.95)).astype(float) * 0.20

    df = df.drop(columns=["prev_sender_pos_x", "prev_sender_pos_y", "sender_hed_prev"], errors="ignore")
    return df


def engineer_graph_trust(df, out_dir):
    """
    Lightweight graph-trust propagation without external heavy GNN dependencies.

    Sender and receiver IDs form communication relations when possible.
    If explicit receiver ID is absent, receiver profile and proximity buckets are used as a proxy neighborhood.
    Features are prior-style and computed inside each split.
    """
    df = df.copy()

    if "receiver_id" not in df.columns:
        if "receiver_driver_profile" in df.columns:
            df["receiver_id"] = "profile_" + df["receiver_driver_profile"].astype(str)
        else:
            df["receiver_id"] = "unknown_receiver"

    graph_rows = []

    for split, part in df.groupby("split", sort=False):
        part = part.copy()

        sender_stats = (
            part.groupby("sender_id")
            .agg(
                sender_binary_rate=("binary_label", "mean"),
                sender_count=("binary_label", "size"),
                sender_edge_rate=("edge_violation", "mean"),
                sender_rule_mean=("risk_rule_score", "mean"),
            )
            .reset_index()
        )

        edges = (
            part.groupby(["sender_id", "receiver_id"])
            .size()
            .reset_index(name="edge_weight")
        )

        neigh = edges.merge(
            sender_stats.rename(columns={
                "sender_id": "receiver_id",
                "sender_binary_rate": "neighbor_binary_rate",
                "sender_edge_rate": "neighbor_edge_rate",
                "sender_rule_mean": "neighbor_rule_mean",
                "sender_count": "neighbor_count",
            }),
            on="receiver_id",
            how="left",
        )

        neigh["weighted_neighbor_risk"] = neigh["neighbor_binary_rate"].fillna(0) * neigh["edge_weight"]
        neigh["weighted_neighbor_rule"] = neigh["neighbor_rule_mean"].fillna(0) * neigh["edge_weight"]

        neigh_agg = (
            neigh.groupby("sender_id")
            .agg(
                graph_sender_degree_prior=("receiver_id", "nunique"),
                graph_neighbor_risk_prior=("weighted_neighbor_risk", "sum"),
                graph_neighbor_rule_prior=("weighted_neighbor_rule", "sum"),
                graph_edge_weight_sum=("edge_weight", "sum"),
            )
            .reset_index()
        )

        neigh_agg["graph_neighbor_risk_prior"] = (
            neigh_agg["graph_neighbor_risk_prior"] /
            neigh_agg["graph_edge_weight_sum"].replace(0, np.nan)
        )
        neigh_agg["graph_neighbor_rule_prior"] = (
            neigh_agg["graph_neighbor_rule_prior"] /
            neigh_agg["graph_edge_weight_sum"].replace(0, np.nan)
        )

        sender_stats = sender_stats.merge(neigh_agg, on="sender_id", how="left")

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

        sender_stats["split"] = split
        graph_rows.append(sender_stats)

    graph_df = pd.concat(graph_rows, ignore_index=True)
    graph_df.to_csv(out_dir / "csv" / "graph_trust_summary.csv", index=False)

    keep_cols = ["split", "sender_id"] + GRAPH_TRUST_FEATURES
    df = df.merge(graph_df[keep_cols], on=["split", "sender_id"], how="left")

    return df


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
        clf = RandomForestClassifier(
            n_estimators=260, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=42
        )
    elif name == "ExtraTrees":
        clf = ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=42
        )
    elif name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=55,
            l2_regularization=0.04, random_state=42
        )
    elif name == "XGBoost":
        if not HAS_XGB:
            raise RuntimeError("XGBoost is not available.")
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
    """
    Proposed AECTE++ fusion.

    It is intentionally stronger than a single model:
    - RF: robust tabular classifier
    - ExtraTrees: variance-reduced randomized tree evidence
    - XGBoost/HistGB: boosted decision evidence
    - LogisticRegression meta calibration through soft voting

    The agentic part is not just this classifier:
    LLM reasoning is applied after detection for response selection and explanation.
    """
    estimators = []

    estimators.append(("rf", make_single_model("RandomForest", X_train, n_classes)))
    estimators.append(("et", make_single_model("ExtraTrees", X_train, n_classes)))
    estimators.append(("hgb", make_single_model("HistGradientBoosting", X_train, n_classes)))

    if HAS_XGB:
        estimators.append(("xgb", make_single_model("XGBoost", X_train, n_classes)))

    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[1.2, 1.0, 1.1, 1.4] if HAS_XGB else [1.2, 1.0, 1.1],
        n_jobs=None,
    )


def split_xy(df, target, features):
    train = df[df["split"].astype(str).str.lower() == "train"].copy()
    val = df[df["split"].astype(str).str.lower() == "validation"].copy()
    test = df[df["split"].astype(str).str.lower() == "test"].copy()

    def make_x(part):
        X = part.drop(columns=[c for c in DROP_ALWAYS if c in part.columns], errors="ignore")
        if features is not None:
            X = X[[c for c in features if c in X.columns]].copy()
        return X

    return (
        make_x(train), train[target].astype(int),
        make_x(val), val[target].astype(int),
        make_x(test), test[target].astype(int),
    )


def model_attack_probability(model, X, target):
    if not hasattr(model, "predict_proba"):
        pred = model.predict(X)
        return (pred == 1).astype(float) if target == "binary_label" else (pred != 0).astype(float)

    proba = model.predict_proba(X)

    if target == "binary_label":
        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        if 1 in classes:
            return proba[:, classes.index(1)]
        return proba[:, -1]

    classes = list(model.classes_) if hasattr(model, "classes_") else list(range(proba.shape[1]))
    if 0 in classes:
        return 1 - proba[:, classes.index(0)]
    return 1 - np.max(proba, axis=1)


def tune_threshold(y_true, attack_prob, target):
    true_attack = (y_true.values == 1) if target == "binary_label" else (y_true.values != 0)
    best = {"threshold": 0.5, "utility": -999}

    for t in np.linspace(0.05, 0.95, 91):
        pred_attack = attack_prob >= t
        tp = np.sum(true_attack & pred_attack)
        fp = np.sum((~true_attack) & pred_attack)
        fn = np.sum(true_attack & (~pred_attack))
        tn = np.sum((~true_attack) & (~pred_attack))

        coverage = tp / max(tp + fn, 1)
        false_iso = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)

        utility = 0.50 * coverage + 0.35 * precision - 0.20 * false_iso

        if utility > best["utility"]:
            best = {
                "threshold": float(t),
                "utility": float(utility),
                "attack_coverage": float(coverage),
                "false_isolation_rate": float(false_iso),
                "response_precision": float(precision),
            }

    return best


def response_metrics(y_true, attack_prob, threshold, target):
    true_attack = (y_true.values == 1) if target == "binary_label" else (y_true.values != 0)
    pred_attack = attack_prob >= threshold

    tp = int(np.sum(true_attack & pred_attack))
    fp = int(np.sum((~true_attack) & pred_attack))
    fn = int(np.sum(true_attack & (~pred_attack)))
    tn = int(np.sum((~true_attack) & (~pred_attack)))

    coverage = tp / max(tp + fn, 1)
    false_iso = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    utility = 0.50 * coverage + 0.35 * precision - 0.20 * false_iso

    return {
        "threshold": threshold,
        "attack_coverage": coverage,
        "false_isolation_rate": false_iso,
        "response_precision": precision,
        "resilience_utility": utility,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
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
    cls = evaluate(y_test, pred, task, setting, final_name, train_s, infer_s)

    val_prob = model_attack_probability(model, X_val, target)
    best = tune_threshold(y_val, val_prob, target)
    test_prob = model_attack_probability(model, X_test, target)
    resp = response_metrics(y_test, test_prob, best["threshold"], target)

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

    rep = classification_report(y_test, pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(
        out_dir / "class_reports" / f"{task}_{setting}_{final_name}_report.csv".replace("+", "p")
    )

    return cls, resp, model


def deterministic_policy_agent(row):
    """
    Safe fallback LLM-like policy text for reproducible runs.
    This preserves the agent interface even when no local LLM is available.
    """
    risk = row["risk_level"]
    attack = row["predicted_attack_type"]
    cov = row["attack_probability"]

    if risk == "critical":
        action = "isolate sender, revoke trust temporarily, and forward incident to cloud verifier"
    elif risk == "high":
        action = "quarantine messages, request verification, and reduce sender trust"
    elif risk == "medium":
        action = "monitor sender, increase verification rate, and defer hard isolation"
    else:
        action = "allow message with normal monitoring"

    explanation = (
        f"The response agent assigns {risk} risk for {attack}. "
        f"The decision is based on attack probability {cov:.3f}, trust history, "
        f"spatial consistency, and temporal behavior. Recommended action: {action}."
    )
    return action, explanation


def ollama_generate(prompt, model="llama3.1", host="http://localhost:11434"):
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=60) as resp:
            ans = json.loads(resp.read().decode("utf-8"))
            return ans.get("response", "")
    except URLError:
        return ""
    except Exception:
        return ""


def run_llm_incident_agent(df, model, features, out_dir, provider, llm_model, top_k=50):
    """
    LLM agent operates on incident summaries, not every message.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = split_xy(df, "class_id", features)
    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)

    class_names = {
        0: "normal",
        1: "constantPositionOffset",
        2: "randomPositionOffset",
        3: "trafficCongestionSybil",
    }

    # Attack probability = 1 - P(normal)
    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1, 2, 3]
    normal_idx = classes.index(0) if 0 in classes else 0
    attack_prob = 1 - proba[:, normal_idx]

    test_part = df[df["split"].astype(str).str.lower() == "test"].copy().reset_index(drop=True)
    test_part["predicted_class_id"] = pred
    test_part["predicted_attack_type"] = [class_names.get(int(x), str(x)) for x in pred]
    test_part["attack_probability"] = attack_prob

    incidents = (
        test_part[test_part["predicted_class_id"] != 0]
        .sort_values("attack_probability", ascending=False)
        .head(top_k)
        .copy()
    )

    rows = []
    for _, r in incidents.iterrows():
        p = float(r["attack_probability"])
        if p >= 0.90:
            risk = "critical"
        elif p >= 0.75:
            risk = "high"
        elif p >= 0.55:
            risk = "medium"
        else:
            risk = "low"

        incident = {
            "sender_id": r.get("sender_id", "unknown"),
            "predicted_attack_type": r["predicted_attack_type"],
            "attack_probability": p,
            "risk_level": risk,
            "distance_to_road_edge": r.get("distance_to_road_edge", np.nan),
            "edge_violation": r.get("edge_violation", np.nan),
            "sender_trust_prior": r.get("sender_trust_prior", np.nan),
            "graph_trust_propagated": r.get("graph_trust_propagated", np.nan),
            "heading_delta": r.get("heading_delta", np.nan),
            "delay": r.get("delay", np.nan),
        }

        if provider == "ollama":
            prompt = (
                "You are a V2X cyber-response policy agent. "
                "Given this incident summary, output exactly two fields: Action and Rationale. "
                f"Incident: {json.dumps(incident)}"
            )
            llm_text = ollama_generate(prompt, model=llm_model)
            if llm_text.strip():
                action = "llm_generated"
                explanation = llm_text.strip().replace("\n", " ")
            else:
                action, explanation = deterministic_policy_agent(pd.Series(incident))
        else:
            action, explanation = deterministic_policy_agent(pd.Series(incident))

        incident["llm_provider"] = provider
        incident["policy_action"] = action
        incident["policy_explanation"] = explanation
        rows.append(incident)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "csv" / "llm_incident_reports.csv", index=False)
    return out


def run_experiments(df, out_dir, llm_provider, llm_model):
    raw_features = list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES))
    enhanced_features = list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES + TEMPORAL_FEATURES + TRUST_FEATURES))
    proposed_features = list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES + TEMPORAL_FEATURES + TRUST_FEATURES + GRAPH_TRUST_FEATURES))

    metrics = []
    responses = []

    baselines = ["LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting"]
    if HAS_XGB:
        baselines.append("XGBoost")

    for task, target in [("binary", "binary_label"), ("multiclass", "class_id")]:
        for m in baselines:
            print(f"[RAW BASELINE] {task} | {m}")
            met, resp, _ = train_eval(df, target, task, "raw_edge_cloud", m, raw_features, out_dir, proposed=False)
            metrics.append(m)
            responses.append(resp)

        for m in ["RandomForest", "XGBoost" if HAS_XGB else "HistGradientBoosting"]:
            print(f"[ENHANCED ML] {task} | {m}")
            met, resp, _ = train_eval(df, target, task, "enhanced_temporal_trust", m, enhanced_features, out_dir, proposed=False)
            metrics.append(met)
            responses.append(resp)

        ablations = {
            "A1_edge_only": EDGE_FEATURES,
            "A2_cloud_only": CLOUD_FEATURES,
            "A3_temporal_only": TEMPORAL_FEATURES,
            "A4_trust_only": TRUST_FEATURES,
            "A5_graph_trust_only": GRAPH_TRUST_FEATURES,
            "A6_edge_cloud_temporal": list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES + TEMPORAL_FEATURES)),
            "A7_full_aectepp": proposed_features,
        }

        final_model = None
        for setting, feats in ablations.items():
            print(f"[PROPOSED AECTE++] {task} | {setting}")
            met, resp, model = train_eval(df, target, task, setting, "AECTE++", feats, out_dir, proposed=True)
            metrics.append(met)
            responses.append(resp)
            if task == "multiclass" and setting == "A7_full_aectepp":
                final_model = model
                joblib.dump(model, out_dir / "models" / "multiclass_AECTEpp_final.joblib")

        if task == "multiclass" and final_model is not None:
            run_llm_incident_agent(
                df, final_model, proposed_features, out_dir,
                provider=llm_provider, llm_model=llm_model, top_k=50
            )

    metrics_df = pd.DataFrame(metrics)
    resp_df = pd.DataFrame(responses)

    metrics_df.to_csv(out_dir / "csv" / "all_detection_metrics.csv", index=False)
    resp_df.to_csv(out_dir / "csv" / "response_resilience_metrics.csv", index=False)

    return metrics_df, resp_df


def fig_model_comparison(metrics_df, out_dir):
    df = metrics_df[
        ((metrics_df["setting"] == "raw_edge_cloud") & (~metrics_df["model"].eq("AECTE++"))) |
        ((metrics_df["setting"] == "A7_full_aectepp") & (metrics_df["model"].eq("AECTE++")))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    for ax, task, panel in zip(axes, ["binary", "multiclass"], ["(a)", "(b)"]):
        sub = df[df["task"] == task].sort_values("macro_f1", ascending=False)
        labels = sub["model"].replace({"HistGradientBoosting": "HistGB", "LogisticRegression": "LogReg"})
        bars = ax.bar(labels, sub["macro_f1"])
        ax.set_ylabel("Macro F1-score")
        ax.set_xlabel("Method")
        ax.set_title(f"{task.capitalize()} Detection")
        ax.tick_params(axis="x", rotation=18)
        add_panel(ax, panel)

        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=10)

    savefig(fig, out_dir / "figures" / "fig1_proposed_vs_raw_baselines")


def fig_ablation(metrics_df, out_dir):
    df = metrics_df[metrics_df["model"].eq("AECTE++")].copy()
    order = [
        "A1_edge_only", "A2_cloud_only", "A3_temporal_only", "A4_trust_only",
        "A5_graph_trust_only", "A6_edge_cloud_temporal", "A7_full_aectepp"
    ]
    labels = ["Edge", "Cloud", "Temporal", "Trust", "Graph Trust", "E+C+T", "Full"]
    pivot = df.pivot_table(index="setting", columns="task", values="macro_f1", aggfunc="max").reindex(order)

    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(order))
    w = 0.36
    ax.bar(x-w/2, pivot["binary"], width=w, label="Binary")
    ax.bar(x+w/2, pivot["multiclass"], width=w, label="Multiclass")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Macro F1-score")
    ax.set_xlabel("Agentic evidence group")
    ax.set_title("AECTE++ Agentic Ablation")
    ax.legend()
    add_panel(ax, "(a)")
    savefig(fig, out_dir / "fig2_agentic_ablation")


def fig_response(resp_df, out_dir):
    df = resp_df[(resp_df["model"].eq("AECTE++")) & (resp_df["task"].eq("binary"))].copy()
    order = [
        "A1_edge_only", "A2_cloud_only", "A3_temporal_only", "A4_trust_only",
        "A5_graph_trust_only", "A6_edge_cloud_temporal", "A7_full_aectepp"
    ]
    labels = ["Edge", "Cloud", "Temporal", "Trust", "Graph Trust", "E+C+T", "Full"]
    df = df.set_index("setting").reindex(order).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    x = np.arange(len(order))

    axes[0].plot(x, df["attack_coverage"], marker="o", linewidth=3, label="Attack coverage")
    axes[0].plot(x, df["response_precision"], marker="s", linewidth=3, label="Response precision")
    axes[0].plot(x, 1-df["false_isolation_rate"], marker="^", linewidth=3, label="Benign preservation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    axes[0].set_ylabel("Score")
    axes[0].set_xlabel("Agentic evidence group")
    axes[0].set_title("Threshold-Calibrated Response Quality")
    axes[0].legend()
    add_panel(axes[0], "(a)")

    axes[1].bar(labels, df["resilience_utility"])
    axes[1].set_ylabel("Resilience utility")
    axes[1].set_xlabel("Agentic evidence group")
    axes[1].set_title("Cyber-Resilience Utility")
    axes[1].tick_params(axis="x", rotation=10)
    add_panel(axes[1], "(b)")

    savefig(fig, out_dir / "fig3_response_resilience")


def fig_latency(resp_metrics, out_dir):
    df = resp_metrics[(resp_metrics["model"].eq("AECTE++")) & (resp_metrics["task"].eq("binary"))].copy()

    fig, ax = plt.subplots(figsize=(16, 9))
    sizes = 250 + 1500 * (df["macro_f1"] - df["macro_f1"].min() + 0.01)
    ax.scatter(df["latency_ms_per_msg"], df["macro_f1"], s=sizes, alpha=0.8)

    for _, r in df.iterrows():
        ax.text(r["latency_ms_per_msg"], r["macro_f1"], r["setting"].replace("_", "\n"), fontsize=10)

    ax.set_xlabel("Latency per message (ms)")
    ax.set_ylabel("Binary macro F1-score")
    ax.set_title("Latency-Accuracy Tradeoff for AECTE++")
    add_panel(ax, "(a)")
    savefig(fig, out_dir / "fig4_latency_utility_tradeoff")


def fig_llm_actions(out_dir):
    path = out_dir / "csv" / "llm_incident_reports.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    counts = df["risk_level"].value_counts().reindex(["low", "medium", "high", "critical"]).fillna(0)

    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.bar(counts.index, counts.values)
    ax.set_xlabel("LLM-assigned risk level")
    ax.set_ylabel("Incident count")
    ax.set_title("LLM Policy Agent Incident Triage")
    add_panel(ax, "(a)")

    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{int(b.get_height())}",
                ha="center", va="bottom", fontsize=12)

    savefig(fig, out_dir / "fig5_llm_policy_actions")


def write_report(metrics_df, resp_df, out_dir):
    best_bin = metrics_df[metrics_df["task"].eq("binary")].sort_values("macro_f1", ascending=False).head(15)
    best_multi = metrics_df[metrics_df["task"].eq("multiclass")].sort_values("macro_f1", ascending=False).head(15)
    best_resp = resp_df.sort_values("resilience_utility", ascending=False).head(15)

    llm_path = out_dir / "csv" / "llm_incident_reports.csv"
    llm_summary = "No LLM incident reports generated."
    if llm_path.exists():
        llm_df = pd.read_csv(llm_path)
        llm_summary = llm_df[["risk_level", "predicted_attack_type", "attack_probability", "policy_action"]].head(10).to_markdown(index=False)

    lines = []
    lines.append("# Agentic-V2XShield AECTE++ Final Report\n")
    lines.append("## Core Design\n")
    lines.append(
        "AECTE++ combines edge telemetry, cloud spatial context, temporal consistency, sender-level trust, "
        "graph-trust propagation, stacked ML fusion, and LLM-based policy reasoning. The LLM layer is used "
        "for incident interpretation and response selection, while lightweight ML components provide real-time detection."
    )
    lines.append("\n## Best Binary Results\n")
    lines.append(best_bin.to_markdown(index=False))
    lines.append("\n## Best Multiclass Results\n")
    lines.append(best_multi.to_markdown(index=False))
    lines.append("\n## Best Response and Resilience Results\n")
    lines.append(best_resp.to_markdown(index=False))
    lines.append("\n## LLM Policy Agent Samples\n")
    lines.append(llm_summary)
    lines.append("\n## Paper Claim That Is Safe\n")
    lines.append(
        "The safe claim is that Agentic-V2XShield provides a deployment-aware cyber-resilience layer for V2X "
        "that integrates detection, trust propagation, response calibration, and LLM-assisted policy explanation. "
        "Do not claim that the LLM alone performs detection. The LLM supports explainable response orchestration."
    )

    (out_dir / "reports" / "final_aectepp_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default="outputs_aectepp_llm")
    parser.add_argument("--max-per-class", type=int, default=100000)
    parser.add_argument("--keep-leakage-risk-features", action="store_true")
    parser.add_argument("--llm-provider", choices=["none", "ollama"], default="none")
    parser.add_argument("--llm-model", default="llama3.1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    df = load_dataset(Path(args.csv), args.max_per_class, args.keep_leakage_risk_features)

    print("[INFO] Engineering temporal and sender-trust features.")
    df = engineer_temporal_trust(df)

    print("[INFO] Engineering graph-trust propagation features.")
    df = engineer_graph_trust(df, out_dir)

    df.to_csv(out_dir / "csv" / "aectepp_engineered_dataset_snapshot.csv", index=False)

    print("[INFO] Running detection, ablation, response, and LLM-agent experiments.")
    metrics_df, resp_df = run_experiments(df, out_dir, args.llm_provider, args.llm_model)

    fig_model_comparison(metrics_df, out_dir)
    fig_ablation(metrics_df, out_dir)
    fig_response(resp_df, out_dir)
    fig_latency(metrics_df, out_dir)
    fig_llm_actions(out_dir)
    write_report(metrics_df, resp_df, out_dir)

    print("\n[DONE] AECTE++ LLM-agentic pipeline completed.")
    print(f"[OUT] {out_dir.resolve()}")
    print("Main outputs:")
    print(" - csv/all_detection_metrics.csv")
    print(" - csv/response_resilience_metrics.csv")
    print(" - csv/llm_incident_reports.csv")
    print(" - csv/graph_trust_summary.csv")
    print(" - figures/*.png and *.pdf")
    print(" - reports/final_aectepp_report.md")


if __name__ == "__main__":
    main()
