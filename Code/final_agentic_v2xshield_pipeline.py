#!/usr/bin/env python
"""
Agentic-V2XShield Final Redesign

Fixes the earlier issues:
1. Removes leakage-prone identifiers by default.
2. Adds temporal consistency features.
3. Adds sender-level prior trust features.
4. Runs real binary and multiclass evaluation.
5. Runs real edge/cloud/temporal/trust ablations.
6. Calibrates cyber-response thresholds on validation data.
7. Exports all results to CSV.
8. Generates IEEE-style 16:9 600-DPI PNG and PDF figures.

Run:
    python final_agentic_v2xshield_pipeline.py ^
      --csv "outputs_multiclass\\veremi_multiclass_balanced.csv" ^
      --out-dir "outputs_final_agentic_v2xshield" ^
      --max-per-class 100000

Debug:
    python final_agentic_v2xshield_pipeline.py ^
      --csv "outputs_multiclass\\veremi_multiclass_balanced.csv" ^
      --out-dir "outputs_final_agentic_v2xshield_test" ^
      --max-per-class 30000
"""

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score,
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
    "figure.figsize": (16, 9), "figure.dpi": 600, "savefig.dpi": 600,
    "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 18,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "font.family": "DejaVu Sans",
})

EDGE_FEATURES = [
    "delay", "sender_spd", "sender_acl", "sender_hed", "receiver_spd", "receiver_acl", "receiver_hed",
    "speed_delta", "accel_delta", "heading_delta", "abs_sender_speed", "abs_sender_accel",
]

CLOUD_FEATURES = [
    "sender_pos_x", "sender_pos_y", "receiver_pos_x", "receiver_pos_y", "sender_receiver_distance",
    "distance_to_road_edge", "edge_violation", "sender_pos_noise_x", "sender_pos_noise_y",
    "receiver_pos_noise_x", "receiver_pos_noise_y", "sender_spd_noise", "receiver_spd_noise",
    "sender_acl_noise", "receiver_acl_noise", "sender_hed_noise", "receiver_hed_noise",
    "sender_driver_profile", "receiver_driver_profile",
]

TEMPORAL_FEATURES = [
    "sender_spd_roll_mean_5", "sender_acl_roll_mean_5", "heading_delta_roll_mean_5",
    "speed_delta_roll_mean_5", "delay_roll_mean_5", "sender_spd_diff", "sender_acl_diff",
    "sender_hed_diff", "sender_pos_step_dist", "msg_time_gap",
]

TRUST_FEATURES = [
    "sender_msg_count_so_far", "sender_attack_rate_prior", "sender_edge_violation_rate_prior",
    "sender_delay_mean_prior", "sender_road_edge_mean_prior", "sender_trust_prior", "risk_rule_score",
]

LEAKAGE_DROP_DEFAULT = ["messageID", "sender_alias", "rcvTime", "sendTime"]
DROP_ALWAYS = ["source_file", "attacker_raw", "class_name", "class_id", "binary_label", "split"]


def ensure_dirs(out_dir: Path):
    for sub in ["csv", "figures", "reports", "models", "confusion_matrices", "class_reports"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_panel(ax, label: str):
    ax.text(0.01, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontsize=18,
            fontweight="bold", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))


def clean_name(x):
    x = str(x).replace("num__", "").replace("cat__", "")
    x = x.replace("sender_", "S-").replace("receiver_", "R-")
    x = x.replace("_", " ").replace("spd", "speed").replace("acl", "accel").replace("hed", "heading")
    x = x.replace("road edge", "road-edge").replace("sender receiver distance", "S-R distance")
    return x


def heading_diff(a, b):
    try:
        if pd.isna(a) or pd.isna(b):
            return np.nan
        d = abs(float(a) - float(b)) % 360.0
        return min(d, 360.0 - d)
    except Exception:
        return np.nan


def load_dataset(csv_path: Path, max_per_class: int, keep_leakage: bool):
    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    required = ["class_id", "class_name", "binary_label", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_per_class and max_per_class > 0:
        df = (df.groupby("class_id", group_keys=False)
                .apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42))
                .reset_index(drop=True))
    if not keep_leakage:
        df = df.drop(columns=[c for c in LEAKAGE_DROP_DEFAULT if c in df.columns], errors="ignore")
    print(f"[INFO] Records: {len(df):,}")
    print(f"[INFO] Classes: {df['class_name'].nunique()}")
    return df


def add_temporal_and_trust_features(df: pd.DataFrame) -> pd.DataFrame:
    """History-safe features. Prior trust uses shift/cumulative history only."""
    df = df.copy()
    if "sender_id" not in df.columns:
        df["sender_id"] = "unknown_sender"

    for c in ["sender_pos_x", "sender_pos_y", "sender_spd", "sender_acl", "sender_hed", "delay",
              "heading_delta", "speed_delta", "sender_receiver_distance", "distance_to_road_edge", "edge_violation"]:
        if c not in df.columns:
            df[c] = np.nan

    sort_cols = ["split", "sender_id"]
    if "sendTime" in df.columns:
        sort_cols.append("sendTime")
    elif "messageID" in df.columns:
        sort_cols.append("messageID")
    else:
        sort_cols.append("class_id")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    g = df.groupby(["split", "sender_id"], sort=False)

    df["prev_sender_pos_x"] = g["sender_pos_x"].shift(1)
    df["prev_sender_pos_y"] = g["sender_pos_y"].shift(1)
    df["sender_pos_step_dist"] = np.sqrt((df["sender_pos_x"] - df["prev_sender_pos_x"]) ** 2 +
                                          (df["sender_pos_y"] - df["prev_sender_pos_y"]) ** 2)
    df["msg_time_gap"] = g["sendTime"].diff() if "sendTime" in df.columns else np.nan
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
        df[dst] = g[src].rolling(window=5, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

    df["sender_msg_count_so_far"] = g.cumcount()
    prior_attack_sum = g["binary_label"].cumsum() - df["binary_label"]
    denom = df["sender_msg_count_so_far"].replace(0, np.nan)
    df["sender_attack_rate_prior"] = prior_attack_sum / denom
    prior_edge_sum = g["edge_violation"].cumsum() - df["edge_violation"].fillna(0)
    df["sender_edge_violation_rate_prior"] = prior_edge_sum / denom
    df["sender_delay_mean_prior"] = (g["delay"].cumsum() - df["delay"].fillna(0)) / denom
    df["sender_road_edge_mean_prior"] = (g["distance_to_road_edge"].cumsum() - df["distance_to_road_edge"].fillna(0)) / denom
    df["sender_trust_prior"] = 1.0 - df["sender_attack_rate_prior"]

    # Label-free rule score used for response, not a replacement for ML.
    q_delay = df["delay"].quantile(0.95)
    q_dist = df["sender_receiver_distance"].quantile(0.95)
    med_speed_delta = df["speed_delta"].median(skipna=True)
    df["risk_rule_score"] = 0.0
    df["risk_rule_score"] += (df["edge_violation"].fillna(0) > 0).astype(float) * 0.35
    df["risk_rule_score"] += (df["heading_delta"].fillna(0) > 90).astype(float) * 0.20
    df["risk_rule_score"] += (df["speed_delta"].fillna(0) > med_speed_delta).astype(float) * 0.15
    df["risk_rule_score"] += (df["delay"].fillna(0) > q_delay).astype(float) * 0.15
    df["risk_rule_score"] += (df["sender_receiver_distance"].fillna(0) > q_dist).astype(float) * 0.15

    return df.drop(columns=["prev_sender_pos_x", "prev_sender_pos_y", "sender_hed_prev"], errors="ignore")


def split_xy(df, target, features):
    train = df[df["split"].astype(str).str.lower() == "train"].copy()
    val = df[df["split"].astype(str).str.lower() == "validation"].copy()
    test = df[df["split"].astype(str).str.lower() == "test"].copy()

    def make_x(part):
        X = part.drop(columns=[c for c in DROP_ALWAYS if c in part.columns], errors="ignore")
        if features is not None:
            keep = [c for c in features if c in X.columns]
            X = X[keep].copy()
        return X
    return make_x(train), train[target].astype(int), make_x(val), val[target].astype(int), make_x(test), test[target].astype(int)


def make_preprocessor(X):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])


def make_model(name, X_train, n_classes):
    pre = make_preprocessor(X_train)
    if name == "LogisticRegression":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=42)
    elif name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=220, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42)
    elif name == "ExtraTrees":
        clf = ExtraTreesClassifier(n_estimators=260, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42)
    elif name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(max_iter=260, learning_rate=0.055, max_leaf_nodes=47,
                                             l2_regularization=0.04, random_state=42)
    elif name == "XGBoost":
        if not HAS_XGB:
            raise RuntimeError("XGBoost is not available.")
        clf = XGBClassifier(n_estimators=420, max_depth=7, learning_rate=0.045, subsample=0.9,
                            colsample_bytree=0.9,
                            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                            eval_metric="mlogloss" if n_classes > 2 else "logloss",
                            tree_method="hist", random_state=42, n_jobs=-1)
    else:
        raise ValueError(name)
    return Pipeline([("preprocess", pre), ("clf", clf)])


def get_attack_probability(model, X, target):
    if not hasattr(model, "predict_proba"):
        pred = model.predict(X)
        return (pred == 1).astype(float) if target == "binary_label" else (pred != 0).astype(float)
    proba = model.predict_proba(X)
    classes = list(model.named_steps["clf"].classes_) if isinstance(model, Pipeline) else list(model.classes_)
    if target == "binary_label":
        return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
    return 1.0 - proba[:, classes.index(0)] if 0 in classes else 1.0 - np.max(proba, axis=1)


def tune_threshold(y_true, prob, target):
    true_attack = (y_true.values == 1) if target == "binary_label" else (y_true.values != 0)
    best = {"threshold": 0.5, "utility": -999.0}
    for t in np.linspace(0.05, 0.95, 91):
        pred = prob >= t
        tp = np.sum(true_attack & pred); fp = np.sum((~true_attack) & pred)
        fn = np.sum(true_attack & (~pred)); tn = np.sum((~true_attack) & (~pred))
        cov = tp / max(tp + fn, 1); fiso = fp / max(fp + tn, 1); prec = tp / max(tp + fp, 1)
        util = 0.55 * cov + 0.35 * prec - 0.25 * fiso
        if util > best["utility"]:
            best = {"threshold": float(t), "utility": float(util)}
    return best


def response_metrics(y_true, prob, thr, target):
    true_attack = (y_true.values == 1) if target == "binary_label" else (y_true.values != 0)
    pred = prob >= thr
    tp = int(np.sum(true_attack & pred)); fp = int(np.sum((~true_attack) & pred))
    fn = int(np.sum(true_attack & (~pred))); tn = int(np.sum((~true_attack) & (~pred)))
    cov = tp / max(tp + fn, 1); fiso = fp / max(fp + tn, 1); prec = tp / max(tp + fp, 1)
    util = 0.55 * cov + 0.35 * prec - 0.25 * fiso
    return {"threshold": thr, "attack_coverage": cov, "false_isolation_rate": fiso,
            "response_precision": prec, "resilience_utility": util,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def eval_classifier(y_true, y_pred, task, setting, model_name, train_s, infer_s):
    return {"task": task, "setting": setting, "model": model_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "training_s": train_s, "inference_s": infer_s,
            "latency_ms_per_msg": infer_s / max(len(y_true), 1) * 1000.0,
            "test_records": len(y_true)}


def train_eval(df, target, task, setting, model_name, features, out_dir):
    X_train, y_train, X_val, y_val, X_test, y_test = split_xy(df, target, features)
    model = make_model(model_name, X_train, len(np.unique(y_train)))
    t0 = time.perf_counter(); model.fit(X_train, y_train); train_s = time.perf_counter() - t0
    t1 = time.perf_counter(); y_pred = model.predict(X_test); infer_s = time.perf_counter() - t1
    clf_metrics = eval_classifier(y_test, y_pred, task, setting, model_name, train_s, infer_s)
    val_prob = get_attack_probability(model, X_val, target)
    thr = tune_threshold(y_val, val_prob, target)
    test_prob = get_attack_probability(model, X_test, target)
    resp = response_metrics(y_test, test_prob, thr["threshold"], target)
    resp.update({"task": task, "setting": setting, "model": model_name,
                 "validation_threshold": thr["threshold"], "validation_utility": thr["utility"]})
    labels = sorted(np.unique(np.concatenate([y_train.unique(), y_test.unique()])))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / "confusion_matrices" / f"{task}_{setting}_{model_name}_cm.csv")
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(out_dir / "class_reports" / f"{task}_{setting}_{model_name}_report.csv")
    return clf_metrics, resp, model


def run_experiments(df, out_dir):
    metrics_rows, response_rows = [], []
    baseline_models = ["LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting"]
    if HAS_XGB:
        baseline_models.append("XGBoost")
    full_features = list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES + TEMPORAL_FEATURES + TRUST_FEATURES))
    for task, target in [("binary", "binary_label"), ("multiclass", "class_id")]:
        for m in baseline_models:
            print(f"[BASELINE] {task} | {m}")
            met, resp, _ = train_eval(df, target, task, "all_features", m, full_features, out_dir)
            metrics_rows.append(met); response_rows.append(resp)
    engine = "XGBoost" if HAS_XGB else "HistGradientBoosting"
    ablations = {
        "E_edge_only": EDGE_FEATURES,
        "C_cloud_only": CLOUD_FEATURES,
        "T_temporal_only": TEMPORAL_FEATURES,
        "R_trust_only": TRUST_FEATURES,
        "EC_edge_cloud": list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES)),
        "ECT_edge_cloud_temporal": list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES + TEMPORAL_FEATURES)),
        "ECTR_full": full_features,
    }
    for task, target in [("binary", "binary_label"), ("multiclass", "class_id")]:
        for setting, feats in ablations.items():
            print(f"[AECTE] {task} | {setting}")
            met, resp, model = train_eval(df, target, task, setting, engine, feats, out_dir)
            met["model"] = f"AECTE-{engine}"; resp["model"] = f"AECTE-{engine}"
            metrics_rows.append(met); response_rows.append(resp)
            if setting == "ECTR_full":
                joblib.dump(model, out_dir / "models" / f"{task}_AECTE_full.joblib")
    metrics_df = pd.DataFrame(metrics_rows)
    response_df = pd.DataFrame(response_rows)
    metrics_df.to_csv(out_dir / "csv" / "classifier_metrics.csv", index=False)
    response_df.to_csv(out_dir / "csv" / "response_resilience_metrics.csv", index=False)
    return metrics_df, response_df


def fig1_overall(metrics_df, out_dir):
    df = metrics_df[((metrics_df["setting"] == "all_features") & (~metrics_df["model"].str.contains("AECTE"))) |
                    ((metrics_df["setting"] == "ECTR_full") & (metrics_df["model"].str.contains("AECTE")))].copy()
    df["display"] = df["model"].replace({"HistGradientBoosting": "HistGB", "LogisticRegression": "LogReg",
                                           "AECTE-XGBoost": "AECTE", "AECTE-HistGradientBoosting": "AECTE"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for ax, task, panel in zip(axes, ["binary", "multiclass"], ["(a)", "(b)"]):
        sub = df[df["task"] == task].sort_values("macro_f1", ascending=False)
        bars = ax.bar(sub["display"], sub["macro_f1"])
        ax.set_ylabel("Macro F1-score"); ax.set_xlabel("Model"); ax.set_title(f"{task.capitalize()} Performance")
        ax.tick_params(axis="x", rotation=18); add_panel(ax, panel)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    savefig(fig, out_dir / "figures" / "fig1_binary_multiclass_model_comparison")


def fig2_ablation(metrics_df, out_dir):
    order = ["E_edge_only", "C_cloud_only", "T_temporal_only", "R_trust_only", "EC_edge_cloud", "ECT_edge_cloud_temporal", "ECTR_full"]
    labels = ["Edge", "Cloud", "Temporal", "Trust", "Edge+Cloud", "E+C+T", "Full"]
    df = metrics_df[metrics_df["model"].str.contains("AECTE") & metrics_df["setting"].isin(order)].copy()
    pivot = df.pivot_table(index="setting", columns="task", values="macro_f1", aggfunc="max").reindex(order)
    fig, ax = plt.subplots(figsize=(16, 9)); x = np.arange(len(order)); w = 0.36
    ax.bar(x - w/2, pivot["binary"], width=w, label="Binary")
    ax.bar(x + w/2, pivot["multiclass"], width=w, label="Multiclass")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Macro F1-score"); ax.set_xlabel("Ablation setting"); ax.set_title("Real Edge-Cloud-Temporal-Trust Ablation")
    ax.legend(); add_panel(ax, "(a)"); savefig(fig, out_dir / "figures" / "fig2_real_ablation")


def fig3_response(response_df, out_dir):
    order = ["E_edge_only", "C_cloud_only", "T_temporal_only", "R_trust_only", "EC_edge_cloud", "ECT_edge_cloud_temporal", "ECTR_full"]
    labels = ["Edge", "Cloud", "Temporal", "Trust", "Edge+Cloud", "E+C+T", "Full"]
    df = response_df[response_df["model"].str.contains("AECTE") & response_df["task"].eq("binary")].copy()
    df = df.set_index("setting").reindex(order).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9)); x = np.arange(len(order))
    axes[0].plot(x, df["attack_coverage"], marker="o", linewidth=3, label="Attack coverage")
    axes[0].plot(x, df["response_precision"], marker="s", linewidth=3, label="Response precision")
    axes[0].plot(x, 1 - df["false_isolation_rate"], marker="^", linewidth=3, label="Benign preservation")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=10)
    axes[0].set_ylabel("Score"); axes[0].set_xlabel("Ablation setting"); axes[0].set_title("Threshold-Calibrated Response Quality")
    axes[0].legend(); add_panel(axes[0], "(a)")
    axes[1].bar(labels, df["resilience_utility"])
    axes[1].set_ylabel("Resilience utility"); axes[1].set_xlabel("Ablation setting"); axes[1].set_title("Cyber-Resilience Utility")
    axes[1].tick_params(axis="x", rotation=10); add_panel(axes[1], "(b)")
    savefig(fig, out_dir / "figures" / "fig3_response_resilience")


def fig4_latency(metrics_df, out_dir):
    df = metrics_df[metrics_df["model"].str.contains("AECTE") & metrics_df["task"].eq("binary")].copy()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(df["latency_ms_per_msg"], df["macro_f1"], s=260, alpha=0.8)
    for _, r in df.iterrows():
        ax.text(r["latency_ms_per_msg"], r["macro_f1"], r["setting"].replace("_", "\n"), fontsize=10)
    ax.set_xlabel("Latency per message (ms)"); ax.set_ylabel("Binary macro F1-score")
    ax.set_title("Deployment Tradeoff Between Detection Quality and Runtime"); add_panel(ax, "(a)")
    savefig(fig, out_dir / "figures" / "fig4_latency_detection_tradeoff")


def fig5_confusion(out_dir):
    cm_path = out_dir / "confusion_matrices" / "multiclass_ECTR_full_XGBoost_cm.csv"
    if not cm_path.exists():
        cm_path = out_dir / "confusion_matrices" / "multiclass_ECTR_full_HistGradientBoosting_cm.csv"
    if not cm_path.exists():
        return
    cm = pd.read_csv(cm_path, index_col=0).values.astype(float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = ["Normal", "Const. Pos.", "Random Pos.", "Sybil"]
    fig, ax = plt.subplots(figsize=(16, 9)); im = ax.imshow(norm, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title("Normalized Multiclass Confusion Matrix")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); add_panel(ax, "(a)")
    savefig(fig, out_dir / "figures" / "fig5_multiclass_confusion_matrix")


def fig6_feature_importance(df, out_dir):
    model_name = "XGBoost" if HAS_XGB else "ExtraTrees"
    features = list(dict.fromkeys(EDGE_FEATURES + CLOUD_FEATURES + TEMPORAL_FEATURES + TRUST_FEATURES))
    X_train, y_train, _, _, _, _ = split_xy(df, "class_id", features)
    model = make_model(model_name, X_train, len(np.unique(y_train)))
    model.fit(X_train, y_train)
    clf = model.named_steps["clf"]; pre = model.named_steps["preprocess"]
    if not hasattr(clf, "feature_importances_"):
        return
    names = [clean_name(n) for n in pre.get_feature_names_out()]
    fi = pd.DataFrame({"feature": names, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
    fi.to_csv(out_dir / "csv" / "final_feature_importance.csv", index=False)
    top = fi.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(16, 9)); ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Relative importance"); ax.set_ylabel("Security evidence")
    ax.set_title("Interpretable Evidence Used by Agentic-V2XShield"); add_panel(ax, "(a)")
    savefig(fig, out_dir / "figures" / "fig6_interpretable_security_evidence")


def write_report(metrics_df, response_df, out_dir):
    best_bin = metrics_df[metrics_df["task"] == "binary"].sort_values("macro_f1", ascending=False).head(12)
    best_multi = metrics_df[metrics_df["task"] == "multiclass"].sort_values("macro_f1", ascending=False).head(12)
    best_resp = response_df.sort_values("resilience_utility", ascending=False).head(12)
    lines = []
    lines.append("# Agentic-V2XShield Final Evaluation Report\n")
    lines.append("## What Was Fixed\n")
    lines.append("- Removed leakage-prone identifiers by default.")
    lines.append("- Added temporal consistency features.")
    lines.append("- Added sender-level prior trust and behavior-history features.")
    lines.append("- Replaced fake ablations with real edge/cloud/temporal/trust ablations.")
    lines.append("- Calibrated response thresholds on validation data.")
    lines.append("- Reported binary detection, multiclass classification, latency, and response metrics.\n")
    lines.append("## Best Binary Results\n"); lines.append(best_bin.to_markdown(index=False))
    lines.append("\n## Best Multiclass Results\n"); lines.append(best_multi.to_markdown(index=False))
    lines.append("\n## Best Response/Resilience Results\n"); lines.append(best_resp.to_markdown(index=False))
    lines.append("\n## Recommended Paper Positioning\n")
    lines.append("The defensible contribution is not a generic classifier. Present Agentic-V2XShield as a deployment-aware edge-cloud cyber-resilience framework that combines fast edge evidence, cloud spatial context, temporal consistency, and sender-level trust priors for V2X attack detection and response.")
    lines.append("\n## Warning\n")
    lines.append("If multiclass macro-F1 remains moderate, do not claim state-of-the-art attack classification. Position multiclass classification as attack-type awareness for response selection, while the stronger claim should center on binary detection, latency, trust-aware mitigation, and resilience utility.")
    (out_dir / "reports" / "final_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default="outputs_final_agentic_v2xshield")
    parser.add_argument("--max-per-class", type=int, default=100000)
    parser.add_argument("--keep-leakage-risk-features", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir); ensure_dirs(out_dir)
    df = load_dataset(Path(args.csv), args.max_per_class, args.keep_leakage_risk_features)
    print("[INFO] Engineering temporal and trust features.")
    df = add_temporal_and_trust_features(df)
    df.to_csv(out_dir / "csv" / "engineered_dataset_snapshot.csv", index=False)
    metrics_df, response_df = run_experiments(df, out_dir)
    fig1_overall(metrics_df, out_dir); fig2_ablation(metrics_df, out_dir)
    fig3_response(response_df, out_dir); fig4_latency(metrics_df, out_dir)
    fig5_confusion(out_dir); fig6_feature_importance(df, out_dir)
    write_report(metrics_df, response_df, out_dir)
    print("\n[DONE] Final redesigned Agentic-V2XShield evaluation completed.")
    print(f"[OUT] {out_dir.resolve()}")
    print("Main outputs:")
    print(" - csv/classifier_metrics.csv")
    print(" - csv/response_resilience_metrics.csv")
    print(" - csv/final_feature_importance.csv")
    print(" - figures/*.png and *.pdf")
    print(" - reports/final_evaluation_report.md")


if __name__ == "__main__":
    main()
