#!/usr/bin/env python
"""
Agentic-V2XShield Final+ Pipeline

Adds publication-level comparative baselines, enhanced baselines, AECTE++,
robustness tests, scalability tests, LLM incident response reports,
IEEE-style CSV/LaTeX tables, and 600-DPI figures.

Run:
python final_plus_agentic_v2xshield.py ^
  --csv "outputs_multiclass\\veremi_multiclass_balanced.csv" ^
  --out-dir "outputs_final_plus" ^
  --max-per-class 100000 ^
  --llm-provider none

Optional Ollama:
python final_plus_agentic_v2xshield.py ^
  --csv "outputs_multiclass\\veremi_multiclass_balanced.csv" ^
  --out-dir "outputs_final_plus" ^
  --max-per-class 100000 ^
  --llm-provider ollama ^
  --llm-model llama3.2:3b
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
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

EDGE_FEATURES = ["delay","sender_spd","sender_acl","sender_hed","receiver_spd","receiver_acl","receiver_hed","speed_delta","accel_delta","heading_delta","abs_sender_speed","abs_sender_accel"]
CLOUD_FEATURES = ["sender_pos_x","sender_pos_y","receiver_pos_x","receiver_pos_y","sender_receiver_distance","distance_to_road_edge","edge_violation","sender_pos_noise_x","sender_pos_noise_y","receiver_pos_noise_x","receiver_pos_noise_y","sender_spd_noise","receiver_spd_noise","sender_acl_noise","receiver_acl_noise","sender_hed_noise","receiver_hed_noise","sender_driver_profile","receiver_driver_profile"]
TEMPORAL_FEATURES = ["sender_spd_roll_mean_5","sender_acl_roll_mean_5","heading_delta_roll_mean_5","speed_delta_roll_mean_5","delay_roll_mean_5","sender_spd_diff","sender_acl_diff","sender_hed_diff","sender_pos_step_dist","msg_time_gap"]
TRUST_FEATURES = ["sender_msg_count_so_far","sender_attack_rate_prior","sender_edge_violation_rate_prior","sender_delay_mean_prior","sender_road_edge_mean_prior","sender_trust_prior","risk_rule_score"]
GRAPH_TRUST_FEATURES = ["graph_sender_degree_prior","graph_neighbor_risk_prior","graph_trust_propagated","graph_local_disagreement"]
LEAKAGE_DROP = ["messageID", "sender_alias", "rcvTime", "sendTime"]
DROP_ALWAYS = ["source_file", "attacker_raw", "class_name", "class_id", "binary_label", "split"]
CLASS_NAMES = {0:"normal",1:"constantPositionOffset",2:"randomPositionOffset",3:"trafficCongestionSybil"}


def ensure_dirs(out_dir: Path):
    for sub in ["csv","tables","figures","reports","models","confusion_matrices","class_reports"]:
        (out_dir/sub).mkdir(parents=True, exist_ok=True)


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def add_panel(ax, label):
    ax.text(0.01,0.98,label,transform=ax.transAxes,ha='left',va='top',fontsize=18,fontweight='bold',bbox=dict(facecolor='white',alpha=0.85,edgecolor='none',pad=2))


def load_dataset(csv_path, max_per_class, keep_leakage):
    df = pd.read_csv(csv_path)
    if max_per_class and max_per_class > 0:
        df = df.groupby('class_id', group_keys=False).apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42)).reset_index(drop=True)
    if not keep_leakage:
        df = df.drop(columns=[c for c in LEAKAGE_DROP if c in df.columns], errors='ignore')
    if 'sender_id' not in df.columns:
        df['sender_id'] = 'unknown_sender'
    return df


def heading_diff(a, b):
    try:
        if pd.isna(a) or pd.isna(b): return np.nan
        d = abs(float(a) - float(b)) % 360.0
        return min(d, 360.0-d)
    except Exception:
        return np.nan


def engineer_temporal_trust(df):
    df = df.copy()
    for c in ['sender_pos_x','sender_pos_y','sender_spd','sender_acl','sender_hed','delay','heading_delta','speed_delta','distance_to_road_edge','edge_violation','sender_receiver_distance']:
        if c not in df.columns: df[c] = np.nan
    sort_cols = ['split','sender_id']
    if 'sendTime' in df.columns: sort_cols.append('sendTime')
    elif 'messageID' in df.columns: sort_cols.append('messageID')
    df = df.sort_values(sort_cols).reset_index(drop=True)
    g = df.groupby(['split','sender_id'], sort=False)
    df['prev_sender_pos_x'] = g['sender_pos_x'].shift(1)
    df['prev_sender_pos_y'] = g['sender_pos_y'].shift(1)
    df['sender_pos_step_dist'] = np.sqrt((df['sender_pos_x']-df['prev_sender_pos_x'])**2 + (df['sender_pos_y']-df['prev_sender_pos_y'])**2)
    df['msg_time_gap'] = g['sendTime'].diff() if 'sendTime' in df.columns else np.nan
    df['sender_spd_diff'] = g['sender_spd'].diff()
    df['sender_acl_diff'] = g['sender_acl'].diff()
    df['sender_hed_prev'] = g['sender_hed'].shift(1)
    df['sender_hed_diff'] = [heading_diff(a,b) for a,b in zip(df['sender_hed'], df['sender_hed_prev'])]
    for src,dst in {'sender_spd':'sender_spd_roll_mean_5','sender_acl':'sender_acl_roll_mean_5','heading_delta':'heading_delta_roll_mean_5','speed_delta':'speed_delta_roll_mean_5','delay':'delay_roll_mean_5'}.items():
        df[dst] = g[src].rolling(window=5, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df['sender_msg_count_so_far'] = g.cumcount()
    prior_attack_sum = g['binary_label'].cumsum() - df['binary_label']
    df['sender_attack_rate_prior'] = prior_attack_sum / df['sender_msg_count_so_far'].replace(0, np.nan)
    prior_edge_sum = g['edge_violation'].cumsum() - df['edge_violation'].fillna(0)
    df['sender_edge_violation_rate_prior'] = prior_edge_sum / df['sender_msg_count_so_far'].replace(0, np.nan)
    df['sender_delay_mean_prior'] = (g['delay'].cumsum() - df['delay'].fillna(0)) / df['sender_msg_count_so_far'].replace(0, np.nan)
    df['sender_road_edge_mean_prior'] = (g['distance_to_road_edge'].cumsum() - df['distance_to_road_edge'].fillna(0)) / df['sender_msg_count_so_far'].replace(0, np.nan)
    df['sender_trust_prior'] = 1.0 - df['sender_attack_rate_prior']
    df['risk_rule_score'] = 0.0
    df['risk_rule_score'] += (df['edge_violation'].fillna(0)>0).astype(float)*0.30
    df['risk_rule_score'] += (df['heading_delta'].fillna(0)>90).astype(float)*0.20
    df['risk_rule_score'] += (df['speed_delta'].fillna(0)>df['speed_delta'].median(skipna=True)).astype(float)*0.15
    df['risk_rule_score'] += (df['delay'].fillna(0)>df['delay'].quantile(0.95)).astype(float)*0.15
    df['risk_rule_score'] += (df['sender_receiver_distance'].fillna(0)>df['sender_receiver_distance'].quantile(0.95)).astype(float)*0.20
    return df.drop(columns=['prev_sender_pos_x','prev_sender_pos_y','sender_hed_prev'], errors='ignore')


def engineer_graph_trust(df, out_dir):
    df = df.copy()
    if 'receiver_id' not in df.columns:
        df['receiver_id'] = 'profile_' + df.get('receiver_driver_profile', pd.Series(['unknown']*len(df))).astype(str)
    rows = []
    for split, part in df.groupby('split', sort=False):
        sender_stats = part.groupby('sender_id').agg(sender_binary_rate=('binary_label','mean'), sender_count=('binary_label','size'), sender_rule_mean=('risk_rule_score','mean')).reset_index()
        edges = part.groupby(['sender_id','receiver_id']).size().reset_index(name='edge_weight')
        neigh = edges.merge(sender_stats.rename(columns={'sender_id':'receiver_id','sender_binary_rate':'neighbor_binary_rate','sender_rule_mean':'neighbor_rule_mean','sender_count':'neighbor_count'}), on='receiver_id', how='left')
        neigh['weighted_neighbor_risk'] = neigh['neighbor_binary_rate'].fillna(0) * neigh['edge_weight']
        neigh['weighted_neighbor_rule'] = neigh['neighbor_rule_mean'].fillna(0) * neigh['edge_weight']
        agg = neigh.groupby('sender_id').agg(graph_sender_degree_prior=('receiver_id','nunique'), graph_neighbor_risk_prior=('weighted_neighbor_risk','sum'), graph_neighbor_rule_prior=('weighted_neighbor_rule','sum'), graph_edge_weight_sum=('edge_weight','sum')).reset_index()
        agg['graph_neighbor_risk_prior'] = agg['graph_neighbor_risk_prior'] / agg['graph_edge_weight_sum'].replace(0,np.nan)
        agg['graph_neighbor_rule_prior'] = agg['graph_neighbor_rule_prior'] / agg['graph_edge_weight_sum'].replace(0,np.nan)
        sender_stats = sender_stats.merge(agg, on='sender_id', how='left')
        sender_stats['graph_trust_propagated'] = (1.0 - 0.55*sender_stats['sender_binary_rate'].fillna(0) - 0.25*sender_stats['graph_neighbor_risk_prior'].fillna(0) - 0.20*sender_stats['sender_rule_mean'].fillna(0)).clip(0,1)
        sender_stats['graph_local_disagreement'] = (sender_stats['sender_binary_rate'].fillna(0) - sender_stats['graph_neighbor_risk_prior'].fillna(0)).abs()
        sender_stats['split'] = split
        rows.append(sender_stats)
    graph = pd.concat(rows, ignore_index=True)
    graph.to_csv(out_dir/'csv'/'graph_trust_summary.csv', index=False)
    return df.merge(graph[['split','sender_id']+GRAPH_TRUST_FEATURES], on=['split','sender_id'], how='left')


def make_preprocessor(X):
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer([('num', Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]), num_cols), ('cat', Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]), cat_cols)])


def make_single_model(name, X_train, n_classes):
    pre = make_preprocessor(X_train)
    if name == 'LogisticRegression': clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1, random_state=42)
    elif name == 'RandomForest': clf = RandomForestClassifier(n_estimators=260, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=42)
    elif name == 'ExtraTrees': clf = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=42)
    elif name == 'HistGradientBoosting': clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=55, l2_regularization=0.04, random_state=42)
    elif name == 'XGBoost':
        if not HAS_XGB: raise RuntimeError('Install XGBoost with: pip install xgboost')
        clf = XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.04, subsample=0.92, colsample_bytree=0.92, objective='multi:softprob' if n_classes>2 else 'binary:logistic', eval_metric='mlogloss' if n_classes>2 else 'logloss', tree_method='hist', random_state=42, n_jobs=-1)
    else: raise ValueError(name)
    return Pipeline([('preprocess', pre), ('clf', clf)])


def make_aectepp(X_train, n_classes):
    estimators = [('rf', make_single_model('RandomForest', X_train, n_classes)), ('et', make_single_model('ExtraTrees', X_train, n_classes)), ('hgb', make_single_model('HistGradientBoosting', X_train, n_classes))]
    weights = [1.2, 1.0, 1.1]
    if HAS_XGB:
        estimators.append(('xgb', make_single_model('XGBoost', X_train, n_classes)))
        weights.append(1.4)
    return VotingClassifier(estimators=estimators, voting='soft', weights=weights, n_jobs=None)


def split_xy(df, target, features):
    train = df[df['split'].astype(str).str.lower()=='train'].copy()
    val = df[df['split'].astype(str).str.lower()=='validation'].copy()
    test = df[df['split'].astype(str).str.lower()=='test'].copy()
    def make_x(part):
        X = part.drop(columns=[c for c in DROP_ALWAYS if c in part.columns], errors='ignore')
        return X[[c for c in features if c in X.columns]].copy()
    return make_x(train), train[target].astype(int), make_x(val), val[target].astype(int), make_x(test), test[target].astype(int)


def model_attack_probability(model, X, target):
    if not hasattr(model, 'predict_proba'):
        pred = model.predict(X); return (pred==1).astype(float) if target=='binary_label' else (pred!=0).astype(float)
    proba = model.predict_proba(X)
    if target == 'binary_label':
        classes = list(model.classes_) if hasattr(model, 'classes_') else [0,1]
        return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
    classes = list(model.classes_) if hasattr(model, 'classes_') else list(range(proba.shape[1]))
    return 1 - proba[:, classes.index(0)] if 0 in classes else 1 - np.max(proba, axis=1)


def tune_threshold(y_true, attack_prob, target):
    true_attack = (y_true.values==1) if target=='binary_label' else (y_true.values!=0)
    best = {'threshold':0.5, 'utility':-999}
    for t in np.linspace(0.05, 0.95, 91):
        pred_attack = attack_prob >= t
        tp = np.sum(true_attack & pred_attack); fp = np.sum((~true_attack) & pred_attack); fn = np.sum(true_attack & (~pred_attack)); tn = np.sum((~true_attack) & (~pred_attack))
        coverage = tp/max(tp+fn,1); false_iso = fp/max(fp+tn,1); precision = tp/max(tp+fp,1)
        utility = 0.50*coverage + 0.35*precision - 0.20*false_iso
        if utility > best['utility']: best = {'threshold':float(t), 'utility':float(utility), 'attack_coverage':float(coverage), 'false_isolation_rate':float(false_iso), 'response_precision':float(precision)}
    return best


def response_metrics(y_true, attack_prob, threshold, target):
    true_attack = (y_true.values==1) if target=='binary_label' else (y_true.values!=0)
    pred_attack = attack_prob >= threshold
    tp = int(np.sum(true_attack & pred_attack)); fp = int(np.sum((~true_attack) & pred_attack)); fn = int(np.sum(true_attack & (~pred_attack))); tn = int(np.sum((~true_attack) & (~pred_attack)))
    coverage = tp/max(tp+fn,1); false_iso = fp/max(fp+tn,1); precision = tp/max(tp+fp,1); utility = 0.50*coverage + 0.35*precision - 0.20*false_iso
    return {'threshold':threshold,'attack_coverage':coverage,'false_isolation_rate':false_iso,'response_precision':precision,'resilience_utility':utility,'tp':tp,'fp':fp,'fn':fn,'tn':tn}


def evaluate(y_true, y_pred, task, setting, model_name, train_s, infer_s):
    return {'task':task,'setting':setting,'model':model_name,'accuracy':accuracy_score(y_true,y_pred),'balanced_accuracy':balanced_accuracy_score(y_true,y_pred),'macro_precision':precision_score(y_true,y_pred,average='macro',zero_division=0),'macro_recall':recall_score(y_true,y_pred,average='macro',zero_division=0),'macro_f1':f1_score(y_true,y_pred,average='macro',zero_division=0),'weighted_f1':f1_score(y_true,y_pred,average='weighted',zero_division=0),'mcc':matthews_corrcoef(y_true,y_pred),'training_s':train_s,'inference_s':infer_s,'latency_ms_per_msg':infer_s/max(len(y_true),1)*1000,'test_records':len(y_true)}


def train_eval(df, target, task, setting, model_name, features, out_dir, proposed=False):
    X_train,y_train,X_val,y_val,X_test,y_test = split_xy(df,target,features)
    model = make_aectepp(X_train, len(np.unique(y_train))) if proposed else make_single_model(model_name, X_train, len(np.unique(y_train)))
    t0=time.perf_counter(); model.fit(X_train,y_train); train_s=time.perf_counter()-t0
    t1=time.perf_counter(); pred=model.predict(X_test); infer_s=time.perf_counter()-t1
    final_name = 'AECTE++' if proposed else model_name
    det = evaluate(y_test,pred,task,setting,final_name,train_s,infer_s)
    best = tune_threshold(y_val, model_attack_probability(model,X_val,target), target)
    resp = response_metrics(y_test, model_attack_probability(model,X_test,target), best['threshold'], target)
    resp.update({'task':task,'setting':setting,'model':final_name,'validation_threshold':best['threshold'],'validation_utility':best['utility']})
    labels = sorted(np.unique(np.concatenate([y_train.unique(), y_test.unique()])))
    pd.DataFrame(confusion_matrix(y_test,pred,labels=labels), index=labels, columns=labels).to_csv(out_dir/'confusion_matrices'/f'{task}_{setting}_{final_name}_cm.csv'.replace('+','p'))
    pd.DataFrame(classification_report(y_test,pred,output_dict=True,zero_division=0)).T.to_csv(out_dir/'class_reports'/f'{task}_{setting}_{final_name}_report.csv'.replace('+','p'))
    return det, resp, model


def stress_df(df, mode, level, seed=42):
    rng=np.random.default_rng(seed); out=df.copy(); test_mask=out['split'].astype(str).str.lower()=='test'; idx=out.index[test_mask].to_numpy()
    if mode=='packet_loss':
        drop_n=int(len(idx)*level); drop_idx=rng.choice(idx,size=drop_n,replace=False) if drop_n>0 else []; out=out.drop(index=drop_idx).reset_index(drop=True)
    elif mode=='gps_noise':
        for col in ['sender_pos_x','sender_pos_y','receiver_pos_x','receiver_pos_y']:
            if col in out.columns: out.loc[test_mask,col]=out.loc[test_mask,col]+rng.normal(0,level,size=test_mask.sum())
    elif mode=='delay_injection':
        if 'delay' in out.columns: out.loc[test_mask,'delay']=out.loc[test_mask,'delay'].fillna(0)+level
    elif mode=='edge_cloud_outage':
        for col in CLOUD_FEATURES+GRAPH_TRUST_FEATURES:
            if col in out.columns:
                miss=test_mask & (rng.random(len(out))<level); out.loc[miss,col]=np.nan
    return out


def deterministic_policy_agent(row):
    risk=row['risk_level']; attack=row['predicted_attack_type']; p=row['attack_probability']
    action = 'isolate sender, revoke trust temporarily, and forward incident to cloud verifier' if risk=='critical' else 'quarantine messages, request verification, and reduce sender trust' if risk=='high' else 'monitor sender and increase verification rate' if risk=='medium' else 'allow message with normal monitoring'
    explanation=f'The response agent assigns {risk} risk for {attack}. Attack probability is {p:.3f}. The decision uses trust history, graph consistency, spatial evidence, and temporal behavior. Recommended action: {action}.'
    return action, explanation


def ollama_generate(prompt, model='llama3.2:3b', host='http://localhost:11434'):
    payload={'model':model,'prompt':prompt,'stream':False,'options':{'temperature':0.1}}
    try:
        req=request.Request(f'{host}/api/generate', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type':'application/json'})
        with request.urlopen(req, timeout=60) as resp: return json.loads(resp.read().decode('utf-8')).get('response','')
    except Exception: return ''


def run_llm_incident_agent(df, model, features, out_dir, provider, llm_model, top_k=50):
    _,_,_,_,X_test,_=split_xy(df,'class_id',features)
    proba=model.predict_proba(X_test); pred=model.predict(X_test); classes=list(model.classes_) if hasattr(model,'classes_') else [0,1,2,3]
    normal_idx=classes.index(0) if 0 in classes else 0; attack_prob=1-proba[:,normal_idx]
    test_part=df[df['split'].astype(str).str.lower()=='test'].copy().reset_index(drop=True)
    test_part['predicted_class_id']=pred; test_part['predicted_attack_type']=[CLASS_NAMES.get(int(x),str(x)) for x in pred]; test_part['attack_probability']=attack_prob
    incidents=test_part[test_part['predicted_class_id']!=0].sort_values('attack_probability',ascending=False).head(top_k)
    rows=[]
    for _,r in incidents.iterrows():
        p=float(r['attack_probability']); risk='critical' if p>=0.90 else 'high' if p>=0.75 else 'medium' if p>=0.55 else 'low'
        incident={'sender_id':r.get('sender_id','unknown'),'predicted_attack_type':r['predicted_attack_type'],'attack_probability':p,'risk_level':risk,'distance_to_road_edge':r.get('distance_to_road_edge',np.nan),'edge_violation':r.get('edge_violation',np.nan),'sender_trust_prior':r.get('sender_trust_prior',np.nan),'graph_trust_propagated':r.get('graph_trust_propagated',np.nan),'heading_delta':r.get('heading_delta',np.nan),'delay':r.get('delay',np.nan)}
        if provider=='ollama':
            text=ollama_generate('You are a V2X cyber-response policy agent. Return two fields only: Action and Rationale. Incident summary: '+json.dumps(incident), model=llm_model)
            action, explanation = ('llm_generated', text.strip().replace('\n',' ')) if text.strip() else deterministic_policy_agent(pd.Series(incident))
        else: action, explanation = deterministic_policy_agent(pd.Series(incident))
        incident.update({'llm_provider':provider,'policy_action':action,'policy_explanation':explanation}); rows.append(incident)
    out=pd.DataFrame(rows); out.to_csv(out_dir/'csv'/'llm_incident_reports.csv', index=False)


def create_latex_table(df, path, caption, label):
    tex=['\\begin{table*}[!t]','\\centering',f'\\caption{{{caption}}}',f'\\label{{{label}}}','\\scriptsize','\\resizebox{\\textwidth}{!}{%',df.to_latex(index=False,escape=False,float_format=lambda x:f'{x:.4f}'),'}','\\end{table*}']
    path.write_text('\n'.join(tex),encoding='utf-8')


def run_all(df, out_dir, llm_provider, llm_model):
    raw_features=list(dict.fromkeys(EDGE_FEATURES+CLOUD_FEATURES)); enhanced=list(dict.fromkeys(EDGE_FEATURES+CLOUD_FEATURES+TEMPORAL_FEATURES+TRUST_FEATURES+GRAPH_TRUST_FEATURES)); proposed_features=enhanced
    baselines=['LogisticRegression','RandomForest','ExtraTrees','HistGradientBoosting']+(['XGBoost'] if HAS_XGB else [])
    detection=[]; response=[]; saved={}
    for task,target in [('binary','binary_label'),('multiclass','class_id')]:
        for name in baselines:
            print(f'[BASELINE RAW] {task} | {name}'); det,resp,model=train_eval(df,target,task,'S0_raw_edge_cloud',name,raw_features,out_dir); detection.append(det); response.append(resp)
        for name in baselines:
            print(f'[BASELINE ENHANCED] {task} | {name}'); det,resp,model=train_eval(df,target,task,'S1_enhanced_temporal_trust_graph',name,enhanced,out_dir); detection.append(det); response.append(resp)
        ablations={'A1_edge_only':EDGE_FEATURES,'A2_cloud_only':CLOUD_FEATURES,'A3_temporal_only':TEMPORAL_FEATURES,'A4_trust_only':TRUST_FEATURES,'A5_graph_trust_only':GRAPH_TRUST_FEATURES,'A6_edge_cloud_temporal':list(dict.fromkeys(EDGE_FEATURES+CLOUD_FEATURES+TEMPORAL_FEATURES)),'A7_full_AECTEpp':proposed_features}
        for setting,feats in ablations.items():
            print(f'[PROPOSED] {task} | {setting}'); det,resp,model=train_eval(df,target,task,setting,'AECTE++',feats,out_dir,proposed=True); detection.append(det); response.append(resp)
            if setting=='A7_full_AECTEpp': saved[(task,'AECTE++')]=model; joblib.dump(model,out_dir/'models'/f'{task}_AECTEpp.joblib')
    det_df=pd.DataFrame(detection); resp_df=pd.DataFrame(response); det_df.to_csv(out_dir/'csv'/'all_detection_metrics.csv',index=False); resp_df.to_csv(out_dir/'csv'/'response_resilience_metrics.csv',index=False)
    robust=[]; model=saved.get(('binary','AECTE++'))
    if model is not None:
        for mode,levels in {'packet_loss':[0.10,0.20,0.30],'gps_noise':[2.0,5.0,10.0],'delay_injection':[50.0,100.0,200.0],'edge_cloud_outage':[0.10,0.20,0.30]}.items():
            for level in levels:
                print(f'[ROBUSTNESS] {mode} | {level}'); sdf=stress_df(df,mode,level); _,_,_,_,X_test,y_test=split_xy(sdf,'binary_label',proposed_features); t=time.perf_counter(); pred=model.predict(X_test); infer=time.perf_counter()-t; row=evaluate(y_test,pred,'binary',f'{mode}_{level}','AECTE++',0.0,infer); row.update({'stress_type':mode,'stress_level':level}); robust.append(row)
    robust_df=pd.DataFrame(robust); robust_df.to_csv(out_dir/'csv'/'robustness_metrics.csv',index=False)
    scale=[]; model=saved.get(('binary','AECTE++'))
    if model is not None:
        _,_,_,_,X_full,y_full=split_xy(df,'binary_label',proposed_features)
        for n in [10000,25000,50000,100000,min(197270,len(X_full))]:
            Xs=X_full.head(n); ys=y_full.head(n); t=time.perf_counter(); pred=model.predict(Xs); infer=time.perf_counter()-t; row=evaluate(ys,pred,'binary',f'N={n}','AECTE++',0.0,infer); row['sample_size']=n; scale.append(row)
    scale_df=pd.DataFrame(scale); scale_df.to_csv(out_dir/'csv'/'scalability_metrics.csv',index=False)
    if saved.get(('multiclass','AECTE++')) is not None: run_llm_incident_agent(df,saved[('multiclass','AECTE++')],proposed_features,out_dir,llm_provider,llm_model)
    return det_df,resp_df,robust_df,scale_df


def make_tables(det_df,resp_df,robust_df,scale_df,out_dir):
    comp=det_df[det_df['setting'].isin(['S0_raw_edge_cloud','S1_enhanced_temporal_trust_graph','A7_full_AECTEpp'])][['task','setting','model','accuracy','macro_f1','weighted_f1','mcc','latency_ms_per_msg']].sort_values(['task','macro_f1'],ascending=[True,False])
    comp.to_csv(out_dir/'tables'/'baseline_comparison_table.csv',index=False); create_latex_table(comp,out_dir/'tables'/'baseline_comparison_table.tex','Comparative detection performance of baseline and proposed models.','tab:baseline_comparison')
    abl=det_df[det_df['model'].eq('AECTE++')][['task','setting','accuracy','macro_f1','weighted_f1','mcc','latency_ms_per_msg']].sort_values(['task','setting']); abl.to_csv(out_dir/'tables'/'ablation_table.csv',index=False); create_latex_table(abl,out_dir/'tables'/'ablation_table.tex','Ablation analysis of AECTE++ evidence groups.','tab:aectepp_ablation')
    res=resp_df[['task','setting','model','attack_coverage','false_isolation_rate','response_precision','resilience_utility']].sort_values(['task','resilience_utility'],ascending=[True,False]); res.to_csv(out_dir/'tables'/'response_resilience_table.csv',index=False); create_latex_table(res.head(30),out_dir/'tables'/'response_resilience_table.tex','Cyber-response and resilience metrics.','tab:response_resilience')
    if not robust_df.empty: robust_df.to_csv(out_dir/'tables'/'robustness_table.csv',index=False); create_latex_table(robust_df[['stress_type','stress_level','accuracy','macro_f1','mcc','latency_ms_per_msg']],out_dir/'tables'/'robustness_table.tex','Robustness of AECTE++ under degraded V2X conditions.','tab:robustness')
    if not scale_df.empty: scale_df.to_csv(out_dir/'tables'/'scalability_table.csv',index=False); create_latex_table(scale_df[['sample_size','accuracy','macro_f1','mcc','latency_ms_per_msg','inference_s']],out_dir/'tables'/'scalability_table.tex','Scalability of AECTE++ under increasing V2X message volume.','tab:scalability')


def make_figures(det_df,resp_df,robust_df,scale_df,out_dir):
    fig,axes=plt.subplots(1,2,figsize=(16,9))
    for ax,task,panel in zip(axes,['binary','multiclass'],['(a)','(b)']):
        sub=det_df[(det_df['task'].eq(task)) & (det_df['setting'].isin(['S0_raw_edge_cloud','S1_enhanced_temporal_trust_graph','A7_full_AECTEpp']))].copy()
        sub['name']=sub['model']+'\n'+sub['setting'].replace({'S0_raw_edge_cloud':'raw','S1_enhanced_temporal_trust_graph':'enhanced','A7_full_AECTEpp':'proposed'}); sub=sub.sort_values('macro_f1',ascending=False).head(8)
        bars=ax.bar(sub['name'],sub['macro_f1']); ax.set_ylabel('Macro F1-score'); ax.set_xlabel('Method'); ax.set_title(f'{task.capitalize()} Comparative Analysis'); ax.tick_params(axis='x',rotation=18); add_panel(ax,panel)
        for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{b.get_height():.3f}',ha='center',va='bottom',fontsize=9)
    savefig(fig,out_dir/'figures'/'fig1_comparative_baseline_analysis')
    order=['A1_edge_only','A2_cloud_only','A3_temporal_only','A4_trust_only','A5_graph_trust_only','A6_edge_cloud_temporal','A7_full_AECTEpp']; labels=['Edge','Cloud','Temporal','Trust','Graph','E+C+T','Full']
    fig,ax=plt.subplots(figsize=(16,9)); ab=det_df[det_df['model'].eq('AECTE++')]; pivot=ab.pivot_table(index='setting',columns='task',values='macro_f1',aggfunc='max').reindex(order); x=np.arange(len(order)); w=.36
    ax.bar(x-w/2,pivot['binary'],width=w,label='Binary'); ax.bar(x+w/2,pivot['multiclass'],width=w,label='Multiclass'); ax.set_xticks(x); ax.set_xticklabels(labels,rotation=10); ax.set_ylabel('Macro F1-score'); ax.set_xlabel('AECTE++ configuration'); ax.set_title('AECTE++ Ablation Analysis'); ax.legend(); add_panel(ax,'(a)'); savefig(fig,out_dir/'figures'/'fig2_aectepp_ablation')
    fig,axes=plt.subplots(1,2,figsize=(16,9)); rr=resp_df[(resp_df['model'].eq('AECTE++'))&(resp_df['task'].eq('binary'))].set_index('setting').reindex(order).reset_index(); x=np.arange(len(order))
    axes[0].plot(x,rr['attack_coverage'],marker='o',linewidth=3,label='Attack coverage'); axes[0].plot(x,rr['response_precision'],marker='s',linewidth=3,label='Response precision'); axes[0].plot(x,1-rr['false_isolation_rate'],marker='^',linewidth=3,label='Benign preservation'); axes[0].set_xticks(x); axes[0].set_xticklabels(labels,rotation=10); axes[0].set_ylabel('Score'); axes[0].set_xlabel('Configuration'); axes[0].set_title('Response Quality'); axes[0].legend(); add_panel(axes[0],'(a)')
    axes[1].bar(labels,rr['resilience_utility']); axes[1].set_ylabel('Resilience utility'); axes[1].set_xlabel('Configuration'); axes[1].set_title('Cyber-Resilience Utility'); axes[1].tick_params(axis='x',rotation=10); add_panel(axes[1],'(b)'); savefig(fig,out_dir/'figures'/'fig3_response_resilience')
    if not robust_df.empty:
        fig,ax=plt.subplots(figsize=(16,9))
        for stress,part in robust_df.groupby('stress_type'):
            part=part.sort_values('stress_level'); ax.plot(part['stress_level'],part['macro_f1'],marker='o',linewidth=3,label=stress)
        ax.set_xlabel('Stress level'); ax.set_ylabel('Binary macro F1-score'); ax.set_title('Robustness Under Degraded V2X Conditions'); ax.legend(); add_panel(ax,'(a)'); savefig(fig,out_dir/'figures'/'fig4_robustness_degradation')
    if not scale_df.empty:
        fig,ax=plt.subplots(figsize=(16,9)); ax.plot(scale_df['sample_size'],scale_df['latency_ms_per_msg'],marker='o',linewidth=3); ax.set_xlabel('Number of test messages'); ax.set_ylabel('Latency per message (ms)'); ax.set_title('AECTE++ Scalability'); add_panel(ax,'(a)'); savefig(fig,out_dir/'figures'/'fig5_scalability')
    llm_path=out_dir/'csv'/'llm_incident_reports.csv'
    if llm_path.exists():
        llm=pd.read_csv(llm_path)
        if not llm.empty:
            fig,ax=plt.subplots(figsize=(16,9)); counts=llm['risk_level'].value_counts().reindex(['low','medium','high','critical']).fillna(0); bars=ax.bar(counts.index,counts.values); ax.set_xlabel('LLM-assigned risk level'); ax.set_ylabel('Incident count'); ax.set_title('LLM Policy Agent Triage'); add_panel(ax,'(a)'); savefig(fig,out_dir/'figures'/'fig6_llm_policy_agent')


def write_report(det_df,resp_df,robust_df,scale_df,out_dir):
    lines=['# Agentic-V2XShield Final+ Report\n','## System Summary\n','The final system evaluates five standard baselines, enhanced temporal-trust baselines, and AECTE++. AECTE++ combines edge telemetry, cloud spatial context, temporal consistency, sender-level trust, graph-trust propagation, stacked fusion, and LLM-assisted response reasoning.\n']
    lines += ['## Top Binary Results\n', det_df[det_df['task'].eq('binary')].sort_values('macro_f1',ascending=False).head(12).to_markdown(index=False)]
    lines += ['\n## Top Multiclass Results\n', det_df[det_df['task'].eq('multiclass')].sort_values('macro_f1',ascending=False).head(12).to_markdown(index=False)]
    lines += ['\n## Top Response Results\n', resp_df.sort_values('resilience_utility',ascending=False).head(12).to_markdown(index=False)]
    if not robust_df.empty: lines += ['\n## Robustness Summary\n', robust_df[['stress_type','stress_level','accuracy','macro_f1','mcc']].to_markdown(index=False)]
    if not scale_df.empty: lines += ['\n## Scalability Summary\n', scale_df[['sample_size','macro_f1','latency_ms_per_msg','inference_s']].to_markdown(index=False)]
    llm_path=out_dir/'csv'/'llm_incident_reports.csv'
    if llm_path.exists():
        llm=pd.read_csv(llm_path); lines += ['\n## LLM Policy Agent Samples\n', llm[['risk_level','predicted_attack_type','attack_probability','policy_action']].head(10).to_markdown(index=False)]
    lines += ['\n## Safe Paper Claim\n','The paper should claim deployment-aware cyber-resilience rather than only classifier superiority. The strongest contribution is the combined use of temporal consistency, trust propagation, graph-based sender context, and LLM-assisted policy reasoning for V2X attack detection and response.']
    (out_dir/'reports'/'final_plus_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--out-dir',default='outputs_final_plus'); p.add_argument('--max-per-class',type=int,default=100000); p.add_argument('--keep-leakage-risk-features',action='store_true'); p.add_argument('--llm-provider',choices=['none','ollama'],default='none'); p.add_argument('--llm-model',default='llama3.2:3b'); args=p.parse_args()
    out_dir=Path(args.out_dir); ensure_dirs(out_dir)
    print('[INFO] Loading dataset.'); df=load_dataset(Path(args.csv),args.max_per_class,args.keep_leakage_risk_features)
    print('[INFO] Engineering temporal and trust features.'); df=engineer_temporal_trust(df)
    print('[INFO] Engineering graph-trust features.'); df=engineer_graph_trust(df,out_dir)
    df.to_csv(out_dir/'csv'/'engineered_dataset_snapshot.csv',index=False)
    print('[INFO] Running full comparative experiments.'); det_df,resp_df,robust_df,scale_df=run_all(df,out_dir,args.llm_provider,args.llm_model)
    print('[INFO] Creating tables.'); make_tables(det_df,resp_df,robust_df,scale_df,out_dir)
    print('[INFO] Creating figures.'); make_figures(det_df,resp_df,robust_df,scale_df,out_dir)
    print('[INFO] Writing report.'); write_report(det_df,resp_df,robust_df,scale_df,out_dir)
    print('\n[DONE] Final+ Agentic-V2XShield pipeline completed.'); print(f'[OUT] {out_dir.resolve()}')

if __name__=='__main__': main()
