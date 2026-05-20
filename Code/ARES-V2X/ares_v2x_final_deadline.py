
#!/usr/bin/env python
"""
ARES-V2X Final Deadline Code
Self-healing agentic edge-cloud cyber-resilience for V2X security.

Run:
python ares_v2x_final_deadline.py --csv "..\\outputs_multiclass\\veremi_multiclass_balanced.csv" --out-dir "..\\outputs_ares_v2x_final" --max-per-class 100000 --llm-provider ollama --llm-model llama3.2:3b
"""
import argparse, json, time, warnings
from pathlib import Path
from urllib import request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
warnings.filterwarnings('ignore')
try:
    from xgboost import XGBClassifier
    HAS_XGB=True
except Exception:
    HAS_XGB=False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM=True
except Exception:
    HAS_LGBM=False
try:
    from catboost import CatBoostClassifier
    HAS_CAT=True
except Exception:
    HAS_CAT=False

plt.rcParams.update({'figure.figsize':(16,9),'figure.dpi':600,'savefig.dpi':600,'font.size':13,'axes.labelsize':15,'axes.titlesize':17,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10,'font.family':'DejaVu Serif','axes.grid':True,'grid.alpha':0.18,'grid.linestyle':'--'})

EDGE=['delay','sender_spd','sender_acl','sender_hed','receiver_spd','receiver_acl','receiver_hed','speed_delta','accel_delta','heading_delta','abs_sender_speed','abs_sender_accel']
CLOUD=['sender_pos_x','sender_pos_y','receiver_pos_x','receiver_pos_y','sender_receiver_distance','distance_to_road_edge','edge_violation','sender_pos_noise_x','sender_pos_noise_y','receiver_pos_noise_x','receiver_pos_noise_y','sender_spd_noise','receiver_spd_noise','sender_acl_noise','receiver_acl_noise','sender_hed_noise','receiver_hed_noise','sender_driver_profile','receiver_driver_profile']
TEMP=['sender_spd_roll_mean_5','sender_acl_roll_mean_5','heading_delta_roll_mean_5','speed_delta_roll_mean_5','delay_roll_mean_5','sender_spd_diff','sender_acl_diff','sender_hed_diff','sender_pos_step_dist','msg_time_gap','temporal_instability']
TRUST=['sender_msg_count_so_far','sender_attack_rate_prior','sender_edge_violation_rate_prior','sender_delay_mean_prior','sender_road_edge_mean_prior','sender_trust_prior','trust_decay','risk_rule_score']
GRAPH=['graph_sender_degree_prior','graph_neighbor_risk_prior','graph_trust_propagated','graph_local_disagreement','gnn_neighbor_speed_mean','gnn_neighbor_accel_mean','gnn_neighbor_heading_delta_mean','gnn_neighbor_rule_mean','gnn_neighbor_trust_mean','gnn_neighbor_degree_mean','gnn_feature_disagreement','gnn_sender_degree_log']
ALL=list(dict.fromkeys(EDGE+CLOUD+TEMP+TRUST+GRAPH)); RAW=list(dict.fromkeys(EDGE+CLOUD)); GRAPH_BASE=list(dict.fromkeys(EDGE+CLOUD+GRAPH))
DROP=['source_file','attacker_raw','class_name','class_id','binary_label','split']; LEAK=['messageID','sender_alias','rcvTime','sendTime']
CLASS={0:'normal',1:'constantPositionOffset',2:'randomPositionOffset',3:'trafficCongestionSybil'}

def ensure_dirs(out):
    for s in ['csv','tables','figures','reports','models','confusion_matrices','class_reports','llm']:
        (out/s).mkdir(parents=True, exist_ok=True)

def savefig(fig,p):
    fig.tight_layout(); fig.savefig(p.with_suffix('.png'),dpi=600,bbox_inches='tight'); fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight'); plt.close(fig)

def panel(ax,label):
    ax.text(.01,.98,label,transform=ax.transAxes,ha='left',va='top',fontsize=18,fontweight='bold',bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=2))

def latex(df,p,cap,lab):
    p.write_text('\n'.join(['\\begin{table*}[!t]','\\centering',f'\\caption{{{cap}}}',f'\\label{{{lab}}}','\\scriptsize','\\resizebox{\\textwidth}{!}{%',df.to_latex(index=False,escape=False,float_format=lambda x:f'{x:.4f}'),'}','\\end{table*}']),encoding='utf-8')

def mm(s):
    s=pd.to_numeric(s,errors='coerce'); a=s.min(skipna=True); b=s.max(skipna=True)
    if pd.isna(a) or pd.isna(b) or abs(b-a)<1e-12: return pd.Series(np.zeros(len(s)),index=s.index)
    return (s-a)/(b-a)

def hdiff(a,b):
    try:
        if pd.isna(a) or pd.isna(b): return np.nan
        d=abs(float(a)-float(b))%360; return min(d,360-d)
    except Exception: return np.nan

def load_data(path,max_per_class=0,keep_leak=False):
    df=pd.read_csv(path)
    miss=[c for c in ['class_id','binary_label','split'] if c not in df.columns]
    if miss: raise ValueError(f'Missing required columns: {miss}')
    if max_per_class and max_per_class>0:
        df=df.groupby('class_id',group_keys=False).apply(lambda x:x.sample(min(len(x),max_per_class),random_state=42)).reset_index(drop=True)
    if not keep_leak: df=df.drop(columns=[c for c in LEAK if c in df.columns],errors='ignore')
    if 'sender_id' not in df.columns: df['sender_id']='unknown_sender'
    return df

def add_temporal_trust(df):
    df=df.copy()
    for c in ['sender_pos_x','sender_pos_y','sender_spd','sender_acl','sender_hed','delay','heading_delta','speed_delta','distance_to_road_edge','edge_violation','sender_receiver_distance']:
        if c not in df.columns: df[c]=np.nan
    sort=['split','sender_id']+([c for c in ['sendTime','messageID'] if c in df.columns][:1])
    df=df.sort_values(sort).reset_index(drop=True); g=df.groupby(['split','sender_id'],sort=False)
    df['prev_x']=g.sender_pos_x.shift(1); df['prev_y']=g.sender_pos_y.shift(1)
    df['sender_pos_step_dist']=np.sqrt((df.sender_pos_x-df.prev_x)**2+(df.sender_pos_y-df.prev_y)**2)
    df['msg_time_gap']=g.sendTime.diff() if 'sendTime' in df.columns else np.nan
    df['sender_spd_diff']=g.sender_spd.diff(); df['sender_acl_diff']=g.sender_acl.diff(); df['prev_h']=g.sender_hed.shift(1)
    df['sender_hed_diff']=[hdiff(a,b) for a,b in zip(df.sender_hed,df.prev_h)]
    for src,dst in {'sender_spd':'sender_spd_roll_mean_5','sender_acl':'sender_acl_roll_mean_5','heading_delta':'heading_delta_roll_mean_5','speed_delta':'speed_delta_roll_mean_5','delay':'delay_roll_mean_5'}.items():
        df[dst]=g[src].rolling(5,min_periods=1).mean().reset_index(level=[0,1],drop=True)
    df['sender_msg_count_so_far']=g.cumcount(); cnt=df.sender_msg_count_so_far.replace(0,np.nan)
    df['sender_attack_rate_prior']=(g.binary_label.cumsum()-df.binary_label)/cnt
    df['sender_edge_violation_rate_prior']=(g.edge_violation.cumsum()-df.edge_violation.fillna(0))/cnt
    df['sender_delay_mean_prior']=(g.delay.cumsum()-df.delay.fillna(0))/cnt
    df['sender_road_edge_mean_prior']=(g.distance_to_road_edge.cumsum()-df.distance_to_road_edge.fillna(0))/cnt
    df['sender_trust_prior']=1-df.sender_attack_rate_prior; df['trust_decay']=1-df.sender_trust_prior
    df['risk_rule_score']=0.0
    df['risk_rule_score']+=(df.edge_violation.fillna(0)>0).astype(float)*.30
    df['risk_rule_score']+=(df.heading_delta.fillna(0)>90).astype(float)*.20
    df['risk_rule_score']+=(df.speed_delta.fillna(0)>df.speed_delta.median(skipna=True)).astype(float)*.15
    df['risk_rule_score']+=(df.delay.fillna(0)>df.delay.quantile(.95)).astype(float)*.15
    df['risk_rule_score']+=(df.sender_receiver_distance.fillna(0)>df.sender_receiver_distance.quantile(.95)).astype(float)*.20
    df['temporal_instability']=.25*mm(df.sender_spd_diff.abs())+.25*mm(df.sender_acl_diff.abs())+.25*mm(df.sender_hed_diff.abs())+.25*mm(df.sender_pos_step_dist.abs())
    return df.drop(columns=['prev_x','prev_y','prev_h'],errors='ignore')

def add_graph(df,out):
    df=df.copy()
    if 'receiver_id' not in df.columns:
        df['receiver_id']='profile_'+df.receiver_driver_profile.astype(str) if 'receiver_driver_profile' in df.columns else 'unknown_receiver'
    rows=[]
    for sp,part in df.groupby('split',sort=False):
        st=part.groupby('sender_id').agg(sender_binary_rate=('binary_label','mean'),sender_count=('binary_label','size'),sender_rule_mean=('risk_rule_score','mean'),sender_speed_mean=('sender_spd','mean'),sender_accel_mean=('sender_acl','mean'),sender_heading_delta_mean=('heading_delta','mean'),sender_trust_mean=('sender_trust_prior','mean')).reset_index()
        e=part.groupby(['sender_id','receiver_id']).size().reset_index(name='edge_weight')
        n=e.merge(st.rename(columns={'sender_id':'receiver_id','sender_binary_rate':'nb_risk','sender_rule_mean':'nb_rule','sender_speed_mean':'nb_speed','sender_accel_mean':'nb_accel','sender_heading_delta_mean':'nb_head','sender_trust_mean':'nb_trust','sender_count':'nb_count'}),on='receiver_id',how='left').fillna(0)
        for c in ['nb_risk','nb_rule','nb_speed','nb_accel','nb_head','nb_trust','nb_count']: n['w_'+c]=n[c]*n.edge_weight
        ag=n.groupby('sender_id').agg(graph_sender_degree_prior=('receiver_id','nunique'),graph_neighbor_risk_prior=('w_nb_risk','sum'),gnn_neighbor_rule_mean=('w_nb_rule','sum'),gnn_neighbor_speed_mean=('w_nb_speed','sum'),gnn_neighbor_accel_mean=('w_nb_accel','sum'),gnn_neighbor_heading_delta_mean=('w_nb_head','sum'),gnn_neighbor_trust_mean=('w_nb_trust','sum'),gnn_neighbor_degree_mean=('w_nb_count','sum'),w=('edge_weight','sum')).reset_index()
        for c in ['graph_neighbor_risk_prior','gnn_neighbor_rule_mean','gnn_neighbor_speed_mean','gnn_neighbor_accel_mean','gnn_neighbor_heading_delta_mean','gnn_neighbor_trust_mean','gnn_neighbor_degree_mean']: ag[c]=ag[c]/ag.w.replace(0,np.nan)
        st=st.merge(ag,on='sender_id',how='left')
        st['graph_trust_propagated']=(1-.55*st.sender_binary_rate.fillna(0)-.25*st.graph_neighbor_risk_prior.fillna(0)-.20*st.sender_rule_mean.fillna(0)).clip(0,1)
        st['graph_local_disagreement']=(st.sender_binary_rate.fillna(0)-st.graph_neighbor_risk_prior.fillna(0)).abs()
        st['gnn_feature_disagreement']=(st.sender_speed_mean.fillna(0)-st.gnn_neighbor_speed_mean.fillna(0)).abs()+(st.sender_accel_mean.fillna(0)-st.gnn_neighbor_accel_mean.fillna(0)).abs()+(st.sender_heading_delta_mean.fillna(0)-st.gnn_neighbor_heading_delta_mean.fillna(0)).abs()
        st['gnn_sender_degree_log']=np.log1p(st.graph_sender_degree_prior.fillna(0)); st['split']=sp; rows.append(st)
    gr=pd.concat(rows,ignore_index=True); gr.to_csv(out/'csv'/'graph_trust_summary.csv',index=False)
    keep=['split','sender_id']+GRAPH
    for c in keep:
        if c not in gr.columns: gr[c]=0.0
    return df.merge(gr[keep],on=['split','sender_id'],how='left')

def prep(X):
    cat=[c for c in X.columns if X[c].dtype=='object']; num=[c for c in X.columns if c not in cat]
    return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])

def make_model(name,X,nc):
    pr=prep(X)
    if name=='LogisticRegression': clf=LogisticRegression(max_iter=1000,class_weight='balanced',n_jobs=-1,random_state=42)
    elif name=='RandomForest': clf=RandomForestClassifier(n_estimators=320,min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=42)
    elif name=='ExtraTrees': clf=ExtraTreesClassifier(n_estimators=380,min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=42)
    elif name=='HistGradientBoosting': clf=HistGradientBoostingClassifier(max_iter=380,learning_rate=.04,max_leaf_nodes=63,l2_regularization=.035,random_state=42)
    elif name=='MLP': clf=MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',alpha=1e-4,max_iter=80,random_state=42)
    elif name=='XGBoost':
        if not HAS_XGB: raise RuntimeError('xgboost not installed')
        clf=XGBClassifier(n_estimators=700,max_depth=8,learning_rate=.032,subsample=.92,colsample_bytree=.92,objective='multi:softprob' if nc>2 else 'binary:logistic',eval_metric='mlogloss' if nc>2 else 'logloss',tree_method='hist',random_state=42,n_jobs=-1)
    elif name=='LightGBM':
        if not HAS_LGBM: raise RuntimeError('lightgbm not installed')
        clf=LGBMClassifier(n_estimators=700,learning_rate=.032,num_leaves=70,subsample=.92,colsample_bytree=.92,random_state=42,n_jobs=-1,objective='multiclass' if nc>2 else 'binary',verbose=-1)
    elif name=='CatBoost':
        if not HAS_CAT: raise RuntimeError('catboost not installed')
        clf=CatBoostClassifier(iterations=600,depth=8,learning_rate=.035,loss_function='MultiClass' if nc>2 else 'Logloss',random_seed=42,verbose=False)
    else: raise ValueError(name)
    return Pipeline([('preprocess',pr),('clf',clf)])

def make_vote(X,nc):
    est=[('rf',make_model('RandomForest',X,nc)),('et',make_model('ExtraTrees',X,nc)),('hgb',make_model('HistGradientBoosting',X,nc))]; w=[1.2,1.0,1.1]
    if HAS_XGB: est.append(('xgb',make_model('XGBoost',X,nc))); w.append(1.4)
    if HAS_LGBM: est.append(('lgbm',make_model('LightGBM',X,nc))); w.append(1.35)
    return VotingClassifier(estimators=est,voting='soft',weights=w,n_jobs=None)

def split_xy(df,target,features,attack_only=False):
    tr=df[df.split.astype(str).str.lower()=='train'].copy(); va=df[df.split.astype(str).str.lower()=='validation'].copy(); te=df[df.split.astype(str).str.lower()=='test'].copy()
    if attack_only:
        tr=tr[tr.binary_label==1]; va=va[va.binary_label==1]; te=te[te.binary_label==1]
    def x(p):
        X=p.drop(columns=[c for c in DROP if c in p.columns],errors='ignore')
        return X[[c for c in features if c in X.columns]].copy()
    return x(tr),tr[target].astype(int),x(va),va[target].astype(int),x(te),te[target].astype(int)

def prob_attack(m,X,target):
    if not hasattr(m,'predict_proba'):
        pred=m.predict(X); return (pred==1).astype(float) if target=='binary_label' else (pred!=0).astype(float)
    p=m.predict_proba(X)
    if target=='binary_label':
        cls=list(m.classes_) if hasattr(m,'classes_') else [0,1]
        return p[:,cls.index(1)] if 1 in cls else p[:,-1]
    cls=list(m.classes_) if hasattr(m,'classes_') else list(range(p.shape[1]))
    return 1-p[:,cls.index(0)] if 0 in cls else 1-np.max(p,axis=1)

def agent_scores(X,ap):
    def c(n): return pd.to_numeric(X[n],errors='coerce').fillna(0) if n in X.columns else pd.Series(np.zeros(len(X)),index=X.index)
    edge=(.35*mm(c('speed_delta'))+.35*mm(c('accel_delta'))+.30*mm(c('heading_delta'))).clip(0,1)
    temporal=c('temporal_instability').clip(0,1)
    trust=(.60*mm(c('trust_decay'))+.40*mm(c('sender_attack_rate_prior'))).clip(0,1)
    graph=(.45*mm(c('graph_local_disagreement'))+.35*(1-mm(c('graph_trust_propagated')))+.20*mm(c('gnn_feature_disagreement'))).clip(0,1)
    cloud=(.35*mm(c('sender_receiver_distance'))+.30*mm(c('distance_to_road_edge').abs())+.20*mm(c('risk_rule_score'))+.15*pd.Series(ap,index=X.index).fillna(0)).clip(0,1)
    return pd.DataFrame({'edge_agent':edge,'temporal_agent':temporal,'trust_agent':trust,'graph_agent':graph,'cloud_agent':cloud,'ml_fusion_agent':pd.Series(ap,index=X.index).fillna(0).clip(0,1)},index=X.index).fillna(0)

def init_w(): return {'edge_agent':.12,'temporal_agent':.14,'trust_agent':.20,'graph_agent':.18,'cloud_agent':.14,'ml_fusion_agent':.22}
def norm_w(w):
    s=sum(max(v,1e-6) for v in w.values()); return {k:max(v,1e-6)/s for k,v in w.items()}
def consensus(scores,w,pen=.10):
    w=norm_w(w); cols=list(w); mat=scores[cols].to_numpy(float); vec=np.array([w[k] for k in cols]); mean=mat@vec; dis=mat.std(axis=1); return np.clip(mean-pen*dis,0,1),dis
def update_w(scores,y,w,lr=.08):
    y=y.astype(int).values; nw=dict(w)
    for a in scores.columns: nw[a]=(1-lr)*nw.get(a,1/len(scores.columns))+lr*(1-np.mean(np.abs(scores[a].values-y)))
    return norm_w(nw)
def tune_threshold(y,risk):
    y=y.values.astype(int); best={'threshold':.5,'utility':-999}
    for t in np.linspace(.05,.95,91):
        pred=risk>=t; tp=np.sum((y==1)&pred); fp=np.sum((y==0)&pred); fn=np.sum((y==1)&(~pred)); tn=np.sum((y==0)&(~pred))
        cov=tp/max(tp+fn,1); fi=fp/max(fp+tn,1); prec=tp/max(tp+fp,1); u=.45*cov+.35*prec-.30*fi
        if u>best['utility']: best={'threshold':float(t),'utility':float(u)}
    return best

def risk_metrics(y,risk,t):
    y=y.values.astype(int); pred=risk>=t; tp=int(np.sum((y==1)&pred)); fp=int(np.sum((y==0)&pred)); fn=int(np.sum((y==1)&(~pred))); tn=int(np.sum((y==0)&(~pred)))
    cov=tp/max(tp+fn,1); fi=fp/max(fp+tn,1); prec=tp/max(tp+fp,1); util=.45*cov+.35*prec-.30*fi
    return {'threshold':t,'attack_coverage':cov,'false_isolation_rate':fi,'response_precision':prec,'monitor_rate':float(np.mean((risk>=t*.65)&(risk<t))),'hard_isolation_rate':float(np.mean(risk>=t)),'resilience_utility':util,'mttd_proxy':1/max(cov,1e-6),'mrt_proxy':1/max(1-fi,1e-6),'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def eval_cls(y,pred,task,setting,model_name,train_s,infer_s):
    return {'task':task,'setting':setting,'model':model_name,'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_precision':precision_score(y,pred,average='macro',zero_division=0),'macro_recall':recall_score(y,pred,average='macro',zero_division=0),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),'weighted_f1':f1_score(y,pred,average='weighted',zero_division=0),'mcc':matthews_corrcoef(y,pred),'training_s':train_s,'inference_s':infer_s,'latency_ms_per_msg':infer_s/max(len(y),1)*1000,'test_records':len(y)}

def train_eval(df,target,task,setting,model_name,features,out,proposed=False,attack_only=False):
    Xtr,ytr,Xv,yv,Xte,yte=split_xy(df,target,features,attack_only=attack_only); nc=len(np.unique(ytr))
    m=make_vote(Xtr,nc) if proposed else make_model(model_name,Xtr,nc)
    t=time.perf_counter(); m.fit(Xtr,ytr); train_s=time.perf_counter()-t
    t=time.perf_counter(); pred=m.predict(Xte); infer_s=time.perf_counter()-t
    name='ARES-V2X-Vote' if proposed else model_name
    labels=sorted(np.unique(np.concatenate([ytr.unique(),yte.unique()])))
    pd.DataFrame(confusion_matrix(yte,pred,labels=labels),index=labels,columns=labels).to_csv(out/'confusion_matrices'/f'{task}_{setting}_{name}_cm.csv'.replace('+','p'))
    pd.DataFrame(classification_report(yte,pred,output_dict=True,zero_division=0)).T.to_csv(out/'class_reports'/f'{task}_{setting}_{name}_report.csv'.replace('+','p'))
    return eval_cls(yte,pred,task,setting,name,train_s,infer_s),m

class MetaFusion:
    def __init__(self, base_models, meta_model, classes):
        self.base_models=base_models; self.meta_model=meta_model; self.classes_=np.array(classes)
    def _features(self,X):
        parts=[]
        for name,m in self.base_models.items():
            p=m.predict_proba(X)
            parts.append(p)
        ap=1-parts[0][:,0] if parts[0].shape[1]>1 else parts[0][:,0]
        scores=agent_scores(X,ap)
        parts.append(scores.to_numpy())
        return np.hstack(parts)
    def fit_meta(self,Xv,yv):
        self.meta_model.fit(self._features(Xv),yv); return self
    def predict(self,X): return self.meta_model.predict(self._features(X))
    def predict_proba(self,X): return self.meta_model.predict_proba(self._features(X))

def train_meta_fusion(df,target,task,out):
    Xtr,ytr,Xv,yv,Xte,yte=split_xy(df,target,ALL); nc=len(np.unique(ytr))
    candidates=['RandomForest','ExtraTrees','HistGradientBoosting']
    if HAS_XGB: candidates.append('XGBoost')
    if HAS_LGBM: candidates.append('LightGBM')
    base={}
    t0=time.perf_counter()
    for name in candidates:
        print(f'[META BASE] {task} | {name}')
        m=make_model(name,Xtr,nc); m.fit(Xtr,ytr); base[name]=m
    meta=LogisticRegression(max_iter=1000,class_weight='balanced',n_jobs=-1,random_state=42)
    mf=MetaFusion(base,meta,sorted(np.unique(ytr))).fit_meta(Xv,yv)
    train_s=time.perf_counter()-t0
    t=time.perf_counter(); pred=mf.predict(Xte); infer_s=time.perf_counter()-t
    row=eval_cls(yte,pred,task,'ARES_meta_fusion','ARES-MetaFusion',train_s,infer_s)
    pd.DataFrame(confusion_matrix(yte,pred,labels=sorted(np.unique(yte))),index=sorted(np.unique(yte)),columns=sorted(np.unique(yte))).to_csv(out/'confusion_matrices'/f'{task}_ARES_MetaFusion_cm.csv')
    pd.DataFrame(classification_report(yte,pred,output_dict=True,zero_division=0)).T.to_csv(out/'class_reports'/f'{task}_ARES_MetaFusion_report.csv')
    joblib.dump(mf,out/'models'/f'{task}_ARES_MetaFusion.joblib')
    return row,mf

def hierarchical_meta(df,binary_m,out):
    # only attack classes train subtype classifier, then route test attacks predicted by binary meta
    Xtr,ytr,Xv,yv,Xte_attack,yte_attack=split_xy(df,'class_id',ALL,attack_only=True)
    expert=make_model('RandomForest',Xtr,len(np.unique(ytr))); expert.fit(pd.concat([Xtr,Xv]),pd.concat([ytr,yv]))
    _,_,_,_,Xte_all,yte_all=split_xy(df,'class_id',ALL)
    ap=prob_attack(binary_m,Xte_all,'binary_label'); bpred=(ap>=0.5).astype(int); final=np.zeros(len(yte_all),dtype=int)
    idx=np.where(bpred==1)[0]
    if len(idx): final[idx]=expert.predict(Xte_all.iloc[idx])
    row=eval_cls(yte_all,final,'multiclass','hierarchical_binary_meta_plus_attack_expert','ARES-Hierarchical-Meta',0.0,0.0)
    joblib.dump(expert,out/'models'/'attack_type_expert_RF.joblib')
    return row

def stress(df,mode,level,seed=42):
    rng=np.random.default_rng(seed); out=df.copy(); tm=out.split.astype(str).str.lower()=='test'; idx=out.index[tm].to_numpy()
    if mode=='packet_loss':
        n=int(len(idx)*level)
        if n: out=out.drop(index=rng.choice(idx,size=n,replace=False)).reset_index(drop=True)
    elif mode=='feature_dropout':
        for c in EDGE+CLOUD+TEMP:
            if c in out.columns: out.loc[tm & (rng.random(len(out))<level),c]=np.nan
    elif mode=='gps_corruption_m':
        for c in ['sender_pos_x','sender_pos_y','receiver_pos_x','receiver_pos_y']:
            if c in out.columns: out.loc[tm,c]=out.loc[tm,c]+rng.normal(0,level,size=tm.sum())
    elif mode=='delay_injection':
        for c in ['delay','delay_roll_mean_5','sender_delay_mean_prior']:
            if c in out.columns: out.loc[tm,c]=out.loc[tm,c].fillna(0)+level
    elif mode=='stale_trust':
        for c in TRUST+GRAPH:
            if c in out.columns:
                sh=out.loc[tm,c].sample(frac=1,random_state=seed).values; mask=tm & (rng.random(len(out))<level); out.loc[mask,c]=sh[:mask.sum()]
    elif mode=='edge_cloud_outage':
        for c in CLOUD+GRAPH:
            if c in out.columns: out.loc[tm & (rng.random(len(out))<level),c]=np.nan
    return out

def corrupt_scores(scores,mode,seed=42):
    rng=np.random.default_rng(seed); s=scores.copy()
    if mode=='none': return s
    if mode in s.columns: s[mode]=1-s[mode]
    elif mode=='trust_graph':
        for c in ['trust_agent','graph_agent']: s[c]=1-s[c]
    elif mode=='edge_temporal':
        for c in ['edge_agent','temporal_agent']: s[c]=1-s[c]
    elif mode=='multi_byzantine':
        for c in ['trust_agent','graph_agent','cloud_agent']: s[c]=1-s[c]
    elif mode=='random_byzantine':
        for c in s.columns:
            mask=rng.random(len(s))<.35; s.loc[mask,c]=rng.random(mask.sum())
    return s.clip(0,1)

def consensus_experiment(df,m,out):
    _,_,Xv,yv,Xte,yte=split_xy(df,'binary_label',ALL)
    val_scores=agent_scores(Xv,prob_attack(m,Xv,'binary_label')); w=init_w(); hist=[]
    for step,idx in enumerate(np.array_split(np.arange(len(Xv)),10)):
        if len(idx):
            w=update_w(val_scores.iloc[idx],yv.iloc[idx],w,lr=.12); r={'step':step}; r.update(w); hist.append(r)
    hist=pd.DataFrame(hist); hist.to_csv(out/'csv'/'agent_reliability_history.csv',index=False)
    val_r,_=consensus(val_scores,w,.10); th=tune_threshold(yv,val_r)['threshold']
    test_scores=agent_scores(Xte,prob_attack(m,Xte,'binary_label')); r,dis=consensus(test_scores,w,.10); pred=(r>=th).astype(int)
    row={'setting':'ARES_adaptive_consensus','threshold':th,'fault_tolerant_accuracy':accuracy_score(yte,pred),'macro_f1':f1_score(yte,pred,average='macro',zero_division=0),'mcc':matthews_corrcoef(yte,pred),'consensus_stability_score':1-float(np.mean(dis)),'agent_reliability_drift':float(np.std(list(w.values()))),**risk_metrics(yte,r,th)}
    for k,v in w.items(): row[f'weight_{k}']=v
    return pd.DataFrame([row]),hist,w,th

def byzantine(df,m,w,th,out):
    _,_,_,_,X,y=split_xy(df,'binary_label',ALL); base=agent_scores(X,prob_attack(m,X,'binary_label')); base_r,_=consensus(base,w); base_acc=accuracy_score(y,(base_r>=th).astype(int)); rows=[]
    for mode in ['none','trust_agent','graph_agent','cloud_agent','temporal_agent','trust_graph','edge_temporal','multi_byzantine','random_byzantine']:
        s=corrupt_scores(base,mode); r,dis=consensus(s,w); pred=(r>=th).astype(int)
        rows.append({'corruption_mode':mode,'fault_tolerant_accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),'mcc':matthews_corrcoef(y,pred),'byzantine_survival_rate':accuracy_score(y,pred)/max(base_acc,1e-9),'consensus_disagreement':float(np.mean(dis)),**risk_metrics(y,r,th)})
    return pd.DataFrame(rows)

def drift_exp(df,m,out):
    _,_,_,_,X,y=split_xy(df,'binary_label',ALL); base=agent_scores(X,prob_attack(m,X,'binary_label')); w=init_w(); th=.31; rows=[]
    for step,idx in enumerate(np.array_split(np.arange(len(X)),8)):
        sc=base.iloc[idx].copy(); strength=step/7
        for c in ['trust_agent','graph_agent']:
            sc[c]=(1-strength)*sc[c]+strength*(1-sc[c])
        r,dis=consensus(sc,w,.10); pred=(r>=th).astype(int); yy=y.iloc[idx]
        w=update_w(sc,yy,w,lr=.18)
        rows.append({'drift_step':step,'drift_strength':strength,'accuracy':accuracy_score(yy,pred),'macro_f1':f1_score(yy,pred,average='macro',zero_division=0),'consensus_disagreement':float(np.mean(dis)),**{f'weight_{k}':v for k,v in w.items()}})
    return pd.DataFrame(rows)

def edge_metrics(det,out):
    rows=[]
    for _,r in det.iterrows():
        mem=35 if r.model=='LogisticRegression' else 180 if r.model in ['RandomForest','ExtraTrees','GraphNeighborhood'] else 140 if r.model=='HistGradientBoosting' else 220 if r.model in ['XGBoost','LightGBM','CatBoost'] else 520 if 'ARES' in r.model else 120
        comm=48 if 'raw' in r.setting else 96 if 'enhanced' in r.setting else 128
        cpu=min(95,8+r.latency_ms_per_msg*800)
        off=.0 if 'raw' in r.setting else .35
        rows.append({'task':r.task,'setting':r.setting,'model':r.model,'latency_ms_per_msg':r.latency_ms_per_msg,'estimated_memory_mb':mem,'estimated_edge_cpu_percent':cpu,'cloud_offload_ratio':off,'comm_overhead_bytes_per_msg':comm})
    return pd.DataFrame(rows)

def ollama(prompt,model='llama3.2:3b'):
    data=json.dumps({'model':model,'prompt':prompt,'stream':False,'options':{'temperature':0.1}}).encode()
    try:
        req=request.Request('http://localhost:11434/api/generate',data=data,headers={'Content-Type':'application/json'})
        with request.urlopen(req,timeout=60) as r: return json.loads(r.read().decode()).get('response','')
    except Exception: return ''

def fallback_policy(item):
    lvl=item['risk_level']; atk=item.get('predicted_attack_type','unknown'); risk=item.get('consensus_risk',0); dis=item.get('consensus_disagreement',0)
    if lvl=='critical': a='isolate sender and trigger cloud-wide trust audit'; ad='decrease unreliable trust/graph weights by 15 percent for 500 messages'; rec='restore weights after 500 stable verified messages'
    elif lvl=='high': a='quarantine and request cross-edge verification'; ad='increase consensus threshold and cloud verification frequency'; rec='recover after verified benign sequence'
    elif lvl=='medium': a='monitor and increase verification sampling'; ad='increase graph consistency checks'; rec='continue observation'
    else: a='allow with normal monitoring'; ad='no policy change'; rec='not required'
    return a,ad,rec,f'Risk={lvl}; attack={atk}; consensus={risk:.3f}; disagreement={dis:.3f}. Action={a}; adaptation={ad}; recovery={rec}.'

def llm_policy(df,m,out,provider,llm_model):
    _,_,_,_,X,_=split_xy(df,'class_id',ALL); pred=m.predict(X); ap=prob_attack(m,X,'class_id'); scores=agent_scores(X,ap); r,dis=consensus(scores,init_w())
    test=df[df.split.astype(str).str.lower()=='test'].copy().reset_index(drop=True); test['predicted_class_id']=pred; test['predicted_attack_type']=[CLASS.get(int(x),str(x)) for x in pred]; test['attack_probability']=ap; test['consensus_risk']=r; test['consensus_disagreement']=dis
    inc=test[test.predicted_class_id!=0].sort_values('consensus_risk',ascending=False).head(50); rows=[]
    for _,x in inc.iterrows():
        lvl='critical' if x.consensus_risk>=.8 or x.consensus_disagreement>=.35 else 'high' if x.consensus_risk>=.65 else 'medium' if x.consensus_risk>=.45 else 'low'
        item={'sender_id':x.get('sender_id','unknown'),'predicted_attack_type':x.predicted_attack_type,'attack_probability':float(x.attack_probability),'consensus_risk':float(x.consensus_risk),'consensus_disagreement':float(x.consensus_disagreement),'risk_level':lvl}
        if provider=='ollama':
            prompt='You are ARES-V2X adaptive policy agent. Return JSON-like fields Action, WeightAdaptation, ThresholdUpdate, RecoveryPlan, Rationale. Incident: '+json.dumps(item)
            txt=ollama(prompt,llm_model)
            if txt.strip(): a=ad=rec='llm_generated'; exp=txt.strip().replace('\n',' ')
            else: a,ad,rec,exp=fallback_policy(item)
        else: a,ad,rec,exp=fallback_policy(item)
        item.update({'llm_provider':provider,'policy_action':a,'policy_adaptation':ad,'recovery_recommendation':rec,'policy_explanation':exp}); rows.append(item)
    pd.DataFrame(rows).to_csv(out/'csv'/'llm_policy_incidents.csv',index=False)

def run_all(df,out,llm_provider,llm_model,quick=False):
    baselines=['LogisticRegression','RandomForest','ExtraTrees','HistGradientBoosting','MLP']
    if HAS_XGB: baselines.append('XGBoost')
    if HAS_LGBM: baselines.append('LightGBM')
    if HAS_CAT and not quick: baselines.append('CatBoost')
    det=[]; models={}
    for task,target in [('binary','binary_label'),('multiclass','class_id')]:
        for b in baselines:
            print(f'[BASELINE RAW] {task} | {b}'); row,_=train_eval(df,target,task,'raw_edge_cloud',b,RAW,out); det.append(row)
        for b in baselines:
            print(f'[BASELINE ENHANCED] {task} | {b}'); row,_=train_eval(df,target,task,'enhanced_temporal_trust_graph',b,ALL,out); det.append(row)
        print(f'[GRAPH BASELINE] {task}'); row,_=train_eval(df,target,task,'graph_neighborhood_baseline','RandomForest',GRAPH_BASE,out); row['model']='GraphNeighborhood'; det.append(row)
        print(f'[ARES VOTE] {task}'); row,m= train_eval(df,target,task,'ARES_vote_inputs','ARES-V2X',ALL,out,proposed=True); det.append(row); models[task+'_vote']=m; joblib.dump(m,out/'models'/f'{task}_ARES_Vote.joblib')
        print(f'[ARES META] {task}'); row,m=train_meta_fusion(df,target,task,out); det.append(row); models[task+'_meta']=m
    print('[HIERARCHICAL META]'); det.append(hierarchical_meta(df,models['binary_meta'],out))
    det=pd.DataFrame(det); det.to_csv(out/'csv'/'all_detection_metrics.csv',index=False)
    print('[CONSENSUS]'); cons,hist,w,th=consensus_experiment(df,models['binary_meta'],out); cons.to_csv(out/'csv'/'consensus_reliability_metrics.csv',index=False)
    print('[BYZANTINE]'); byz=byzantine(df,models['binary_meta'],w,th,out); byz.to_csv(out/'csv'/'byzantine_resilience_metrics.csv',index=False)
    print('[DRIFT]'); drift=drift_exp(df,models['binary_meta'],out); drift.to_csv(out/'csv'/'temporal_drift_self_healing.csv',index=False)
    print('[ROBUSTNESS]'); robust=[]
    stress_plan={'packet_loss':[.1,.3,.5] if quick else [.1,.2,.3,.5],'feature_dropout':[.1,.3,.5] if quick else [.1,.2,.3,.5],'gps_corruption_m':[50,250] if quick else [25,50,100,250],'delay_injection':[250,1000] if quick else [100,250,500,1000],'stale_trust':[.1,.3,.5] if quick else [.1,.2,.3,.5],'edge_cloud_outage':[.1,.3,.5] if quick else [.1,.2,.3,.5]}
    for mode,levels in stress_plan.items():
        for lv in levels:
            print(f'[ROBUSTNESS] {mode} {lv}'); sdf=stress(df,mode,lv); *_,X,y=split_xy(sdf,'binary_label',ALL); t=time.perf_counter(); p=models['binary_meta'].predict(X); inf=time.perf_counter()-t; row=eval_cls(y,p,'binary',f'{mode}_{lv}','ARES-MetaFusion',0,inf); row['stress_type']=mode; row['stress_level']=lv; robust.append(row)
    robust=pd.DataFrame(robust); robust.to_csv(out/'csv'/'robustness_metrics.csv',index=False)
    print('[SCALABILITY]'); scale=[]; *_,Xf,yf=split_xy(df,'binary_label',ALL)
    for n in [10000,25000,50000,100000,min(197270,len(Xf))]:
        if n>len(Xf): continue
        X=Xf.head(n); y=yf.head(n); t=time.perf_counter(); p=models['binary_meta'].predict(X); inf=time.perf_counter()-t; row=eval_cls(y,p,'binary',f'N={n}','ARES-MetaFusion',0,inf); row['sample_size']=n; scale.append(row)
    scale=pd.DataFrame(scale); scale.to_csv(out/'csv'/'scalability_metrics.csv',index=False)
    edge=edge_metrics(det,out); edge.to_csv(out/'csv'/'edge_deployment_metrics.csv',index=False)
    print('[LLM POLICY]'); llm_policy(df,models['multiclass_meta'],out,llm_provider,llm_model)
    return det,cons,byz,robust,scale,hist,drift,edge

def make_figures(det,cons,byz,rob,scale,hist,drift,edge,df,out):
    palette=['#264653','#2A9D8F','#E9C46A','#F4A261','#E76F51','#6D597A','#355070','#43AA8B','#577590']
    fig,axs=plt.subplots(1,2,figsize=(16,9))
    for ax,task,lab in zip(axs,['binary','multiclass'],['(a)','(b)']):
        s=det[det.task.eq(task)].sort_values('macro_f1',ascending=False).head(9); labels=s.model+'\n'+s.setting.str.replace('_',' ',regex=False).str[:12]
        bars=ax.bar(labels,s.macro_f1,color=palette[:len(s)]); ax.set_ylabel('Macro F1-score'); ax.set_xlabel('Method'); ax.set_title(f'{task.capitalize()} comparative detection'); ax.tick_params(axis='x',rotation=25); panel(ax,lab); ax.set_ylim(max(0,s.macro_f1.min()-.08),min(1.02,s.macro_f1.max()+.04))
        for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{b.get_height():.3f}',ha='center',va='bottom',fontsize=8)
    savefig(fig,out/'figures'/'fig1_detection_comparison')
    fig,axs=plt.subplots(1,2,figsize=(16,9)); axs[0].bar(cons.setting,cons.fault_tolerant_accuracy,color=palette[1]); axs[0].set_ylabel('Fault-tolerant accuracy'); axs[0].set_title('Consensus reliability'); panel(axs[0],'(a)'); axs[1].bar(cons.setting,cons.consensus_stability_score,color=palette[2]); axs[1].set_ylabel('Consensus stability score'); axs[1].set_title('Consensus stability'); panel(axs[1],'(b)'); savefig(fig,out/'figures'/'fig2_consensus_reliability')
    fig,axs=plt.subplots(1,2,figsize=(16,9)); axs[0].bar(byz.corruption_mode,byz.fault_tolerant_accuracy,color=palette[:len(byz)]); axs[0].tick_params(axis='x',rotation=25); axs[0].set_ylabel('Fault-tolerant accuracy'); axs[0].set_title('Byzantine-agent survival'); panel(axs[0],'(a)'); axs[1].bar(byz.corruption_mode,byz.byzantine_survival_rate,color=palette[:len(byz)]); axs[1].tick_params(axis='x',rotation=25); axs[1].set_ylabel('Byzantine survival rate'); axs[1].set_title('Resilience under agent corruption'); panel(axs[1],'(b)'); savefig(fig,out/'figures'/'fig3_byzantine_resilience')
    fig,axs=plt.subplots(1,2,figsize=(16,9))
    for i,(st,p) in enumerate(rob.groupby('stress_type')):
        p=p.sort_values('stress_level'); axs[0].plot(p.stress_level,p.macro_f1,marker='o',lw=2.5,label=st,color=palette[i%len(palette)]); axs[1].plot(p.stress_level,p.mcc,marker='s',lw=2.5,label=st,color=palette[i%len(palette)])
    axs[0].set_xlabel('Stress level'); axs[0].set_ylabel('Macro F1-score'); axs[0].set_title('Robustness degradation'); axs[0].legend(ncol=2); panel(axs[0],'(a)'); axs[1].set_xlabel('Stress level'); axs[1].set_ylabel('MCC'); axs[1].set_title('Dependability under stress'); axs[1].legend(ncol=2); panel(axs[1],'(b)'); savefig(fig,out/'figures'/'fig4_robustness_stress')
    fig,ax=plt.subplots(figsize=(16,9))
    for i,c in enumerate([c for c in hist.columns if c!='step']): ax.plot(hist.step,hist[c],lw=2.5,label=c,color=palette[i%len(palette)])
    ax.set_xlabel('Reliability update step'); ax.set_ylabel('Adaptive agent weight'); ax.set_title('Self-healing agent reliability evolution'); ax.legend(ncol=2); panel(ax,'(a)'); savefig(fig,out/'figures'/'fig5_self_healing_reliability')
    fig,axs=plt.subplots(1,2,figsize=(16,9)); axs[0].plot(drift.drift_step,drift.macro_f1,marker='o',lw=3,color=palette[0]); axs[0].set_xlabel('Drift step'); axs[0].set_ylabel('Macro F1-score'); axs[0].set_title('Temporal drift recovery'); panel(axs[0],'(a)')
    for i,c in enumerate([c for c in drift.columns if c.startswith('weight_')]): axs[1].plot(drift.drift_step,drift[c],lw=2,label=c.replace('weight_',''),color=palette[i%len(palette)])
    axs[1].set_xlabel('Drift step'); axs[1].set_ylabel('Adaptive weight'); axs[1].set_title('Weight adaptation under drift'); axs[1].legend(ncol=2); panel(axs[1],'(b)'); savefig(fig,out/'figures'/'fig6_temporal_drift_self_healing')
    fig,ax=plt.subplots(figsize=(16,9)); ax.plot(scale.sample_size,scale.latency_ms_per_msg,marker='o',lw=3,color=palette[1]); ax.set_xlabel('Number of test messages'); ax.set_ylabel('Latency per message (ms)'); ax.set_title('ARES-V2X scalability'); panel(ax,'(a)'); savefig(fig,out/'figures'/'fig7_scalability')
    sm=df[df.split.astype(str).str.lower().eq('test')].sample(min(50000,len(df)),random_state=42); fig,axs=plt.subplots(1,2,figsize=(16,9)); axs[0].scatter(sm.sender_trust_prior,sm.graph_trust_propagated,s=8,alpha=.25,color=palette[1]); axs[0].set_xlabel('Sender prior trust'); axs[0].set_ylabel('Graph-propagated trust'); axs[0].set_title('Trust propagation explainability'); panel(axs[0],'(a)'); axs[1].scatter(sm.risk_rule_score,sm.graph_local_disagreement,s=8,alpha=.25,color=palette[4]); axs[1].set_xlabel('Rule-based risk'); axs[1].set_ylabel('Graph local disagreement'); axs[1].set_title('Graph disagreement evidence'); panel(axs[1],'(b)'); savefig(fig,out/'figures'/'fig8_trust_graph_explainability')
    fig,axs=plt.subplots(1,2,figsize=(16,9)); s=edge.sort_values('latency_ms_per_msg').head(12); labels=s.model+'\n'+s.setting.str[:8]; axs[0].bar(labels,s.latency_ms_per_msg,color=palette[:len(s)]); axs[0].set_ylabel('Latency/msg (ms)'); axs[0].set_title('Edge latency'); axs[0].tick_params(axis='x',rotation=25); panel(axs[0],'(a)'); axs[1].bar(labels,s.estimated_memory_mb,color=palette[:len(s)]); axs[1].set_ylabel('Estimated memory (MB)'); axs[1].set_title('Edge memory estimate'); axs[1].tick_params(axis='x',rotation=25); panel(axs[1],'(b)'); savefig(fig,out/'figures'/'fig9_edge_deployment_metrics')
    lp=out/'csv'/'llm_policy_incidents.csv'
    if lp.exists():
        llm=pd.read_csv(lp)
        if not llm.empty:
            fig,axs=plt.subplots(1,2,figsize=(16,9)); counts=llm.risk_level.value_counts().reindex(['low','medium','high','critical']).fillna(0); axs[0].bar(counts.index,counts.values,color=palette[:4]); axs[0].set_xlabel('LLM risk level'); axs[0].set_ylabel('Incident count'); axs[0].set_title('LLM policy triage'); panel(axs[0],'(a)'); atk=llm.predicted_attack_type.value_counts(); axs[1].bar(atk.index,atk.values,color=palette[:len(atk)]); axs[1].set_xlabel('Predicted attack type'); axs[1].set_ylabel('Incident count'); axs[1].set_title('LLM incident composition'); axs[1].tick_params(axis='x',rotation=15); panel(axs[1],'(b)'); savefig(fig,out/'figures'/'fig10_llm_policy_agent')

def make_tables(det,cons,byz,rob,scale,drift,edge,out):
    comp=det[['task','setting','model','accuracy','macro_f1','weighted_f1','mcc','latency_ms_per_msg']].sort_values(['task','macro_f1'],ascending=[True,False]); comp.to_csv(out/'tables'/'baseline_comparison_table.csv',index=False); latex(comp.head(60),out/'tables'/'baseline_comparison_table.tex','Comparative performance of baselines and ARES-V2X variants.','tab:baseline_comparison')
    cons.to_csv(out/'tables'/'consensus_reliability_table.csv',index=False); latex(cons,out/'tables'/'consensus_reliability_table.tex','Adaptive consensus reliability and response performance.','tab:consensus_reliability')
    byz.to_csv(out/'tables'/'byzantine_resilience_table.csv',index=False); latex(byz,out/'tables'/'byzantine_resilience_table.tex','ARES-V2X resilience under compromised-agent conditions.','tab:byzantine_resilience')
    rob.to_csv(out/'tables'/'robustness_table.csv',index=False); latex(rob[['stress_type','stress_level','accuracy','macro_f1','mcc','latency_ms_per_msg']],out/'tables'/'robustness_table.tex','ARES-V2X robustness under degraded V2X conditions.','tab:robustness')
    scale.to_csv(out/'tables'/'scalability_table.csv',index=False); latex(scale[['sample_size','macro_f1','latency_ms_per_msg','inference_s']],out/'tables'/'scalability_table.tex','ARES-V2X scalability under increasing message volume.','tab:scalability')
    drift.to_csv(out/'tables'/'temporal_drift_table.csv',index=False); latex(drift,out/'tables'/'temporal_drift_table.tex','Temporal drift and self-healing reliability adaptation.','tab:temporal_drift')
    edge.to_csv(out/'tables'/'edge_deployment_table.csv',index=False); latex(edge.head(50),out/'tables'/'edge_deployment_table.tex','Estimated edge deployment metrics.','tab:edge_deployment')

def report(det,cons,byz,rob,scale,drift,edge,out):
    b=det[det.task.eq('binary')].sort_values('macro_f1',ascending=False).head(12); m=det[det.task.eq('multiclass')].sort_values('macro_f1',ascending=False).head(12)
    lines=['# ARES-V2X Final Deadline Report\n','## Summary\n','ARES-V2X is positioned as a self-healing agentic edge-cloud cyber-resilience framework for dependable V2X security. The final version adds ARES-MetaFusion, stronger tabular baselines, adaptive consensus reliability, Byzantine-agent stress tests, temporal drift recovery, edge deployment profiling, and LLM-assisted response orchestration.','\n## Top Binary Results\n',b.to_markdown(index=False),'\n## Top Multiclass Results\n',m.to_markdown(index=False),'\n## Adaptive Consensus Reliability\n',cons.to_markdown(index=False),'\n## Byzantine/Compromised-Agent Resilience\n',byz.to_markdown(index=False),'\n## Temporal Drift and Self-Healing\n',drift.to_markdown(index=False),'\n## Edge Deployment Metrics\n',edge.head(25).to_markdown(index=False),'\n## Robustness Summary\n',rob[['stress_type','stress_level','accuracy','macro_f1','mcc']].to_markdown(index=False),'\n## Scalability Summary\n',scale[['sample_size','macro_f1','latency_ms_per_msg','inference_s']].to_markdown(index=False)]
    lp=out/'csv'/'llm_policy_incidents.csv'
    if lp.exists():
        llm=pd.read_csv(lp); cols=['risk_level','predicted_attack_type','attack_probability','consensus_risk','consensus_disagreement','policy_action','policy_adaptation','recovery_recommendation']; lines+=['\n## LLM Policy Agent Samples\n',llm[[c for c in cols if c in llm.columns]].head(10).to_markdown(index=False)]
    lines+=['\n## Recommended Claim\n','The main claim should not be raw classifier superiority alone. The strongest claim is dependable cyber-resilience: ARES-V2X combines meta-fusion detection, adaptive consensus reliability, Byzantine survivability, temporal drift recovery, edge deployment profiling, and LLM-assisted response/recovery orchestration for V2X edge-cloud security.']
    (out/'reports'/'ares_v2x_final_deadline_report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--out-dir',default='outputs_ares_v2x_final_deadline'); ap.add_argument('--max-per-class',type=int,default=100000); ap.add_argument('--keep-leakage-risk-features',action='store_true'); ap.add_argument('--llm-provider',choices=['none','ollama'],default='none'); ap.add_argument('--llm-model',default='llama3.2:3b'); ap.add_argument('--quick',action='store_true')
    args=ap.parse_args(); out=Path(args.out_dir); ensure_dirs(out)
    print('[INFO] Loading dataset.'); df=load_data(Path(args.csv),args.max_per_class,args.keep_leakage_risk_features)
    print('[INFO] Engineering temporal/trust features.'); df=add_temporal_trust(df)
    print('[INFO] Engineering graph features.'); df=add_graph(df,out); df.to_csv(out/'csv'/'engineered_dataset_snapshot.csv',index=False)
    print('[INFO] Running final ARES-V2X experiments.'); det,cons,byz,rob,scale,hist,drift,edge=run_all(df,out,args.llm_provider,args.llm_model,quick=args.quick)
    print('[INFO] Creating tables.'); make_tables(det,cons,byz,rob,scale,drift,edge,out)
    print('[INFO] Creating figures.'); make_figures(det,cons,byz,rob,scale,hist,drift,edge,df,out)
    print('[INFO] Writing report.'); report(det,cons,byz,rob,scale,drift,edge,out)
    print('\n[DONE] ARES-V2X final deadline pipeline completed.'); print('[OUT]',out.resolve())

if __name__=='__main__': main()
