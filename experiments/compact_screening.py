"""Training-only compact pure-hurdle screening; no supplied test file is read."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import z5429763 as s

p=argparse.ArgumentParser()
p.add_argument('train_file')
p.add_argument('--seed',type=int,default=42)
p.add_argument('--reg-trees',type=int,nargs='+',default=[300,500,700])
p.add_argument('--cls-trees',type=int,nargs='+',default=[250,350,450])
p.add_argument('--output',required=True)
a=p.parse_args()
s.THREADS=4
train=pd.read_csv(a.train_file)
fit,val=train_test_split(np.arange(len(train)),test_size=.2,random_state=a.seed,stratify=train.rating)
A,V=train.iloc[fit],train.iloc[val]
y=np.log1p(A.revenue.to_numpy()); truth=np.log1p(V.revenue.to_numpy())
is_low=y<=np.log1p(1000)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
report=dict(seed=a.seed,train_sha256=hashlib.sha256(Path(a.train_file).read_bytes()).hexdigest(),baseline_source_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),fit_rows=len(fit),val_rows=len(val),threads=4,gate_params=dict(iterations=350,depth=5,learning_rate=.08,l2_leaf_reg=10,max_ctr_complexity=1),reg_params=dict(iterations=max(a.reg_trees),depth=6,learning_rate=.08,l2_leaf_reg=10,max_ctr_complexity=1),light_params=s.REVENUE_LGB_PARAMS,cls_params=dict(iterations=max(a.cls_trees),depth=6,learning_rate=.1,l2_leaf_reg=5,max_ctr_complexity=1),results=[])
def save():
 out.write_text(json.dumps(report,indent=2)+'\n')
save()

started=perf_counter()
builder=s.RevenueRegressor()
GX=builder._features(A,True);GV=builder._features(V)
gate=CatBoostClassifier(**report['gate_params'],loss_function='Logloss',random_seed=s.SEED,thread_count=s.THREADS,verbose=False,allow_writing_files=False).fit(GX,is_low.astype(int),cat_features=s.categorical_columns(GX))
prob=gate.predict_proba(GV)[:,1]
report['gate_seconds']=perf_counter()-started
print('Gate completed',round(report['gate_seconds'],2),flush=True)
save()

def expert(mask):
 started=perf_counter()
 frame=A.iloc[mask];target=y[mask]
 builder=s.TargetEncodedRevenueRegressor()
 X=builder._features(frame,True);D=builder._features(V)
 C=builder._cat_features(frame,X,target,True);E=builder._cat_features(V,D)
 cat=CatBoostRegressor(**report['reg_params'],loss_function='RMSE',random_seed=s.SEED,thread_count=s.THREADS,verbose=False,allow_writing_files=False).fit(C,target,cat_features=s.categorical_columns(C))
 X,D=X.copy(),D.copy();X.columns=[re.sub('[^A-Za-z0-9_]+','_',c) for c in X];D.columns=X.columns
 for name in s.categorical_columns(X):
  dtype=pd.CategoricalDtype(pd.unique(X[name]));X[name]=X[name].astype(dtype);D[name]=D[name].where(D[name].isin(dtype.categories),None).astype(dtype)
 light=LGBMRegressor(**s.REVENUE_LGB_PARAMS,n_jobs=s.THREADS,random_state=s.SEED,verbosity=-1).fit(X,target)
 lp=light.predict(D)
 preds={n:.5*cat.predict(E,ntree_end=n)+.5*lp for n in a.reg_trees}
 return preds,perf_counter()-started

lower,low_seconds=expert(is_low)
print('Low expert completed',round(low_seconds,2),flush=True)
higher,high_seconds=expert(~is_low)
print('High expert completed',round(high_seconds,2),flush=True)
report['low_seconds']=low_seconds;report['high_seconds']=high_seconds
for n in a.reg_trees:
 pred=np.maximum(0,prob*np.clip(lower[n],0,np.log1p(1000))+(1-prob)*np.maximum(np.log1p(1000),higher[n]))
 mask=V.revenue.le(1000).to_numpy()
 row=dict(task='regression',trees=n,RMSLE=float(np.sqrt(np.mean((pred-truth)**2))),tiny_RMSLE=float(np.sqrt(np.mean((pred[mask]-truth[mask])**2))),other_RMSLE=float(np.sqrt(np.mean((pred[~mask]-truth[~mask])**2))))
 report['results'].append(row)
 np.savez_compressed(out.with_name(out.stem+f'-reg{n}.npz'),pred=pred,target=truth,validation=val)
 print(json.dumps(row),flush=True)
save()

started=perf_counter()
classifier=s.make_classifier(**report['cls_params']).fit(GX,A.rating,cat_features=s.categorical_columns(GX))
report['classification_max_trees_seconds']=perf_counter()-started
for n in a.cls_trees:
 pred=classifier.predict(GV,ntree_end=n).reshape(-1)
 row=dict(task='classification',trees=n,MacroF1=float(f1_score(V.rating,pred,average='macro')),accuracy=float(accuracy_score(V.rating,pred)))
 report['results'].append(row)
 np.savez_compressed(out.with_name(out.stem+f'-cls{n}.npz'),pred=pred,target=V.rating.to_numpy(dtype=str),validation=val)
 print(json.dumps(row),flush=True)
save()
