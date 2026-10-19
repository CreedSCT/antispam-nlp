"""Reproducible training and single final evaluation. Run from project root."""
from pathlib import Path
import json,csv,platform,hashlib
import numpy as np
import sklearn,joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
from app.text import read_archive

ROOT=Path(__file__).resolve().parent

def scores(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p,zero_division=0)),f1=float(f1_score(y,p,zero_division=0)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),fpr=float(fp/(tn+fp)) if tn+fp else 0)

def pipeline(kind):
    return Pipeline([('tfidf',TfidfVectorizer(lowercase=True,strip_accents='unicode',ngram_range=(1,2),min_df=2,max_df=.99,sublinear_tf=True,max_features=60000)),('model',LogisticRegression(C=4,max_iter=1000,random_state=42) if kind=='logistic' else ComplementNB(alpha=.5))])

def main(test_path):
    test=read_archive(test_path,'teacher_test')
    test_hash={r['hash'] for r in test}
    rows=[];manifest=[]
    for path in sorted((ROOT/'data/raw').glob('*.tar.*')):
        source='LingSpam' if 'lingspam' in path.name else 'SpamAssassin'
        chunk=read_archive(path,source);rows+=chunk
        manifest.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parsed=len(chunk),source=source))
    unique={};overlap=0;duplicates=0;conflicts=0
    for row in rows:
        if row['hash'] in test_hash:overlap+=1;continue
        if row['hash'] in unique:
            duplicates+=1
            if unique[row['hash']]['label']!=row['label']:conflicts+=1
            continue
        unique[row['hash']]=row
    rows=list(unique.values())
    X=np.array([r['text'] for r in rows],dtype=object);y=np.array([r['label'] for r in rows])
    tr,va=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    experiments={};models={}
    for kind in ['naive_bayes','logistic']:
        model=pipeline(kind)
        cv=cross_validate(model,X[tr],y[tr],cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring=['precision','recall','f1'],n_jobs=1)
        model.fit(X[tr],y[tr]);models[kind]=model
        experiments[kind]={'cv':{metric:{'mean':float(cv['test_'+metric].mean()),'std':float(cv['test_'+metric].std())} for metric in ['precision','recall','f1']},'validation_default':scores(y[va],model.predict(X[va]))}
        print(kind,experiments[kind],flush=True)
    # Select threshold using ONLY validation set: FPR <= 1%, then maximum F1.
    candidates=[]
    for kind,model in models.items():
        probs=model.predict_proba(X[va])[:,1]
        for threshold in np.linspace(.05,.99,95):
            metric=scores(y[va],probs>=threshold)
            candidates.append((kind,float(threshold),metric))
    feasible=[x for x in candidates if x[2]['fpr']<=.01 and x[2]['tp']>0]
    selected=max(feasible,key=lambda x:(x[2]['f1'],x[2]['precision'],x[1])) if feasible else min(candidates,key=lambda x:(x[2]['fpr'],-x[2]['f1']))
    kind,threshold,validation=selected
    # Keep the exact model whose validation predictions selected the threshold.
    model=models[kind]
    test_probs=model.predict_proba([r['text'] for r in test])[:,1]
    pred=(test_probs>=threshold).astype(int)
    final=scores([r['label'] for r in test],pred)
    result=dict(seed=42,python=platform.python_version(),sklearn=sklearn.__version__,manifest=manifest,raw_train=len(rows)+duplicates+overlap,unique_train_pool=len(rows),train_count=len(tr),validation_count=len(va),removed_test_overlap=overlap,removed_duplicates=duplicates,label_conflicts=conflicts,train_sources={s:sum(r['source']==s for r in rows) for s in sorted(set(r['source'] for r in rows))},train_labels={'ham':int(sum(y==0)),'spam':int(sum(y==1))},test_labels={'ham':sum(r['label']==0 for r in test),'spam':sum(r['label']==1 for r in test)},test_unique=len(test_hash),experiments=experiments,selected_model=kind,threshold=threshold,validation=validation,test=final,notes=['Test excluded from fitting, model selection and threshold tuning. Exact normalized text overlaps removed from training pool.','CV at default threshold on train only. Validation used for model/threshold selection. No refit after selection.','Near duplicates and shared corpus origins may remain. Test performance is not a guarantee on modern or Russian email.'])
    import tarfile
    with tarfile.open(test_path, 'r:*') as archive:
        total_test_files=sum(item.isfile() for item in archive)
    result.update(test_total_files=total_test_files, test_skipped_empty_text=total_test_files-len(test), test_scored_count=len(test), test_duplicate_texts=len(test)-len(test_hash))
    (ROOT/'reports/metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    joblib.dump({'pipeline':model,'threshold':threshold,'model_name':kind},ROOT/'artifacts/model.joblib')
    with (ROOT/'reports/test_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['file','true_label','predicted_label','spam_score'])
        for r,p,prob in zip(test,pred,test_probs):w.writerow([r['file'],r['label'],int(p),float(prob)])
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--test',required=True);a=p.parse_args();main(a.test)
