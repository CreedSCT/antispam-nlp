from pathlib import Path
import joblib,json
root=Path(__file__).resolve().parent
b=joblib.load(root/'artifacts/model.joblib');v=b['pipeline']['tfidf'];m=b['pipeline']['model']
(root/'site_source').mkdir(exist_ok=True)
model={'vocabulary':{k:int(i) for k,i in v.vocabulary_.items()},'idf':v.idf_.tolist(),'weights':(m.feature_log_prob_[1]-m.feature_log_prob_[0]).tolist(),'threshold':b['threshold'],'model_name':b['model_name']}
(root/'site_source/model.json').write_text(json.dumps(model,separators=(',',':')))
