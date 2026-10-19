"""Experimental category model. Synthetic training data, NOT real-world validation."""
import json,joblib,numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold,cross_val_score
R=Path(__file__).resolve().parent
examples={
'advertising':['Buy discounted shoes in our online store','Save money on electronics during our sale','Special offer on health products today','Shop luxury watches at low prices','Subscribe to our premium software deal','Order vitamins with free shipping','Get a coupon for your next purchase','Book a holiday package at a discount'],
'jobs':['We are hiring a Python developer','Apply for an open engineering position','Your job interview is scheduled next week','Our company offers a full time vacancy','Send your resume to the recruitment team','A remote analyst role is available','We would like to discuss your career experience','Invitation to interview for a security specialist job'],
'dating':['I would love to meet you for a date','Find your romantic partner on our dating site','Singles near you are looking for love','Let us arrange a romantic dinner together','Your dating profile received a new match','Meet someone special and start a relationship','I enjoyed our date and want to see you again','Join our singles club to meet your soulmate'],
'fraud':['Send your bank password to claim a prize','Transfer an advance fee to receive your inheritance','Verify your credit card number immediately or lose access','You won a lottery you never entered pay a processing fee','I need your help moving millions from a frozen bank account','Provide your login credentials to prevent account suspension','Pay a release fee for your unexpected cash reward','Send your secret recovery phrase to unlock your wallet'],
'ordinary':['The project meeting will take place on Tuesday','Please review the attached budget report','Your lecture notes are ready for review','The server maintenance window starts tonight','Can you confirm the agenda for our discussion','Here is the summary of our research findings','The library book you requested is available','We have updated the documentation for the API']}
wrappers=['Subject: Message\n\n{}','Hello. {}. Thank you.','Good afternoon. {}. Regards.','{} Please read this message.']
texts=[];labels=[];groups=[]
for label,items in examples.items():
 for i,item in enumerate(items):
  for w in wrappers:texts.append(w.format(item));labels.append(label);groups.append(label+str(i))
model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,strip_accents='unicode')),('model',LogisticRegression(C=4,max_iter=1000,random_state=42))])
cv=cross_val_score(model,texts,labels,groups=groups,cv=GroupKFold(4),scoring='f1_macro')
model.fit(texts,labels);joblib.dump(model,R/'artifacts/categories.joblib')
(R/'data/category_synthetic.json').write_text(json.dumps(examples,indent=2))
(R/'reports/categories.json').write_text(json.dumps({'synthetic':True,'base_texts':40,'augmented_texts':160,'group_cv_f1_macro':cv.tolist(),'mean':float(cv.mean()),'limitation':'Artificial examples only. No independent real email category labels. Do not interpret as real-world quality.'},indent=2))
(R/'site_source').mkdir(exist_ok=True)
v=model['tfidf'];m=model['model']
(R/'site_source/categories.json').write_text(json.dumps({'vocabulary':{k:int(i) for k,i in v.vocabulary_.items()},'idf':v.idf_.tolist(),'weights':m.coef_.tolist(),'intercepts':m.intercept_.tolist(),'classes':m.classes_.tolist()},separators=(',',':')))
print(cv)
