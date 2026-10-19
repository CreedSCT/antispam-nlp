from pathlib import Path
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[1]
bundle=joblib.load(ROOT/'artifacts/model.joblib')
category_model=joblib.load(ROOT/'artifacts/categories.joblib')
app=FastAPI(title='Антиспам NLP',version='1.0.0')
class EmailInput(BaseModel):
    text: str = Field(min_length=1,max_length=50000)
@app.get('/health')
def health():return {'status':'ok','model':bundle['model_name']}
@app.post('/predict')
def predict(item:EmailInput):
    score=float(bundle['pipeline'].predict_proba([item.text])[0,1])
    category=str(category_model.predict([item.text])[0])
    return {'label':'spam' if score>=bundle['threshold'] else 'ham','spam_score':score,'threshold':bundle['threshold'],'category':category,'category_experimental':True}
@app.get('/',response_class=HTMLResponse)
def index():return (ROOT/'app/index.html').read_text(encoding='utf-8')
