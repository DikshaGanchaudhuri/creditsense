import os
import sys
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.pipeline import load_raw_data, build_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              precision_score, recall_score,
                              confusion_matrix)

print("Loading model and data...")
model    = joblib.load(os.path.join(project_root, 'src', 'models', 'credit_risk_model.pkl'))
raw      = load_raw_data()
X, y     = build_features(raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, 1]

print("Computing metrics...")

auc      = roc_auc_score(y_test, y_prob)
acc      = accuracy_score(y_test, y_pred)
prec     = precision_score(y_test, y_pred)
rec      = recall_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

y_pred_035 = (y_prob >= 0.35).astype(int)
tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_035).ravel()

missed_reduction_pct = round((fn - fn2) / fn * 100, 1) if fn > 0 else 0

features = joblib.load(os.path.join(project_root, 'src', 'models', 'feature_columns.pkl'))
importances = pd.Series(model.feature_importances_, index=features)
top3 = importances.sort_values(ascending=False).head(3)

engine  = create_engine('postgresql+psycopg2://localhost/creditsense')
stats   = pd.read_sql("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN class='bad' THEN 1 ELSE 0 END) as defaults,
        ROUND(AVG(credit_amount)) as avg_loan,
        ROUND(AVG(age)) as avg_age
    FROM credit_data
""", engine).iloc[0]

metrics = {
    "total_applicants":       int(stats['total']),
    "total_defaults":         int(stats['defaults']),
    "default_rate_pct":       round(stats['defaults'] / stats['total'] * 100, 1),
    "avg_loan_amount":        int(stats['avg_loan']),
    "avg_applicant_age":      int(stats['avg_age']),
    "model_auc":              round(auc, 3),
    "model_accuracy_pct":     round(acc * 100, 1),
    "model_precision_pct":    round(prec * 100, 1),
    "model_recall_pct":       round(rec * 100, 1),
    "defaulters_caught_050":  int(tp),
    "defaulters_missed_050":  int(fn),
    "defaulters_caught_035":  int(tp2),
    "defaulters_missed_035":  int(fn2),
    "missed_reduction_pct":   missed_reduction_pct,
    "top_feature_1":          top3.index[0],
    "top_feature_2":          top3.index[1],
    "top_feature_3":          top3.index[2],
}

print("\nLive metrics computed:")
for k, v in metrics.items():
    print(f"   {k}: {v}")

template = """
You are a senior credit risk analyst writing an executive summary 
for a bank's lending committee. Based on the following model metrics 
from the CreditSense risk analytics system, write a clear, professional, 
plain-English narrative summary. 

Do not use bullet points. Write in flowing paragraphs. 
Keep it under 200 words. Focus on business implications.

DATASET:
- Total loan applicants analyzed: {total_applicants}
- Total defaults in dataset: {total_defaults} ({default_rate_pct}% default rate)
- Average loan amount: {avg_loan_amount} DM
- Average applicant age: {avg_applicant_age} years

MODEL PERFORMANCE:
- ROC-AUC score: {model_auc}
- Accuracy: {model_accuracy_pct}%
- Precision (when flagging risk): {model_precision_pct}%
- Recall (defaulters caught): {model_recall_pct}%

THRESHOLD A/B TEST:
- At threshold 0.50: caught {defaulters_caught_050} defaulters, 
  missed {defaulters_missed_050}
- At threshold 0.35: caught {defaulters_caught_035} defaulters, 
  missed {defaulters_missed_035}
- Switching to 0.35 reduces missed defaulters by {missed_reduction_pct}%

TOP PREDICTIVE FEATURES:
- {top_feature_1}, {top_feature_2}, {top_feature_3}

Write the executive summary now:
"""

prompt = PromptTemplate(
    input_variables=list(metrics.keys()),
    template=template
)

print("\n Sending to Gemini...")

llm = Ollama(model="llama3")

chain = prompt | llm
response = chain.invoke(metrics)

summary_text = response

print("\n" + "="*60)
print("CREDITSENSE — AI GENERATED RISK NARRATIVE")
print("="*60)
print(summary_text)
print("="*60)

output_path = os.path.join(project_root, 'data', 'ai_narrative.txt')
with open(output_path, 'w') as f:
    f.write("CREDITSENSE — AI GENERATED RISK NARRATIVE\n")
    f.write("="*60 + "\n\n")
    f.write(summary_text)
    f.write("\n\n" + "="*60)

print(f"\n✅ Narrative saved to data/ai_narrative.txt")