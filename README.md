# CreditSense — AI-Augmented Credit Risk Analytics Dashboard

> End-to-end data science project: raw data ingestion → feature engineering 
> → ML modeling → interactive dashboard → AI-generated risk narratives.

---

## Live Demo
- 📊 [Tableau Dashboard](https://public.tableau.com/app/profile/diksha.ganchaudhuri/viz/CreditSenseCreditRiskDashboard/CreditSense)

---

## Project Highlights

- **0.802 ROC-AUC** on 1,000-record UCI German Credit dataset
- **12 domain-driven features** engineered from raw loan application data
- **18% reduction in missed defaulters** via structured A/B threshold test
- **5-view interactive Tableau dashboard** connected to live CSV exports
- **AI narration layer** using LangChain + Ollama (Llama 3) that 
  auto-generates plain-English risk summaries from live metrics

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data storage | PostgreSQL |
| Data processing | Python, pandas |
| Machine learning | XGBoost, scikit-learn |
| Visualisation | Tableau Public |
| AI narration | LangChain, Ollama (Llama 3) |
| Version control | Git, GitHub |

---

## Key Findings

### Model Performance
The XGBoost classifier achieved a ROC-AUC of 0.802, trained on 12 
engineered features including checking account status, employment 
stability score, monthly burden ratio, and debt burden index.

### Threshold A/B Test
Comparing classification thresholds of 0.50 vs 0.35 revealed that 
a risk-averse threshold of 0.35 reduces missed defaulters by 33.3%, accepting a moderate increase in false positives — a sound strategy for conservative lending portfolios.

### Top Predictive Features
1. `no_checking` — absence of a checking account is the strongest default signal
2. `checking_status_no_checking` — corroborates checking account risk
3. `savings_status_lt100` — low savings strongly predicts default

### AI-Generated Executive Summary
> *"In light of these results, I recommend adjusting our threshold to 0.35, 
> which would reduce missed defaulters by 33.3%. Our top predictive features 
> — no_checking, checking_status_no_checking, and savings_status_lt100 — 
> provide valuable insights into the characteristics of high-risk borrowers."*
>
> — CreditSense AI Narration Layer (LangChain + Llama 3)

---

## Dataset
[UCI German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))  
1,000 loan applicants · 20 original features · Binary classification (good/bad credit risk)