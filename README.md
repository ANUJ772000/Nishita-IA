# 💊 MedGuard — Counterfeit Medicine Detection Platform

An interactive data analytics dashboard built with Streamlit for detecting counterfeit and substandard medicines in India's pharmaceutical supply chain.

---

## 🚀 Live Demo

Deployed on Streamlit Community Cloud.  
**[👉 Click here to open the dashboard](https://share.streamlit.io)**

---

## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Course** | MBA/MGB — Data Analytics |
| **Business** | MedGuard — Counterfeit Medicine Detection Platform |
| **Dataset** | 200 synthetic medicine inspection records (25 features) |
| **Target** | Binary: Genuine vs Counterfeit/Substandard |
| **Data seed** | 99 (reproducible; WHO/CDSCO risk-weighted model) |

### Problem
Counterfeit and substandard medicines cause thousands of deaths annually in India.
Frontline pharmacy staff and drug inspectors lack a systematic, data-driven tool to flag
suspicious medicines at the point of purchase or dispensing.

### Solution
MedGuard is a mobile + web platform where staff log medicine inspections via a structured
20-question survey. Machine learning assesses counterfeit risk in real time, clusters users
by detection capability, and surfaces hidden supply chain patterns through association rule mining.

---

## 📊 Dashboard Pages

| Page | Technique | What It Does |
|------|-----------|-------------|
| **1 — Overview & EDA** | Exploratory Data Analysis | KPI cards, demographics, counterfeit rates, correlation heatmap, feature importance, outlier analysis |
| **2 — Classification** | 6 ML Models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, Naive Bayes — with confusion matrices, ROC curves, metric comparison |
| **3 — K-Means Clustering** | Unsupervised Clustering | Elbow method, silhouette score, cluster profiles, inspector segment naming |
| **4 — Association Rules** | Apriori Algorithm | Support/confidence/lift analysis, actionable counterfeit risk rules |
| **5 — Regression** | Linear / Ridge / Lasso | Predict Days_To_Escalation; feature selection via Lasso coefficient shrinkage |
| **6 — PCA** | Dimensionality Reduction | Scree plot, biplot with loading arrows, 3D scatter, feature loading heatmap |

---

## 🏗️ Repository Structure

```
medguard-dashboard/
│
├── app.py               ← Main Streamlit app (self-contained, no CSV needed)
├── requirements.txt     ← Python dependencies
├── runtime.txt          ← Python version for Streamlit Cloud
└── README.md            ← This file
```

> **Note:** `app.py` generates the synthetic dataset automatically at startup using
> `@st.cache_data` — no separate data file is required. The same seed (99) is used
> every time, ensuring fully reproducible results.

---

## ⚙️ How to Deploy on Streamlit Community Cloud

1. **Fork or upload** this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select your repository → set the main file to `app.py`.
4. Click **"Deploy"** — the app will be live in ~2 minutes.

---

## 💻 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/medguard-dashboard.git
cd medguard-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## 🔬 Technical Stack

| Component | Library/Version |
|-----------|----------------|
| Dashboard framework | Streamlit 1.35 |
| Data manipulation | Pandas 2.2, NumPy 1.26 |
| Visualisation | Plotly 5.22 (all charts) |
| Machine learning | scikit-learn 1.5 |
| Association rules | mlxtend 0.23 |
| Python | 3.11 |

---

## 🎨 Design

- **Colour scheme:** Deep navy `#0a0f2c` + Crimson red `#c0392b` + White (medical/trust/danger theme)
- **Typography:** Sora (display) + DM Sans (body) via Google Fonts
- **Charts:** 100% Plotly — no matplotlib or seaborn
- **Layout:** Wide layout · Sidebar navigation · Styled KPI cards · Insight boxes after every chart

---

## 📚 Academic Notes

1. **Recall > Accuracy** — In counterfeit medicine detection, a false negative (calling a fake genuine) causes patient harm. MedGuard optimises for maximum Recall.
2. **PCA for rural deployment** — Dimensionality reduction enables real-time inference on low-bandwidth 2G/3G mobile devices used by rural pharmacists and drug inspectors.
3. **Outlier awareness** — 16 deliberate outliers (~8%) represent real-world anomalies: over-cautious inspectors, sophisticated fakes, legitimate grey-market clearance, and inexperienced inspectors who missed counterfeits.
4. **Association rules** — Focus on rules with `IsCounterfeit` as the consequent — these are MedGuard's most actionable supply chain alerts.
5. **Synthetic data** — Generated using a logically derived risk score based on WHO and CDSCO counterfeit medicine risk factors. Academically defensible for demonstration purposes.

---

## 👤 Author

MBA/MGB Data Analytics — Individual Academic Project  
MedGuard Platform · India · Academic Year 2024–25
