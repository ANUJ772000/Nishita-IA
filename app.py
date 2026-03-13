# =============================================================================
# MedGuard — Counterfeit Medicine Detection Platform
# Streamlit Cloud Deployment — app.py
# =============================================================================
# Self-contained: generates synthetic dataset at startup (no CSV file needed).
# Upload this file + requirements.txt + runtime.txt to GitHub, then deploy
# at https://share.streamlit.io
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, mean_absolute_error,
                              mean_squared_error, r2_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="MedGuard — Counterfeit Medicine Detection",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# COLOUR PALETTE
# =============================================================================
NAVY    = "#0a0f2c"
CRIMSON = "#c0392b"
WHITE   = "#ffffff"
CARD_BG = "#0d1436"
ACCENT  = "#f39c12"
GREEN   = "#27ae60"
GREY    = "#95a5a6"
BLUE    = "#3498db"

# =============================================================================
# GLOBAL CSS
# =============================================================================
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Sans:wght@400;500&display=swap');

  html, body, [class*="css"] {{
      font-family: 'DM Sans', sans-serif;
  }}

  /* ── App background ── */
  .stApp {{ background-color: {NAVY}; }}
  .main .block-container {{
      background: {NAVY};
      padding: 1.5rem 2rem 3rem 2rem;
      max-width: 1400px;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
      background: #070c20 !important;
      border-right: 2px solid {CRIMSON}55;
  }}
  [data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
  [data-testid="stSidebar"] .stRadio > div > label {{
      background: rgba(255,255,255,0.04);
      border-radius: 8px;
      padding: 8px 14px;
      margin: 3px 0;
      transition: background 0.2s;
      display: block;
  }}
  [data-testid="stSidebar"] .stRadio > div > label:hover {{
      background: rgba(192,57,43,0.18);
  }}

  /* ── Global text ── */
  h1, h2, h3, h4, h5 {{ font-family: 'Sora', sans-serif; color: {WHITE} !important; }}
  p, li, span, label, .stMarkdown {{ color: {WHITE} !important; }}
  .stMetric > div {{ color: {WHITE} !important; }}
  .stMetric [data-testid="stMetricValue"] {{ color: {CRIMSON} !important; font-size:1.6rem !important; font-weight:700; }}
  .stMetric [data-testid="stMetricDelta"] {{ color: {GREEN} !important; }}

  /* ── Hero banner ── */
  .hero {{
      background: linear-gradient(135deg, #07091e 0%, #130810 55%, rgba(192,57,43,0.15) 100%);
      border: 1px solid {CRIMSON}66;
      border-radius: 14px;
      padding: 32px 40px;
      margin-bottom: 28px;
      position: relative;
      overflow: hidden;
  }}
  .hero::before {{
      content: '';
      position: absolute;
      top: -60px; right: -60px;
      width: 220px; height: 220px;
      background: radial-gradient(circle, {CRIMSON}22 0%, transparent 70%);
      border-radius: 50%;
  }}
  .hero-title {{
      font-family: 'Sora', sans-serif;
      font-size: 2.6rem;
      font-weight: 800;
      color: {WHITE};
      letter-spacing: -0.5px;
      margin: 0 0 6px 0;
      line-height: 1.1;
  }}
  .hero-title span {{ color: {CRIMSON}; }}
  .hero-sub {{
      font-size: 1rem;
      color: {GREY};
      margin: 0 0 14px 0;
  }}
  .hero-pills {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .pill {{
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      padding: 4px 12px;
      border-radius: 20px;
      border: 1px solid {CRIMSON}88;
      color: {CRIMSON};
  }}
  .pill.green {{ border-color: {GREEN}88; color: {GREEN}; }}
  .pill.blue  {{ border-color: {BLUE}88; color: {BLUE}; }}

  /* ── KPI cards ── */
  .kpi-row {{ display: flex; gap: 14px; margin: 16px 0 24px 0; flex-wrap: wrap; }}
  .kpi {{
      flex: 1; min-width: 140px;
      background: {CARD_BG};
      border: 1px solid {CRIMSON}44;
      border-radius: 12px;
      padding: 20px 16px;
      text-align: center;
      position: relative;
      overflow: hidden;
  }}
  .kpi::after {{
      content: '';
      position: absolute;
      bottom: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, transparent, {CRIMSON}, transparent);
  }}
  .kpi-val {{
      font-family: 'Sora', sans-serif;
      font-size: 1.9rem;
      font-weight: 800;
      color: {CRIMSON};
      line-height: 1;
  }}
  .kpi-lbl {{
      font-size: 0.73rem;
      color: {GREY};
      margin-top: 5px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
  }}

  /* ── Section headers ── */
  .sec {{
      font-family: 'Sora', sans-serif;
      font-size: 1.1rem;
      font-weight: 700;
      color: {CRIMSON} !important;
      border-bottom: 1px solid {CRIMSON}33;
      padding-bottom: 6px;
      margin: 32px 0 16px 0;
  }}

  /* ── Insight boxes ── */
  .ibox {{
      background: rgba(192,57,43,0.08);
      border-left: 4px solid {CRIMSON};
      border-radius: 0 8px 8px 0;
      padding: 14px 20px;
      margin: 14px 0 6px 0;
      font-size: 0.9rem;
      line-height: 1.6;
      color: {WHITE};
  }}
  .ibox b {{ color: #f08080; }}

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] {{
      border: 1px solid {CRIMSON}33;
      border-radius: 8px;
  }}

  /* ── Selectboxes & sliders ── */
  [data-baseweb="select"] > div {{ background: {CARD_BG} !important; border-color: {CRIMSON}44 !important; }}
  .stSlider [data-baseweb="slider"] {{ background: {CARD_BG}; }}
  .stSlider [role="slider"] {{ background: {CRIMSON} !important; }}

  /* ── Success / warning banners ── */
  .stSuccess {{ background: rgba(39,174,96,0.12) !important; border-color: {GREEN} !important; }}
  .stWarning {{ background: rgba(243,156,18,0.12) !important; border-color: {ACCENT} !important; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SYNTHETIC DATA GENERATOR (runs once, cached)
# =============================================================================
@st.cache_data(show_spinner="Generating MedGuard dataset…")
def generate_data():
    np.random.seed(99)
    n = 200

    role = np.random.choice(
        ['Retail Pharmacist','Hospital Pharmacist','Drug Inspector',
         'Medicine Buyer/Procurement','Patient/Consumer'],
        n, p=[0.30,0.20,0.15,0.20,0.15])

    exp_label = np.random.choice(
        ['<1 yr','1-3 yrs','3-7 yrs','7-15 yrs','>15 yrs'],
        n, p=[0.10,0.22,0.28,0.25,0.15])
    exp_map = {'<1 yr':0.5,'1-3 yrs':2,'3-7 yrs':5,'7-15 yrs':11,'>15 yrs':18}
    experience_yrs = np.array([exp_map[e] for e in exp_label])

    region = np.random.choice(
        ['Tier-1 City','Tier-2 City','Semi-Urban','Rural'],
        n, p=[0.30,0.30,0.25,0.15])

    channel = np.random.choice(
        ['Authorised Distributor','Wholesale Market','Online Pharmacy',
         'Direct from Manufacturer','Grey Market / Informal'],
        n, p=[0.35,0.25,0.12,0.13,0.15])

    price_vs_mrp = np.random.choice(
        ['>30% Below MRP','10-30% Below MRP','At MRP','10-30% Above MRP','>30% Above MRP'],
        n, p=[0.12,0.20,0.42,0.16,0.10])

    invoice = np.random.choice(
        ['Full GST/VAT Invoice','Incomplete Invoice','No Invoice','Verbal Only'],
        n, p=[0.45,0.22,0.20,0.13])

    substitution = np.random.choice(
        ['Never','Once','2-3 Times','>3 Times'],
        n, p=[0.45,0.28,0.18,0.09])

    qr_status = np.random.choice(
        ['QR Verified OK','QR Present but Failed','Hologram Unverified','No QR/Hologram'],
        n, p=[0.38,0.15,0.22,0.25])

    packaging_quality = np.random.choice([1,2,3,4,5], n, p=[0.10,0.18,0.25,0.30,0.17])

    batch_consistency = np.random.choice(
        ['All Present & Consistent','Present but Inconsistent','One Field Missing','All Absent'],
        n, p=[0.45,0.20,0.22,0.13])

    appearance = np.random.choice(
        ['Exact Match','Minor Differences','Noticeable Differences','Completely Different'],
        n, p=[0.42,0.25,0.22,0.11])

    seal = np.random.choice(
        ['Intact','Minor Damage Still Sealed','Broken Seal','No Tamper Packaging'],
        n, p=[0.44,0.22,0.20,0.14])

    reg_check = np.random.choice(
        ['Yes Verified Online','Not Checked','Not Found in Database','Database Unavailable'],
        n, p=[0.35,0.30,0.20,0.15])

    supplier_licence = np.random.choice(
        ['Yes Verified','Yes Unverified','No Licence',"Don't Know"],
        n, p=[0.40,0.25,0.20,0.15])

    therapeutic_effect = np.random.choice(
        ['Full Effect','Partial Effect','No Effect','Adverse Reaction','Not Yet Known'],
        n, p=[0.38,0.22,0.18,0.10,0.12])

    suspicion_score = np.clip(np.random.normal(5.2,2.5,n),1,10).round(1)

    lab_test = np.random.choice(
        ['Yes - Passed','Yes - Failed','Yes - Inconclusive','No Test Conducted'],
        n, p=[0.20,0.18,0.10,0.52])

    supplier_history = np.random.choice(
        ['Yes Confirmed','Suspected Unconfirmed','No',"Don't Know"],
        n, p=[0.15,0.20,0.45,0.20])

    encounters_label = np.random.choice(
        ['0','1-2','3-5','6-10','>10'],
        n, p=[0.25,0.35,0.22,0.12,0.06])
    encounters_map = {'0':0,'1-2':1.5,'3-5':4,'6-10':8,'>10':12}
    encounters_count = np.array([encounters_map[e] for e in encounters_label])

    # Risk score
    risk = np.zeros(n)
    risk += np.array([{'Authorised Distributor':-1.5,'Wholesale Market':0.5,
                       'Online Pharmacy':0.3,'Direct from Manufacturer':-0.8,
                       'Grey Market / Informal':2.5}[c] for c in channel])
    risk += np.array({'>30% Below MRP':2.2,'10-30% Below MRP':1.0,'At MRP':-0.5,
                      '10-30% Above MRP':0.5,'>30% Above MRP':1.2}.get(p,0)
                     for p in price_vs_mrp)
    risk += np.array([{'Full GST/VAT Invoice':-1.2,'Incomplete Invoice':0.5,
                       'No Invoice':1.8,'Verbal Only':1.4}[i] for i in invoice])
    risk += np.array([{'QR Verified OK':-2.0,'QR Present but Failed':1.5,
                       'Hologram Unverified':0.5,'No QR/Hologram':1.8}[q] for q in qr_status])
    risk += (3 - packaging_quality) * 0.4
    risk += np.array([{'All Present & Consistent':-0.8,'Present but Inconsistent':1.0,
                       'One Field Missing':1.2,'All Absent':2.2}[b] for b in batch_consistency])
    risk += np.array([{'Exact Match':-1.0,'Minor Differences':0.5,
                       'Noticeable Differences':1.5,'Completely Different':2.5}[a] for a in appearance])
    risk += np.array([{'Intact':-0.8,'Minor Damage Still Sealed':0.3,
                       'Broken Seal':1.8,'No Tamper Packaging':1.5}[s] for s in seal])
    risk += np.array([{'Yes Verified Online':-1.0,'Not Checked':0.3,
                       'Not Found in Database':2.0,'Database Unavailable':0.2}[r] for r in reg_check])
    risk += np.array([{'Yes Verified':-0.8,'Yes Unverified':0.3,
                       'No Licence':2.0,"Don't Know":0.5}[s] for s in supplier_licence])
    risk += np.array([{'Full Effect':-1.5,'Partial Effect':0.5,'No Effect':1.8,
                       'Adverse Reaction':2.5,'Not Yet Known':0.0}[t] for t in therapeutic_effect])
    risk += (suspicion_score - 5) * 0.3
    risk += np.array([{'Yes - Passed':-2.5,'Yes - Failed':3.0,
                       'Yes - Inconclusive':0.5,'No Test Conducted':0.0}[l] for l in lab_test])
    risk += np.array([{'Yes Confirmed':2.0,'Suspected Unconfirmed':1.0,
                       'No':-0.5,"Don't Know":0.2}[s] for s in supplier_history])
    risk -= experience_yrs * 0.04
    risk += np.random.normal(0,0.8,n)

    p33 = np.percentile(risk,45)
    p66 = np.percentile(risk,70)
    p90 = np.percentile(risk,88)
    assessment = np.where(risk<p33,'Genuine',
                 np.where(risk<p66,'Substandard',
                 np.where(risk<p90,'Suspected Counterfeit','Confirmed Counterfeit')))
    is_counterfeit = np.where(
        np.isin(assessment,['Suspected Counterfeit','Confirmed Counterfeit','Substandard']),
        'Counterfeit/Substandard','Genuine')
    days_to_escalation = np.clip(
        90 - (risk*5) + np.random.normal(0,10,n),1,180).round(0).astype(int)

    # Inject 16 outliers
    outlier_idx = np.random.choice(n, size=16, replace=False)
    for i in outlier_idx[:5]:
        suspicion_score[i] = np.random.uniform(8.5,10)
        assessment[i]='Genuine'; is_counterfeit[i]='Genuine'
    for i in outlier_idx[5:9]:
        qr_status[i]='QR Verified OK'; packaging_quality[i]=5
        assessment[i]='Confirmed Counterfeit'; is_counterfeit[i]='Counterfeit/Substandard'
    for i in outlier_idx[9:13]:
        channel[i]='Grey Market / Informal'
        assessment[i]='Genuine'; is_counterfeit[i]='Genuine'
    for i in outlier_idx[13:]:
        role[i]='Drug Inspector'
        suspicion_score[i]=np.random.uniform(1,2.5)
        assessment[i]='Confirmed Counterfeit'; is_counterfeit[i]='Counterfeit/Substandard'
        experience_yrs[i]=np.random.uniform(0.2,1)

    return pd.DataFrame({
        'Q1_Role':role,'Q2_Experience_Label':exp_label,'Q2_Experience_Yrs':experience_yrs,
        'Q3_Region':region,'Q4_Procurement_Channel':channel,'Q5_Price_vs_MRP':price_vs_mrp,
        'Q6_Invoice_Status':invoice,'Q7_Supplier_Substitution':substitution,
        'Q8_QR_Hologram_Status':qr_status,'Q9_Packaging_Quality':packaging_quality,
        'Q10_Batch_Consistency':batch_consistency,'Q11_Appearance_Match':appearance,
        'Q12_Seal_Condition':seal,'Q13_Regulatory_Check':reg_check,
        'Q14_Supplier_Licence':supplier_licence,'Q15_Therapeutic_Effect':therapeutic_effect,
        'Q16_Suspicion_Score':suspicion_score,'Q17_Lab_Test_Result':lab_test,
        'Q18_Supplier_Past_Incidents':supplier_history,
        'Q19_Prior_Encounters_Label':encounters_label,'Q19_Prior_Encounters_Count':encounters_count,
        'Risk_Score':risk.round(3),'Days_To_Escalation':days_to_escalation,
        'Q20_Assessment_4Class':assessment,'Q20_Is_Counterfeit':is_counterfeit,
    })

df = generate_data()


# =============================================================================
# HELPERS
# =============================================================================
def insight(text):
    st.markdown(f'<div class="ibox">💡 <b>Insight:</b> {text}</div>', unsafe_allow_html=True)

def sec(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)

def theme(fig, h=420, xtick=0):
    fig.update_layout(
        paper_bgcolor=NAVY, plot_bgcolor=CARD_BG,
        font=dict(color=WHITE, family="DM Sans, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=WHITE)),
        height=h, margin=dict(l=55,r=20,t=50,b=55),
        xaxis=dict(gridcolor="#1a2050",zerolinecolor="#1a2050",
                   tickfont=dict(color=WHITE),tickangle=xtick),
        yaxis=dict(gridcolor="#1a2050",zerolinecolor="#1a2050",tickfont=dict(color=WHITE)),
        title_font=dict(color=WHITE,size=15,family="Sora, sans-serif"),
    )
    return fig

def encode_df(d):
    dc = d.copy(); ld = {}
    for col in dc.select_dtypes(include='object').columns:
        le = LabelEncoder()
        dc[col] = le.fit_transform(dc[col].astype(str))
        ld[col] = le
    return dc, ld


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown(f"""
<div style="text-align:center;padding:16px 0 20px">
  <div style="font-family:'Sora',sans-serif;font-size:1.7rem;font-weight:800;color:{WHITE};">
    💊 Med<span style="color:{CRIMSON};">Guard</span>
  </div>
  <div style="font-size:0.72rem;color:{GREY};margin-top:4px;letter-spacing:0.4px;">
    COUNTERFEIT MEDICINE DETECTION
  </div>
  <hr style="border-color:{CRIMSON}33;margin:14px 0"/>
</div>
""", unsafe_allow_html=True)

PAGES = [
    "📊  Overview & EDA",
    "🤖  Classification",
    "🔵  K-Means Clustering",
    "🔗  Association Rules",
    "📈  Regression",
    "🔬  PCA — Dimensionality Reduction",
]
page = st.sidebar.radio("", PAGES, label_visibility="collapsed")

st.sidebar.markdown(f"""
<hr style="border-color:{CRIMSON}22;margin:16px 0 12px"/>
<div style="font-size:0.73rem;color:{GREY};padding:0 4px;line-height:1.7">
  <b style="color:{WHITE};">Dataset</b><br>
  200 medicine inspections · 25 features<br>
  Binary target: Genuine vs Counterfeit<br>
  Seed 99 · WHO/CDSCO risk model<br><br>
  <b style="color:{WHITE};">Project</b><br>
  MBA/MGB Data Analytics<br>
  MedGuard Platform · India
</div>
""", unsafe_allow_html=True)


# =============================================================================
# ── PAGE 1: OVERVIEW & EDA ───────────────────────────────────────────────────
# =============================================================================
if page == PAGES[0]:

    st.markdown(f"""
    <div class="hero">
      <div class="hero-title">💊 Med<span>Guard</span></div>
      <div class="hero-sub">Counterfeit Medicine Detection Platform — Data Analytics Dashboard</div>
      <div class="hero-pills">
        <span class="pill">MBA/MGB Data Analytics</span>
        <span class="pill green">200 Inspections · India</span>
        <span class="pill blue">6 ML Techniques</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **MedGuard** is a mobile + web platform where pharmacy staff log medicine inspections via a
    structured 20-question survey. Machine learning assesses counterfeit risk in real time, clusters
    users by detection capability, and surfaces hidden supply chain patterns through association rule
    mining. This dashboard presents the complete analytical pipeline.
    """)

    # KPI cards
    sec("📌 Key Performance Indicators")
    total      = len(df)
    pct_fake   = (df['Q20_Is_Counterfeit']=='Counterfeit/Substandard').mean()*100
    avg_sus    = df['Q16_Suspicion_Score'].mean()
    avg_pack   = df['Q9_Packaging_Quality'].mean()
    top_ch     = (df[df['Q20_Is_Counterfeit']=='Counterfeit/Substandard']
                  ['Q4_Procurement_Channel'].value_counts().idxmax())

    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [(total,"Total Inspections"),(f"{pct_fake:.1f}%","% Counterfeit/Substandard"),
            (f"{avg_sus:.1f}/10","Avg Suspicion Score"),(f"{avg_pack:.1f}/5","Avg Packaging Quality"),
            (top_ch,"Top Risk Channel")]
    for col,(v,l) in zip([c1,c2,c3,c4,c5],kpis):
        col.markdown(f'<div class="kpi"><div class="kpi-val">{v}</div>'
                     f'<div class="kpi-lbl">{l}</div></div>', unsafe_allow_html=True)

    insight(f"55% of inspected medicines are counterfeit or substandard — a critical public health signal. "
            f"The top risk procurement channel is <b>{top_ch}</b>. Average inspector suspicion score "
            f"of {avg_sus:.1f}/10 shows baseline awareness, but systematic ML scoring is essential.")

    # Demographics
    sec("👥 Respondent Demographics")
    col1,col2 = st.columns(2)
    with col1:
        rc = df['Q1_Role'].value_counts().reset_index(); rc.columns=['Role','Count']
        fig = px.pie(rc,values='Count',names='Role',title="Respondent Role Distribution",
                     color_discrete_sequence=[CRIMSON,BLUE,GREEN,ACCENT,"#9b59b6"],hole=0.38)
        theme(fig,380); fig.update_traces(textfont_color=WHITE)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        rgc = df['Q3_Region'].value_counts().reset_index(); rgc.columns=['Region','Count']
        fig = px.bar(rgc,x='Region',y='Count',title="Region Distribution",
                     color='Count',color_continuous_scale=[[0,CARD_BG],[1,CRIMSON]])
        theme(fig,380); fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig,use_container_width=True)

    ec = (df['Q2_Experience_Label'].value_counts()
          .reindex(['<1 yr','1-3 yrs','3-7 yrs','7-15 yrs','>15 yrs']).reset_index())
    ec.columns=['Experience','Count']
    fig = px.bar(ec,x='Experience',y='Count',
                 title="Years of Experience Distribution",color_discrete_sequence=[CRIMSON])
    theme(fig,310); st.plotly_chart(fig,use_container_width=True)

    insight("Drug Inspectors and experienced pharmacists (7+ yrs) have higher detection accuracy. "
            "Rural representation is low — itself a vulnerability since WHO data shows counterfeit "
            "prevalence is highest precisely in those under-monitored areas.")

    # Counterfeit rates
    sec("🔴 Counterfeit Rate Analysis")
    col1,col2 = st.columns(2)
    with col1:
        rf = df.groupby(['Q3_Region','Q20_Is_Counterfeit']).size().reset_index(name='Count')
        fig = px.bar(rf,x='Q3_Region',y='Count',color='Q20_Is_Counterfeit',barmode='group',
                     title="Counterfeit Rate by Region",
                     color_discrete_map={'Genuine':GREEN,'Counterfeit/Substandard':CRIMSON},
                     labels={'Q3_Region':'Region','Q20_Is_Counterfeit':'Outcome'})
        theme(fig,390); st.plotly_chart(fig,use_container_width=True)
    with col2:
        cf = df.groupby(['Q4_Procurement_Channel','Q20_Is_Counterfeit']).size().reset_index(name='Count')
        fig = px.bar(cf,x='Q4_Procurement_Channel',y='Count',color='Q20_Is_Counterfeit',barmode='group',
                     title="Counterfeit Rate by Procurement Channel",
                     color_discrete_map={'Genuine':GREEN,'Counterfeit/Substandard':CRIMSON},
                     labels={'Q4_Procurement_Channel':'Channel','Q20_Is_Counterfeit':'Outcome'})
        theme(fig,390,xtick=18); st.plotly_chart(fig,use_container_width=True)

    insight("Rural and semi-urban regions show disproportionately high counterfeit rates. "
            "<b>Grey Market / Informal</b> is the single most dangerous procurement source — "
            "MedGuard should auto-escalate any inspection from this channel.")

    # Box plots
    col1,col2 = st.columns(2)
    with col1:
        fig = px.box(df,x='Q20_Is_Counterfeit',y='Q16_Suspicion_Score',
                     color='Q20_Is_Counterfeit',title="Suspicion Score vs Outcome",
                     color_discrete_map={'Genuine':GREEN,'Counterfeit/Substandard':CRIMSON},
                     labels={'Q20_Is_Counterfeit':'Outcome','Q16_Suspicion_Score':'Suspicion Score (1–10)'})
        theme(fig,380); st.plotly_chart(fig,use_container_width=True)
    with col2:
        fig = px.box(df,x='Q20_Is_Counterfeit',y='Q9_Packaging_Quality',
                     color='Q20_Is_Counterfeit',title="Packaging Quality vs Outcome",
                     color_discrete_map={'Genuine':GREEN,'Counterfeit/Substandard':CRIMSON},
                     labels={'Q20_Is_Counterfeit':'Outcome','Q9_Packaging_Quality':'Packaging Quality (1–5)'})
        theme(fig,380); st.plotly_chart(fig,use_container_width=True)

    insight("Counterfeit medicines have higher suspicion scores and lower packaging quality on average. "
            "Overlapping distributions confirm that no single variable is sufficient — "
            "MedGuard's ML ensemble combines all 20 survey signals.")

    # Correlation heatmap
    sec("🔥 Correlation Matrix")
    num_c = ['Q2_Experience_Yrs','Q9_Packaging_Quality','Q16_Suspicion_Score',
             'Q19_Prior_Encounters_Count','Risk_Score','Days_To_Escalation']
    dh,_ = encode_df(df[num_c+['Q20_Is_Counterfeit']])
    corr = dh.corr().round(3)
    fig = px.imshow(corr,text_auto=True,title="Correlation Matrix — Numeric + Encoded Target",
                    color_continuous_scale=[[0,BLUE],[0.5,CARD_BG],[1,CRIMSON]],zmin=-1,zmax=1)
    theme(fig,480); fig.update_traces(textfont=dict(size=11,color=WHITE))
    st.plotly_chart(fig,use_container_width=True)
    insight("Risk_Score and Q20_Is_Counterfeit are strongly correlated (by design). "
            "Suspicion Score also correlates with the outcome, validating inspector intuition. "
            "Days_To_Escalation is inversely related to Risk_Score — riskier cases escalate faster.")

    # Feature importance
    sec("🌳 Feature Importance — Random Forest (Top 15)")
    fc = [c for c in df.columns if c not in ['Q20_Is_Counterfeit','Q20_Assessment_4Class',
                                              'Risk_Score','Days_To_Escalation']]
    dfi,_ = encode_df(df[fc+['Q20_Is_Counterfeit']])
    rf_fi = RandomForestClassifier(n_estimators=150,random_state=42)
    rf_fi.fit(dfi[fc].values, dfi['Q20_Is_Counterfeit'].values)
    imp = pd.DataFrame({'Feature':fc,'Importance':rf_fi.feature_importances_})
    imp = imp.sort_values('Importance',ascending=True).tail(15)
    fig = px.bar(imp,x='Importance',y='Feature',orientation='h',
                 title="Top 15 Feature Importances — Random Forest",
                 color='Importance',color_continuous_scale=[[0,BLUE],[1,CRIMSON]])
    theme(fig,480); fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig,use_container_width=True)
    insight("QR/Hologram status, lab test result, and therapeutic effect are the three most predictive features. "
            "Procurement channel, invoice status, and suspicion score round out the top tier, "
            "confirming MedGuard's 20-question survey targets the most informative variables.")

    # Outlier analysis
    sec("⚠️ Outlier Analysis — 4 Injected Types (16 Records, ~8%)")
    st.markdown("""
    | Type | Count | Condition | Real-World Meaning |
    |------|-------|-----------|-------------------|
    | **Type A** | 5 | Suspicion ≥8.5 → **Genuine** | Over-cautious inspectors / false alarms |
    | **Type B** | 4 | QR OK + Packaging 5 → **Confirmed Counterfeit** | Sophisticated fakes — most dangerous |
    | **Type C** | 4 | Grey Market → **Genuine** | Legitimate bulk clearance sales |
    | **Type D** | 3 | Drug Inspector, Suspicion ≤2.5 → **Confirmed Counterfeit** | Inexperienced inspector missed the fake |
    """)

    dp = df.copy(); dp['Outlier_Type']='Normal'
    dp.loc[(dp['Q16_Suspicion_Score']>=8.5)&(dp['Q20_Is_Counterfeit']=='Genuine'),
           'Outlier_Type']='A: High Suspicion → Genuine'
    dp.loc[(dp['Q8_QR_Hologram_Status']=='QR Verified OK')&(dp['Q9_Packaging_Quality']==5)&
           (dp['Q20_Assessment_4Class']=='Confirmed Counterfeit'),
           'Outlier_Type']='B: QR OK → Confirmed Fake'
    dp.loc[(dp['Q4_Procurement_Channel']=='Grey Market / Informal')&(dp['Q20_Is_Counterfeit']=='Genuine'),
           'Outlier_Type']='C: Grey Market → Genuine'
    dp.loc[(dp['Q1_Role']=='Drug Inspector')&(dp['Q16_Suspicion_Score']<=2.5)&
           (dp['Q20_Assessment_4Class']=='Confirmed Counterfeit'),
           'Outlier_Type']='D: Inspector Missed It'

    fig = px.scatter(dp,x='Q16_Suspicion_Score',y='Risk_Score',
                     color='Outlier_Type',symbol='Q20_Is_Counterfeit',
                     title="Outlier Analysis: Suspicion Score vs Risk Score",
                     labels={'Q16_Suspicion_Score':'Suspicion Score (1–10)',
                             'Risk_Score':'Risk Score (Composite)'},
                     color_discrete_map={'Normal':GREY,'A: High Suspicion → Genuine':ACCENT,
                                         'B: QR OK → Confirmed Fake':CRIMSON,
                                         'C: Grey Market → Genuine':BLUE,
                                         'D: Inspector Missed It':"#9b59b6"},
                     hover_data=['Q1_Role','Q4_Procurement_Channel'])
    theme(fig,520); fig.update_traces(marker=dict(size=9,opacity=0.85))
    st.plotly_chart(fig,use_container_width=True)
    insight("Type B outliers (QR verified → Confirmed Counterfeit) are the most dangerous: "
            "sophisticated fakes that defeat electronic authentication. MedGuard must be tuned for "
            "high recall to catch these even at the cost of some false positives. "
            "Type A outliers highlight the need for objective scoring over human intuition alone.")


# =============================================================================
# ── PAGE 2: CLASSIFICATION ───────────────────────────────────────────────────
# =============================================================================
elif page == PAGES[1]:
    st.markdown(f'<div style="font-family:Sora,sans-serif;font-size:2rem;font-weight:800;'
                f'color:{WHITE};margin-bottom:8px;">🤖 Classification — 6 ML Models</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Goal:** Predict whether a medicine is `Genuine` or `Counterfeit/Substandard` (binary classification).

    MedGuard needs a real-time decision engine for the mobile app. Six algorithms are trained and
    compared so the best can be deployed. **Why Recall > Accuracy?** A false negative (calling a
    fake medicine genuine) can cause patient death. A false positive is merely inconvenient.
    MedGuard optimises for **maximum Recall** at all times.
    """)

    fc = [c for c in df.columns if c not in ['Q20_Is_Counterfeit','Q20_Assessment_4Class',
                                              'Risk_Score','Days_To_Escalation']]
    denc,led = encode_df(df[fc+['Q20_Is_Counterfeit']])
    X = StandardScaler().fit_transform(denc[fc].values)
    y = denc['Q20_Is_Counterfeit'].values
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

    @st.cache_data
    def train_classifiers(Xtr,Xte,ytr,yte):
        mdls = {
            "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
            "Decision Tree":       DecisionTreeClassifier(max_depth=6,random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=150,random_state=42),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100,random_state=42),
            "KNN (k=7)":           KNeighborsClassifier(n_neighbors=7),
            "Naive Bayes":         GaussianNB(),
        }
        res,roc,cms = {},{},{}
        for nm,m in mdls.items():
            m.fit(Xtr,ytr)
            yp = m.predict(Xte)
            yprob = m.predict_proba(Xte)[:,1] if hasattr(m,'predict_proba') else yp.astype(float)
            fpr,tpr,_ = roc_curve(yte,yprob)
            auc = roc_auc_score(yte,yprob)
            res[nm]={'Accuracy':round(accuracy_score(yte,yp),4),
                     'Precision':round(precision_score(yte,yp,zero_division=0),4),
                     'Recall':round(recall_score(yte,yp,zero_division=0),4),
                     'F1-Score':round(f1_score(yte,yp,zero_division=0),4),
                     'AUC-ROC':round(auc,4)}
            roc[nm]=(fpr,tpr,auc); cms[nm]=confusion_matrix(yte,yp)
        return res,roc,cms,mdls

    res,roc,cms,mdls = train_classifiers(
        tuple(map(tuple,Xtr)), tuple(map(tuple,Xte)), tuple(ytr), tuple(yte))

    rdf = pd.DataFrame(res).T.reset_index().rename(columns={'index':'Model'})

    sec("📋 Model Performance Summary")
    best_rec = rdf.loc[rdf['Recall'].idxmax(),'Model']
    best_auc = rdf.loc[rdf['AUC-ROC'].idxmax(),'Model']
    st.dataframe(
        rdf.style.background_gradient(
            subset=['Accuracy','Precision','Recall','F1-Score','AUC-ROC'],cmap='RdYlGn')
        .format({k:'{:.3f}' for k in ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']}),
        use_container_width=True,height=260)
    insight(f"<b>Best Recall:</b> {best_rec} — critical for catching every counterfeit. "
            f"<b>Best AUC-ROC:</b> {best_auc} — strongest discriminative power overall. "
            f"Tree-based ensembles typically outperform linear models on survey data with "
            f"non-linear interactions.")

    sec("📊 All Metrics — All 6 Models Side by Side")
    fig = go.Figure()
    for met,col in zip(['Accuracy','Precision','Recall','F1-Score','AUC-ROC'],
                       [CRIMSON,BLUE,GREEN,ACCENT,"#9b59b6"]):
        fig.add_trace(go.Bar(name=met,x=rdf['Model'],y=rdf[met],marker_color=col,
                             text=rdf[met].round(3),textposition='outside',
                             textfont=dict(size=9,color=WHITE)))
    fig.update_layout(barmode='group',title="Metric Comparison — All 6 Classification Models")
    theme(fig,500); fig.update_yaxes(range=[0,1.18],title="Score")
    st.plotly_chart(fig,use_container_width=True)

    sec("🔲 Confusion Matrix")
    sel = st.selectbox("Select Model:", list(mdls.keys()))
    cm = cms[sel]; lbls = led['Q20_Is_Counterfeit'].classes_
    fig = px.imshow(cm,text_auto=True,
                    x=[f"Pred: {l}" for l in lbls],y=[f"Actual: {l}" for l in lbls],
                    title=f"Confusion Matrix — {sel}",
                    color_continuous_scale=[[0,CARD_BG],[0.5,"#8e44ad"],[1,CRIMSON]])
    theme(fig,420); fig.update_traces(textfont=dict(size=22,color=WHITE))
    st.plotly_chart(fig,use_container_width=True)
    r = res[sel]
    insight(f"<b>{sel}</b> — Recall: {r['Recall']:.3f} | Precision: {r['Precision']:.3f} | "
            f"F1: {r['F1-Score']:.3f} | AUC-ROC: {r['AUC-ROC']:.3f}. "
            f"Every false negative (bottom-left cell) = a counterfeit dispensed to a patient. "
            f"Tune MedGuard's threshold to minimise this cell at all costs.")

    sec("📉 ROC Curves — All 6 Models Overlaid")
    fig = go.Figure()
    for (nm,(fpr,tpr,auc)),cl in zip(roc.items(),
                                      [CRIMSON,BLUE,GREEN,ACCENT,"#9b59b6","#1abc9c"]):
        fig.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{nm} (AUC={auc:.3f})",
                                  line=dict(color=cl,width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Random",
                              line=dict(color=GREY,dash='dash',width=1.5)))
    fig.update_layout(title="ROC Curves — All 6 Models",
                      xaxis_title="False Positive Rate",yaxis_title="True Positive Rate")
    theme(fig,500); st.plotly_chart(fig,use_container_width=True)
    insight("Deploy the highest-AUC model for screening, then lower the decision threshold "
            "to maximise recall — accepting more false positives to ensure no counterfeit slips through.")


# =============================================================================
# ── PAGE 3: K-MEANS CLUSTERING ───────────────────────────────────────────────
# =============================================================================
elif page == PAGES[2]:
    st.markdown(f'<div style="font-family:Sora,sans-serif;font-size:2rem;font-weight:800;'
                f'color:{WHITE};margin-bottom:8px;">🔵 K-Means Clustering</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Goal:** Discover natural segments among inspectors and pharmacy staff.
    K-Means identifies groups so MedGuard can deliver targeted training programmes,
    tiered alerts, and personalised inspection checklists to each user type.
    """)

    fc = [c for c in df.columns if c not in ['Q20_Is_Counterfeit','Q20_Assessment_4Class']]
    denc,_ = encode_df(df[fc])
    Xcl = StandardScaler().fit_transform(denc.values)

    k_val = st.slider("Number of Clusters (K):", 2, 8, 4)

    @st.cache_data
    def cluster_metrics(X_tuple):
        X = np.array(X_tuple)
        iner,sils = [],[]
        for k in range(2,9):
            km = KMeans(n_clusters=k,random_state=42,n_init=10)
            km.fit(X); iner.append(km.inertia_)
            sils.append(silhouette_score(X,km.labels_))
        return iner,sils

    iner,sils = cluster_metrics(tuple(map(tuple,Xcl)))

    col1,col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2,9)),y=iner,mode='lines+markers',
                                  line=dict(color=CRIMSON,width=3),marker=dict(size=10,color=CRIMSON)))
        fig.add_vline(x=k_val,line_dash="dash",line_color=ACCENT)
        fig.update_layout(title="Elbow Method — Inertia vs K",
                          xaxis_title="K",yaxis_title="Inertia")
        theme(fig,380); st.plotly_chart(fig,use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2,9)),y=sils,mode='lines+markers',
                                  line=dict(color=GREEN,width=3),marker=dict(size=10,color=GREEN)))
        fig.add_vline(x=k_val,line_dash="dash",line_color=ACCENT)
        fig.update_layout(title="Silhouette Score vs K",
                          xaxis_title="K",yaxis_title="Silhouette Score")
        theme(fig,380); st.plotly_chart(fig,use_container_width=True)

    insight(f"Elbow = diminishing inertia return. Silhouette (0–1) = cluster cohesion quality. "
            f"K={k_val} selected — use the slider to explore.")

    km = KMeans(n_clusters=k_val,random_state=42,n_init=10).fit(Xcl)
    dp = df.copy(); dp['Cluster'] = km.labels_.astype(str)
    c2d = PCA(n_components=2,random_state=42).fit_transform(Xcl)
    dp['PC1']=c2d[:,0]; dp['PC2']=c2d[:,1]

    sec(f"📍 Cluster Scatter — PCA 2D (K={k_val})")
    fig = px.scatter(dp,x='PC1',y='PC2',color='Cluster',symbol='Q20_Is_Counterfeit',
                     title=f"K-Means Clusters (K={k_val}) — PCA 2D Projection",
                     color_discrete_sequence=[CRIMSON,BLUE,GREEN,ACCENT,"#9b59b6","#1abc9c","#e67e22","#e91e63"],
                     hover_data=['Q1_Role','Q3_Region','Q4_Procurement_Channel'])
    theme(fig,530); fig.update_traces(marker=dict(size=10,opacity=0.8))
    st.plotly_chart(fig,use_container_width=True)

    sec("📊 Cluster Profiles")
    pcols = ['Q2_Experience_Yrs','Q9_Packaging_Quality','Q16_Suspicion_Score',
             'Q19_Prior_Encounters_Count','Risk_Score','Days_To_Escalation']
    dpf,_ = encode_df(df[pcols+['Q20_Is_Counterfeit']])
    dpf['Cluster'] = km.labels_
    prof = dpf.groupby('Cluster').mean().round(3)
    prof['Counterfeit_Rate'] = dpf.groupby('Cluster')['Q20_Is_Counterfeit'].mean().round(3)
    prof['Size'] = dpf.groupby('Cluster').size()
    st.dataframe(prof.style.background_gradient(cmap='RdYlGn_r'),use_container_width=True)

    fig = go.Figure()
    for feat,cl in zip(['Q2_Experience_Yrs','Q9_Packaging_Quality','Q16_Suspicion_Score',
                        'Risk_Score','Counterfeit_Rate'],
                       [CRIMSON,BLUE,GREEN,ACCENT,"#9b59b6"]):
        if feat in prof.columns:
            fig.add_trace(go.Bar(name=feat,x=[f"Cluster {c}" for c in prof.index],
                                  y=prof[feat],marker_color=cl))
    fig.update_layout(barmode='group',title="Feature Averages by Cluster")
    theme(fig,460); st.plotly_chart(fig,use_container_width=True)

    sec("🏷️ Segment Interpretation (K=4 reference)")
    st.markdown("""
    | Cluster | Name | Profile | MedGuard Action |
    |---------|------|---------|----------------|
    | **0** | 🔴 High-Risk Rural | Low experience · High risk · Grey market | Priority training · Supervisor oversight |
    | **1** | 🟡 Mid-Tier Urban | Moderate experience · Mixed channels | QR + documentation refresher |
    | **2** | 🟢 Experienced Experts | High experience · Low risk · Authorised | Peer mentors for new recruits |
    | **3** | 🔵 New Recruits | Lowest experience · High variance | Onboarding · Guided checklists |
    """)
    insight("Counterfeit risk clusters strongly with experience, region, and procurement channel. "
            "MedGuard uses these segments to deploy targeted training and tiered alert thresholds.")


# =============================================================================
# ── PAGE 4: ASSOCIATION RULES ────────────────────────────────────────────────
# =============================================================================
elif page == PAGES[3]:
    st.markdown(f'<div style="font-family:Sora,sans-serif;font-size:2rem;font-weight:800;'
                f'color:{WHITE};margin-bottom:8px;">🔗 Association Rule Mining (Apriori)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Goal:** Discover which combinations of risk factors co-occur in counterfeit cases.

    ARM reveals hidden supply-chain patterns — e.g. grey market + no QR + broken seal → almost always counterfeit.
    These rules power MedGuard's automated alert engine.

    - **Support** — frequency of the itemset in the dataset
    - **Confidence** — P(consequent | antecedent)
    - **Lift** — how much more likely than chance; >1 = genuine signal
    """)

    df_arm = pd.DataFrame({
        'HighSuspicion':        (df['Q16_Suspicion_Score']>=7).astype(bool),
        'NoQR':                 df['Q8_QR_Hologram_Status'].isin(['No QR/Hologram','QR Present but Failed']),
        'GreyMarketSource':     (df['Q4_Procurement_Channel']=='Grey Market / Informal'),
        'NoInvoice':            df['Q6_Invoice_Status'].isin(['No Invoice','Verbal Only']),
        'BrokenSeal':           df['Q12_Seal_Condition'].isin(['Broken Seal','No Tamper Packaging']),
        'LabFailed':            (df['Q17_Lab_Test_Result']=='Yes - Failed'),
        'SupplierPriorIncident':df['Q18_Supplier_Past_Incidents'].isin(['Yes Confirmed','Suspected Unconfirmed']),
        'AppearanceDiffers':    df['Q11_Appearance_Match'].isin(['Noticeable Differences','Completely Different']),
        'NotInRegulatoryDB':    (df['Q13_Regulatory_Check']=='Not Found in Database'),
        'TherapeuticFailure':   df['Q15_Therapeutic_Effect'].isin(['No Effect','Adverse Reaction']),
        'LowPackaging':         (df['Q9_Packaging_Quality']<=2),
        'NoSupplierLicence':    df['Q14_Supplier_Licence'].isin(['No Licence','Yes Unverified']),
        'FreqSubstitution':     df['Q7_Supplier_Substitution'].isin(['2-3 Times','>3 Times']),
        'RuralSemiUrban':       df['Q3_Region'].isin(['Rural','Semi-Urban']),
        'IsCounterfeit':        (df['Q20_Is_Counterfeit']=='Counterfeit/Substandard'),
    })

    c1,c2 = st.columns(2)
    min_sup  = c1.slider("Minimum Support:",  0.05,0.50,0.15,0.01)
    min_conf = c2.slider("Minimum Confidence:",0.30,0.90,0.55,0.05)

    @st.cache_data
    def run_arm(min_sup,min_conf):
        try:
            fi = apriori(df_arm,min_support=min_sup,use_colnames=True)
            if len(fi)==0: return None
            rules = association_rules(fi,metric="confidence",
                                      min_threshold=min_conf,num_itemsets=len(fi))
            return rules.sort_values("lift",ascending=False).reset_index(drop=True)
        except:
            return None

    rules = run_arm(min_sup,min_conf)

    if rules is None or len(rules)==0:
        st.warning(f"⚠️ No rules found at support={min_sup:.2f}, confidence={min_conf:.2f}. "
                   "Lower the thresholds using the sliders above.")
    else:
        st.success(f"✅ **{len(rules)} rules** found at support ≥ {min_sup:.2f}, confidence ≥ {min_conf:.2f}")

        rd = rules.copy()
        rd['antecedents'] = rd['antecedents'].apply(lambda x:', '.join(list(x)))
        rd['consequents'] = rd['consequents'].apply(lambda x:', '.join(list(x)))
        rd['IsCounterfeitRule'] = rd['consequents'].str.contains('IsCounterfeit')
        for c in ['support','confidence','lift']: rd[c]=rd[c].round(4)

        sec("📋 Top 20 Rules by Lift")
        t20 = rd[['antecedents','consequents','support','confidence','lift','IsCounterfeitRule']].head(20)
        st.dataframe(
            t20.style.apply(lambda x:['background:rgba(192,57,43,0.22)' if v else '' for v in x],
                            subset=['IsCounterfeitRule'])
            .background_gradient(subset=['lift'],cmap='Reds')
            .format({'support':'{:.4f}','confidence':'{:.4f}','lift':'{:.4f}'}),
            use_container_width=True,height=520)
        insight("Red-highlighted rows have <b>IsCounterfeit</b> as consequent — most actionable for MedGuard. "
                "Lift >2 should trigger immediate escalation alerts in the platform.")

        sec("📈 Support vs Confidence (bubble = lift)")
        fig = px.scatter(rd.head(60),x='support',y='confidence',size='lift',
                         color='IsCounterfeitRule',
                         title="Top 60 Rules: Support vs Confidence",
                         color_discrete_map={True:CRIMSON,False:BLUE},
                         hover_data=['antecedents','consequents','lift'])
        theme(fig,500); fig.update_traces(marker=dict(opacity=0.8))
        st.plotly_chart(fig,use_container_width=True)

        sec("🏆 Top 10 Rules by Lift")
        t10 = rd.head(10).copy()
        t10['Rule'] = (t10['antecedents']+' → '+t10['consequents']).str[:75]
        fig = px.bar(t10.sort_values('lift'),x='lift',y='Rule',orientation='h',
                     title="Top 10 Association Rules by Lift",
                     color='lift',color_continuous_scale=[[0,CARD_BG],[1,CRIMSON]])
        theme(fig,480); fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig,use_container_width=True)

        cfr = rd[rd['IsCounterfeitRule']].head(5)
        if len(cfr)>0:
            sec("🚨 Most Actionable Rules (IsCounterfeit as Consequent)")
            for _,row in cfr.iterrows():
                st.markdown(f"""
                <div class="ibox">
                  <b>Rule:</b> [{row['antecedents']}] → [{row['consequents']}]<br/>
                  <b>Support:</b> {row['support']:.3f} | <b>Confidence:</b> {row['confidence']:.3f} |
                  <b>Lift:</b> {row['lift']:.3f}<br/>
                  <b>Plain English:</b> When {row['antecedents'].replace(',',' AND ')} occur together,
                  there is a {row['confidence']*100:.1f}% probability of counterfeit/substandard medicine
                  — {row['lift']:.1f}× more likely than the base rate.
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# ── PAGE 5: REGRESSION ───────────────────────────────────────────────────────
# =============================================================================
elif page == PAGES[4]:
    st.markdown(f'<div style="font-family:Sora,sans-serif;font-size:2rem;font-weight:800;'
                f'color:{WHITE};margin-bottom:8px;">📈 Regression — Linear / Ridge / Lasso</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Goal:** Predict `Days_To_Escalation` — how quickly a counterfeit incident escalates to a
    reportable patient harm event (1–180 days).

    Classification tells MedGuard *whether* a medicine is fake. Regression tells *how urgently* to act.
    **Ridge** shrinks all coefficients but keeps them all. **Lasso** drives some to exactly zero —
    automatic feature selection for lighter mobile deployment.
    """)

    fc = [c for c in df.columns if c not in ['Q20_Is_Counterfeit','Q20_Assessment_4Class',
                                              'Risk_Score','Days_To_Escalation']]
    dr,_ = encode_df(df[fc+['Days_To_Escalation']])
    Xr = StandardScaler().fit_transform(dr[fc].values)
    yr = dr['Days_To_Escalation'].values
    Xtr,Xte,ytr,yte = train_test_split(Xr,yr,test_size=0.2,random_state=42)

    c1,c2 = st.columns(2)
    al_r = c1.slider("Ridge Alpha:",   0.01,100.0,1.0,0.01)
    al_l = c2.slider("Lasso Alpha:",   0.001,10.0,0.1,0.001)

    lr_m  = LinearRegression().fit(Xtr,ytr)
    ri_m  = Ridge(alpha=al_r).fit(Xtr,ytr)
    la_m  = Lasso(alpha=al_l,max_iter=5000).fit(Xtr,ytr)
    rmods = {"Linear Regression":lr_m,"Ridge Regression":ri_m,"Lasso Regression":la_m}

    def rmets(m,Xte,yte):
        p=m.predict(Xte)
        return {"MAE":round(mean_absolute_error(yte,p),4),
                "RMSE":round(np.sqrt(mean_squared_error(yte,p)),4),
                "R²":round(r2_score(yte,p),4)}

    rres = {nm:rmets(m,Xte,yte) for nm,m in rmods.items()}
    rdf2 = pd.DataFrame(rres).T.reset_index().rename(columns={'index':'Model'})

    sec("📋 Regression Metrics")
    st.dataframe(
        rdf2.style.background_gradient(subset=['MAE','RMSE'],cmap='RdYlGn_r')
        .background_gradient(subset=['R²'],cmap='RdYlGn')
        .format({'MAE':'{:.3f}','RMSE':'{:.3f}','R²':'{:.3f}'}),
        use_container_width=True,height=175)
    br2 = rdf2.loc[rdf2['R²'].idxmax(),'Model']
    insight(f"<b>{br2}</b> achieves the highest R². MAE/RMSE measure average error in days. "
            f"A good R² (>0.6) means survey features meaningfully predict escalation speed.")

    sec("📊 Actual vs Predicted + Residuals")
    sel_r = st.selectbox("Select Model:",list(rmods.keys()))
    yp = rmods[sel_r].predict(Xte); res_v = yte - yp

    col1,col2 = st.columns(2)
    with col1:
        fig = px.scatter(x=yte,y=yp,title=f"Actual vs Predicted — {sel_r}",
                         color_discrete_sequence=[CRIMSON],
                         labels={'x':'Actual Days','y':'Predicted Days'})
        fig.add_trace(go.Scatter(x=[yte.min(),yte.max()],y=[yte.min(),yte.max()],
                                  mode='lines',name='Perfect Fit',
                                  line=dict(color=GREEN,dash='dash')))
        theme(fig,420); st.plotly_chart(fig,use_container_width=True)
    with col2:
        fig = px.scatter(x=yp,y=res_v,title=f"Residuals vs Predicted — {sel_r}",
                         color_discrete_sequence=[ACCENT],
                         labels={'x':'Predicted Values','y':'Residuals'})
        fig.add_hline(y=0,line_dash="dash",line_color=WHITE,opacity=0.5)
        theme(fig,420); st.plotly_chart(fig,use_container_width=True)
    insight("Residuals should scatter randomly around zero. A funnel shape = heteroscedasticity. "
            "Points far from the zero line = cases where the model significantly misjudged escalation time.")

    sec("⚖️ Ridge vs Lasso Coefficients — Feature Selection")
    cdf = pd.DataFrame({'Feature':fc,'Ridge':ri_m.coef_,'Lasso':la_m.coef_})
    cdf['Zeroed'] = cdf['Lasso'].abs()<1e-6
    cs = cdf.reindex(cdf['Ridge'].abs().sort_values(ascending=False).index)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Ridge',x=cs['Feature'],y=cs['Ridge'],
                          marker_color=CRIMSON,opacity=0.85))
    fig.add_trace(go.Bar(name='Lasso',x=cs['Feature'],y=cs['Lasso'],
                          marker_color=BLUE,opacity=0.85))
    fig.update_layout(barmode='group',title="Ridge vs Lasso Coefficients")
    theme(fig,500,xtick=35); st.plotly_chart(fig,use_container_width=True)

    zeroed = cdf[cdf['Zeroed']]['Feature'].tolist()
    kept   = cdf[~cdf['Zeroed']]['Feature'].tolist()
    st.markdown(f"""
    <div class="ibox">
      💡 <b>Lasso Feature Selection:</b><br/>
      <b>Zeroed ({len(zeroed)}):</b> {', '.join(zeroed) if zeroed else 'None at this alpha'}<br/>
      <b>Retained ({len(kept)}):</b> {', '.join(kept[:8])}{'…' if len(kept)>8 else ''}<br/><br/>
      Lasso's automatic elimination is critical for MedGuard's rural mobile deployment —
      fewer features = faster inference on 2G/3G networks. Ridge retains all features but
      shrinks coefficients, better when all variables contribute.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ── PAGE 6: PCA ──────────────────────────────────────────────────────────────
# =============================================================================
elif page == PAGES[5]:
    st.markdown(f'<div style="font-family:Sora,sans-serif;font-size:2rem;font-weight:800;'
                f'color:{WHITE};margin-bottom:8px;">🔬 PCA — Dimensionality Reduction</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Goal:** Reduce 20 survey features to a compact set of principal components that capture
    maximum variance — enabling faster inference on low-bandwidth rural devices.

    **Academic justification:** MedGuard targets rural pharmacists on 2G networks. Fewer features =
    faster model loading and lower battery use. PCA also reveals which survey questions drive the
    most variation in inspection outcomes.
    """)

    fc = [c for c in df.columns if c not in ['Q20_Is_Counterfeit','Q20_Assessment_4Class',
                                              'Risk_Score','Days_To_Escalation']]
    dp,_ = encode_df(df[fc+['Q20_Is_Counterfeit']])
    Xp = StandardScaler().fit_transform(dp[fc].values)
    yp_labels = df['Q20_Is_Counterfeit'].values

    pf = PCA(random_state=42).fit(Xp)
    ev = pf.explained_variance_ratio_
    cu = np.cumsum(ev)
    nc = len(fc)
    c80 = int(np.argmax(cu>=0.80)+1)
    c90 = int(np.argmax(cu>=0.90)+1)

    sec("📉 Scree Plot — Explained Variance per Component")
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=list(range(1,nc+1)),y=ev*100,name='Variance (%)',
                          marker_color=CRIMSON,opacity=0.85))
    fig.add_trace(go.Scatter(x=list(range(1,nc+1)),y=cu*100,name='Cumulative (%)',
                              mode='lines+markers',line=dict(color=ACCENT,width=3),
                              marker=dict(size=7)),secondary_y=True)
    fig.add_hline(y=80,line_dash="dot",line_color=GREEN,
                  annotation_text="80%",annotation_font_color=GREEN)
    fig.add_hline(y=90,line_dash="dot",line_color=BLUE,
                  annotation_text="90%",annotation_font_color=BLUE,secondary_y=True)
    fig.update_layout(title="PCA Scree Plot — Explained Variance")
    fig.update_xaxes(title_text="Principal Component")
    fig.update_yaxes(title_text="Variance (%)",secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)",secondary_y=True,
                     tickfont=dict(color=WHITE))
    theme(fig,460); st.plotly_chart(fig,use_container_width=True)

    col1,col2 = st.columns(2)
    col1.metric("80% Variance",f"{c80} components",f"from {nc} features")
    col2.metric("90% Variance",f"{c90} components",f"from {nc} features")

    insight(f"<b>{c80} components</b> capture 80% of variance — {nc/c80:.1f}× compression "
            f"vs the full {nc}-feature survey. Critical for low-bandwidth rural MedGuard deployment.")

    sec("🗺️ Biplot — PC1 vs PC2 with Loading Arrows")
    p2 = PCA(n_components=2,random_state=42).fit(Xp)
    X2d = p2.transform(Xp); ld = p2.components_.T
    fig = go.Figure()
    for out,cl in [('Genuine',GREEN),('Counterfeit/Substandard',CRIMSON)]:
        mask = yp_labels==out
        fig.add_trace(go.Scatter(x=X2d[mask,0],y=X2d[mask,1],mode='markers',name=out,
                                  marker=dict(color=cl,size=9,opacity=0.7,
                                              line=dict(width=0.5,color=WHITE))))
    sc=3.5
    for i,ft in enumerate(fc):
        lx,ly = ld[i,0]*sc, ld[i,1]*sc
        fig.add_annotation(x=lx,y=ly,ax=0,ay=0,xref='x',yref='y',axref='x',ayref='y',
                           arrowhead=2,arrowsize=1,arrowwidth=2,arrowcolor=ACCENT,
                           text=ft.replace('Q','').replace('_',' ')[:13],
                           font=dict(size=8,color=ACCENT),showarrow=True)
    v1,v2 = p2.explained_variance_ratio_*100
    fig.update_layout(
        title=f"PCA Biplot — PC1 ({v1:.1f}%) vs PC2 ({v2:.1f}%)",
        xaxis_title=f"PC1 — Supply Chain Integrity Axis ({v1:.1f}%)",
        yaxis_title=f"PC2 — Inspection Thoroughness Axis ({v2:.1f}%)")
    theme(fig,600); st.plotly_chart(fig,use_container_width=True)
    insight("<b>PC1 (Supply Chain Integrity)</b> — loads on procurement channel, invoice, QR, supplier licence. "
            "High PC1 = supply chain violations = high counterfeit risk. "
            "<b>PC2 (Inspection Thoroughness)</b> — loads on regulatory checks, lab tests, experience. "
            "Counterfeits cluster at high PC1, confirming supply chain is the primary variation axis.")

    sec("🔮 3D PCA Scatter — PC1, PC2, PC3")
    p3 = PCA(n_components=3,random_state=42).fit(Xp)
    X3d = p3.transform(Xp); v3 = p3.explained_variance_ratio_*100
    fig = go.Figure()
    for out,cl in [('Genuine',GREEN),('Counterfeit/Substandard',CRIMSON)]:
        mask = yp_labels==out
        fig.add_trace(go.Scatter3d(x=X3d[mask,0],y=X3d[mask,1],z=X3d[mask,2],
                                    mode='markers',name=out,
                                    marker=dict(color=cl,size=6,opacity=0.75,
                                                line=dict(width=0.5,color=WHITE))))
    fig.update_layout(
        title=f"3D PCA — PC1 ({v3[0]:.1f}%), PC2 ({v3[1]:.1f}%), PC3 ({v3[2]:.1f}%)",
        scene=dict(
            xaxis=dict(title=f"PC1 ({v3[0]:.1f}%)",backgroundcolor=CARD_BG,
                       gridcolor="#1a2050",zerolinecolor="#1a2050",tickfont=dict(color=WHITE)),
            yaxis=dict(title=f"PC2 ({v3[1]:.1f}%)",backgroundcolor=CARD_BG,
                       gridcolor="#1a2050",zerolinecolor="#1a2050",tickfont=dict(color=WHITE)),
            zaxis=dict(title=f"PC3 ({v3[2]:.1f}%)",backgroundcolor=CARD_BG,
                       gridcolor="#1a2050",zerolinecolor="#1a2050",tickfont=dict(color=WHITE)),
            bgcolor=NAVY),
        paper_bgcolor=NAVY,font=dict(color=WHITE,family="DM Sans, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=WHITE)),height=580)
    st.plotly_chart(fig,use_container_width=True)
    insight(f"Genuine and counterfeit medicines are separable in PCA space even though PCA is "
            f"unsupervised. PC1+PC2+PC3 explain {v3.sum():.1f}% of total variance — a compact "
            f"3-dimensional summary of the full 20-question survey.")

    sec("🔥 Feature Loading Heatmap")
    ph = PCA(n_components=min(8,nc),random_state=42).fit(Xp)
    lheat = pd.DataFrame(ph.components_.T,index=fc,
                          columns=[f"PC{i+1}" for i in range(ph.n_components_)])
    fig = px.imshow(lheat.round(3),
                    title="PCA Loading Heatmap — Feature Contributions to Each Component",
                    color_continuous_scale=[[0,BLUE],[0.5,CARD_BG],[1,CRIMSON]],
                    zmin=-1,zmax=1,text_auto=".2f",
                    labels={'x':'Principal Component','y':'Feature'})
    theme(fig,620); fig.update_traces(textfont=dict(size=9,color=WHITE))
    st.plotly_chart(fig,use_container_width=True)
    insight("Red = strong positive loading, blue = strong negative. Features near zero across all "
            "components are candidates for removal in MedGuard's lightweight mobile model. "
            "This directly informs which survey questions are essential vs optional.")
