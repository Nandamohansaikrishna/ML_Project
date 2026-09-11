import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(".env")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="💳",
    layout="wide"
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "dark" not in st.session_state: 
    st.session_state.dark = True
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = []
if "customer_idx" not in st.session_state: 
    st.session_state.customer_idx = 0

# ─────────────────────────────────────────────
# THEME COLORS
# ─────────────────────────────────────────────
def get_colors(dark):
    if dark:
        return {
            "bg": "#0d1117",
            "bg2": "#161b22",
            "card": "#1c2128",
            "border": "#30363d",
            "text": "#f0f6fc",
            "muted": "#8b949e",
            "blue": "#38bdf8",
            "blue2": "#58a6ff",
            "green": "#3fb950",
            "red": "#f85149",
            "yellow": "#e3b341",
            "sidebar": "#161b22",
        }
    return {
        "bg": "#ffffff",
        "bg2": "#f6f8fa",
        "card": "#ffffff",
        "border": "#d0d7de",
        "text": "#1f2328",
        "muted": "#656d76",
        "blue": "#0284c7",
        "blue2": "#0ea5e9",
        "green": "#1a7f37",
        "red": "#cf222e",
        "yellow": "#9a6700",
        "sidebar": "#f6f8fa",
    }

C = get_colors(st.session_state.dark)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {{ font-family: 'Inter', sans-serif !important; }}

.stApp {{
    background-color: {C["bg"]} !important;
}}

section[data-testid="stSidebar"] {{
    background-color: {C["sidebar"]} !important;
    border-right: 1px solid {C["border"]} !important;
}}

section[data-testid="stSidebar"] * {{
    color: {C["text"]} !important;
}}

.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}}

h1, h2, h3, h4, p, span, div, label {{
    color: {C["text"]} !important;
}}

.metric-card {{
    background: {C["card"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}}

.metric-label {{
    font-size: 0.72rem;
    font-weight: 600;
    color: {C["muted"]} !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 4px;
}}

.metric-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {C["blue"]} !important;
}}

.section-card {{
    background: {C["card"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 14px;
}}

.section-title {{
    font-size: 0.85rem;
    font-weight: 700;
    color: {C["blue"]} !important;
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.detail-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid {C["border"]};
    font-size: 0.86rem;
}}

.detail-label {{
    color: {C["muted"]} !important;
    font-weight: 500;
}}

.detail-value {{
    color: {C["text"]} !important;
    font-weight: 600;
}}

.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
}}

.badge-low    {{ background: rgba(63, 185, 80, 0.15); color: {C["green"]} !important; border: 1px solid {C["green"]}; }}
.badge-medium {{ background: rgba(227, 179, 65, 0.15); color: {C["yellow"]} !important; border: 1px solid {C["yellow"]}; }}
.badge-high   {{ background: rgba(248, 81, 73, 0.15); color: {C["red"]} !important; border: 1px solid {C["red"]}; }}
.badge-blue   {{ background: rgba(56, 189, 248, 0.15); color: {C["blue"]} !important; border: 1px solid {C["blue"]}; }}

.stButton > button {{
    background: {C["blue"]} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    width: 100% !important;
}}

.stButton > button:hover {{
    opacity: 0.85 !important;
}}

.stSelectbox > div > div,
div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {{
    background: {C["card"]} !important;
    color: {C["text"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 8px !important;
}}

div[data-testid="stChatMessage"] {{
    background: {C["card"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 8px !important;
}}

#MainMenu, footer {{ visibility: hidden; }}

::-webkit-scrollbar       {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {C["bg"]}; }}
::-webkit-scrollbar-thumb {{ background: rgba(56, 189, 248, 0.35); border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING — Graceful fallbacks
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        cleaned = pd.read_csv("data/credit_default_cleaned.csv")
        x_test = pd.read_csv("data/X_test.csv")
        y_test = pd.read_csv("data/y_test.csv").squeeze()
        return cleaned, x_test, y_test, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/best_model.pkl")
        return model, None
    except Exception as e:
        return None, str(e)

cleaned_df, x_test_df, y_test_df, data_err = load_data()
model, model_err = load_model()

# ─────────────────────────────────────────────
# EDUCATION & MARRIAGE MAPS
# ─────────────────────────────────────────────
EDU_MAP = {
    0: "Graduate",
    1: "University",
    2: "High School",
    3: "Others"
}
MAR_MAP = {
    0: "Married",
    1: "Single",
    2: "Others"
}
SEX_MAP = {
    0: "Female",
    1: "Male"
}
PAY_MAP = {
    -2: "No consumption",
    -1: "Paid on time",
     0: "Revolving credit",
     1: "1 month late",
     2: "2 months late",
     3: "3 months late",
     4: "4 months late",
     5: "5 months late",
     6: "6 months late",
     7: "7 months late",
     8: "8+ months late"
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_risk_badge(prob):
    if prob >= 0.65:
        return "HIGH", "badge-high"
    elif prob >= 0.35:
        return "MEDIUM", "badge-medium"
    return "LOW", "badge-low"


def predict_customer(idx):
    if model is None or x_test_df is None:
        return 0.5, "MEDIUM", "badge-medium"
    row = x_test_df.iloc[[idx]]
    proba = model.predict_proba(row)[0][1]
    label, badge = get_risk_badge(proba)
    return round(proba, 4), label, badge


def get_customer_display(idx):
    if cleaned_df is None or x_test_df is None:
        return {}

    offset = len(cleaned_df) - len(x_test_df)
    row = cleaned_df.iloc[offset + idx]
    return row


def build_groq_context(row, prob, risk_label):
    edu = EDU_MAP.get(int(row.get("EDUCATION", 0)), "Unknown")
    mar = MAR_MAP.get(int(row.get("MARRIAGE", 0)), "Unknown")
    sex = SEX_MAP.get(int(row.get("SEX", 0)), "Unknown")

    pay_sep = PAY_MAP.get(int(row.get("PAY_0", -1)), "Unknown")

    limit = row.get("LIMIT_BAL", 0)
    age = row.get("AGE", 0)
    bill1 = row.get("BILL_AMT1", 0)
    paid1 = row.get("PAY_AMT1", 0)

    util = round((bill1 / (limit + 1)) * 100, 1)
    pay_ratio = round((paid1 / (bill1 + 1)) * 100, 1)

    return f"""
Customer Profile:
- Age: {int(age)} years old
- Gender: {sex}
- Education: {edu}
- Marital Status: {mar}
- Credit Limit: NT$ {int(limit):,}
- Latest Bill Amount: NT$ {int(bill1):,}
- Latest Payment Made: NT$ {int(paid1):,}
- Credit Utilization: {util}%
- Payment-to-Bill Ratio: {pay_ratio}%
- Recent Payment Status: {pay_sep}

Risk Assessment:
- Default Probability: {round(prob * 100, 1)}%
- Risk Classification: {risk_label}

You are an expert loan advisor.
Analyze this customer profile and help the user
understand the risk and what can be improved.
Keep answers professional, clear, and under 4 sentences.
"""

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def benchmark_chart(row):
    limit = row.get("LIMIT_BAL", 1)
    bill1 = row.get("BILL_AMT1", 0)
    paid1 = row.get("PAY_AMT1", 0)
    pay_0 = row.get("PAY_0", -1)

    util = min(round((bill1 / (limit + 1)) * 100, 1), 150)
    pay_ratio = min(round((paid1 / (bill1 + 1)) * 100, 1), 150)
    delay_score = max(0, int(pay_0)) * 12.5

    customer_vals = [util, pay_ratio, delay_score]
    safe_targets = [30, 80, 0]
    labels = ["Utilization %", "Pay Ratio %", "Delay Score"]

    colors_cust = []
    for i, (cv, sv) in enumerate(zip(customer_vals, safe_targets)):
        if i == 0:
            colors_cust.append(C["red"] if cv > 50 else C["green"])
        elif i == 1:
            colors_cust.append(C["green"] if cv > 50 else C["red"])
        else:
            colors_cust.append(C["red"] if cv > 0 else C["green"])

    # Convert hex or dark/light muted color safely for Plotly
    safe_bar_color = "rgba(139, 148, 158, 0.35)" if st.session_state.dark else "rgba(101, 109, 118, 0.35)"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Your Customer",
        x=labels,
        y=customer_vals,
        marker_color=colors_cust,
        text=[f"{v:.0f}" for v in customer_vals],
        textposition="outside",
        textfont=dict(color=C["text"], size=11)
    ))

    fig.add_trace(go.Bar(
        name="Safe Target",
        x=labels,
        y=safe_targets,
        marker_color=[safe_bar_color] * 3,
        text=[f"{v:.0f}" for v in safe_targets],
        textposition="outside",
        textfont=dict(color=C["muted"], size=11)
    ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=C["muted"],
        font_family="Inter",
        height=260,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=C["text"])
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color=C["text"], size=11)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=C["border"],
            tickfont=dict(color=C["muted"], size=10)
        )
    )
    return fig


def probability_donut(prob):
    fig = go.Figure(go.Pie(
        values=[prob * 100, (1 - prob) * 100],
        labels=["Default Risk", "Safe"],
        hole=0.72,
        marker=dict(
            colors=[C["red"], C["green"]],
            line=dict(color=C["bg"], width=2)
        ),
        textinfo="none",
        hoverinfo="label+percent"
    ))

    fig.add_annotation(
        text=f"{round(prob*100,1)}%",
        x=0.5,
        y=0.55,
        font=dict(size=22, color=C["text"], family="Inter"),
        showarrow=False
    )
    fig.add_annotation(
        text="Default Risk",
        x=0.5,
        y=0.38,
        font=dict(size=10, color=C["muted"], family="Inter"),
        showarrow=False
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=180,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.markdown(f"""
    <div style='padding: 12px 0 8px;'>
        <div style='font-size:0.72rem; font-weight:700;
                    color:{C["muted"]}; text-transform:uppercase;
                    letter-spacing:0.08em; margin-bottom:12px;'>
            Settings
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "☀️ Light Mode" if st.session_state.dark else "🌙 Dark Mode",
        use_container_width=True
    ):
        st.session_state.dark = not st.session_state.dark
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.markdown(f"""
    <div style='font-size:0.72rem; font-weight:700;
                color:{C["muted"]}; text-transform:uppercase;
                letter-spacing:0.08em; margin-bottom:8px;'>
        Select Customer
    </div>
    """, unsafe_allow_html=True)

    if x_test_df is not None:
        total = len(x_test_df)

        customer_idx = st.selectbox(
            "Customer",
            options=list(range(total)),
            format_func=lambda i: f"Customer #{i + 1:04d}",
            index=st.session_state.customer_idx,
            label_visibility="collapsed"
        )
        st.session_state.customer_idx = customer_idx

        row = get_customer_display(customer_idx)
        prob, risk_label, risk_badge = predict_customer(customer_idx)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:0.72rem; font-weight:700;
                    color:{C["muted"]}; text-transform:uppercase;
                    letter-spacing:0.08em; margin-bottom:10px;'>
            Quick Profile
        </div>
        """, unsafe_allow_html=True)

        edu = EDU_MAP.get(int(row.get("EDUCATION", 0)), "—")
        mar = MAR_MAP.get(int(row.get("MARRIAGE", 0)), "—")
        sex = SEX_MAP.get(int(row.get("SEX", 0)), "—")
        age = int(row.get("AGE", 0))

        st.markdown(f"""
        <div style='display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px;'>
            <span class='badge badge-blue'>Age {age}</span>
            <span class='badge badge-blue'>{sex}</span>
            <span class='badge badge-blue'>{edu}</span>
            <span class='badge badge-blue'>{mar}</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        actual = int(y_test_df.iloc[customer_idx]) if y_test_df is not None else None
        if actual is not None:
            actual_text = "Defaulted" if actual == 1 else "Did Not Default"
            actual_color = C["red"] if actual == 1 else C["green"]
            st.markdown(f"""
            <div style='font-size:0.72rem; font-weight:700;
                        color:{C["muted"]}; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:6px;'>
                Actual Outcome
            </div>
            <div style='font-size:0.9rem; font-weight:600;
                        color:{actual_color};'>
                {"⚠️" if actual == 1 else "✅"} {actual_text}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error(f"Data load error: {data_err}")
        row = {}
        prob = 0.5
        risk_label = "UNKNOWN"
        risk_badge = "badge-medium"
        customer_idx = 0

    st.divider()

    st.markdown(f"""
    <div style='font-size:0.78rem; color:{C["muted"]}; line-height:2.1;'>
        <b style='color:{C["text"]};'>Stack</b><br>
        Random Forest<br>
        SHAP Explainability<br>
        FastAPI Backend<br>
        Groq LLM Advisor<br>
        AUC-ROC : 0.772
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style='margin-bottom: 20px;'>
    <h1 style='font-size:1.9rem; font-weight:800;
            color:{C["text"]} !important; margin:0; padding:0;'>
        💳 Credit Risk Scorer & Advisor
    </h1>
    <p style='font-size:0.88rem; color:{C["muted"]} !important;
            margin-top:4px;'>
        Select a customer from the sidebar to analyze their credit risk profile.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TOP METRIC STRIP — 4 Cards
# ─────────────────────────────────────────────
if row is not None and len(row) > 0:
    limit = int(row.get("LIMIT_BAL", 0))
    cust_id = customer_idx + 1
    prob_pct = round(prob * 100, 1)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Customer ID</div>
            <div class='metric-value'>#{cust_id:04d}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Credit Limit</div>
            <div class='metric-value'>NT${limit:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        color = C["red"] if prob_pct >= 65 else C["yellow"] if prob_pct >= 35 else C["green"]
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Default Probability</div>
            <div class='metric-value' style='color:{color} !important;'>
                {prob_pct}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        badge_colors = {
            "HIGH": C["red"],
            "MEDIUM": C["yellow"],
            "LOW": C["green"]
        }
        bc = badge_colors.get(risk_label, C["blue"])
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Risk Classification</div>
            <div style='margin-top:6px;'>
                <span style='background:rgba(56, 189, 248, 0.15); border:1px solid {bc};
                            color:{bc} !important; padding:5px 16px;
                            border-radius:20px; font-size:0.95rem;
                            font-weight:700;'>
                    {risk_label}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # MAIN TWO COLUMNS
    # ─────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    # ─────────────────────────────
    # LEFT COLUMN
    # ─────────────────────────────
    with left_col:

        # Customer Profile Card
        st.markdown(f"""
        <div class='section-card'>
            <div class='section-title'>👤 Customer Profile</div>
        """, unsafe_allow_html=True)

        edu = EDU_MAP.get(int(row.get("EDUCATION", 0)), "—")
        mar = MAR_MAP.get(int(row.get("MARRIAGE", 0)), "—")
        sex = SEX_MAP.get(int(row.get("SEX", 0)), "—")
        age = int(row.get("AGE", 0))
        pay_sep = PAY_MAP.get(int(row.get("PAY_0", -1)), "—")
        pay_aug = PAY_MAP.get(int(row.get("PAY_2", -1)), "—")
        bill_sep = int(row.get("BILL_AMT1", 0))
        bill_aug = int(row.get("BILL_AMT2", 0))
        paid_sep = int(row.get("PAY_AMT1", 0))
        paid_aug = int(row.get("PAY_AMT2", 0))

        profile_rows = [
            ("Age", f"{age} years"),
            ("Gender", sex),
            ("Education", edu),
            ("Marital Status", mar),
            ("Credit Limit", f"NT$ {limit:,}"),
            ("Bill (Sep)", f"NT$ {bill_sep:,}"),
            ("Bill (Aug)", f"NT$ {bill_aug:,}"),
            ("Paid (Sep)", f"NT$ {paid_sep:,}"),
            ("Paid (Aug)", f"NT$ {paid_aug:,}"),
            ("Payment Status (Sep)", pay_sep),
            ("Payment Status (Aug)", pay_aug),
        ]

        for label, value in profile_rows:
            st.markdown(f"""
            <div class='detail-row'>
                <span class='detail-label'>{label}</span>
                <span class='detail-value'>{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Risk Donut
        st.markdown(f"""
        <div class='section-card' style='text-align:center;'>
            <div class='section-title'>📊 Default Risk Score</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(
            probability_donut(prob),
            use_container_width=True,
            config={"displayModeBar": False}
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Benchmark Bar Chart
        st.markdown(f"""
        <div class='section-card'>
            <div class='section-title'>📈 Benchmark vs Healthy Targets</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(
            benchmark_chart(row),
            use_container_width=True,
            config={"displayModeBar": False}
        )

        # Improvement targets
        bill1 = row.get("BILL_AMT1", 0)
        paid1 = row.get("PAY_AMT1", 0)
        util = round((bill1 / (limit + 1)) * 100, 1)
        pr = round((paid1 / (bill1 + 1)) * 100, 1)

        st.markdown(f"""
        <div style='margin-top:12px; padding-top:12px;
                    border-top:1px solid {C["border"]};'>
            <div style='font-size:0.78rem; font-weight:700;
                        color:{C["muted"]}; text-transform:uppercase;
                        letter-spacing:0.06em; margin-bottom:10px;'>
                Improvement Targets
            </div>
        """, unsafe_allow_html=True)

        targets = [
            ("Credit Utilization",
            f"Currently {util}%",
            "Keep below 30%",
            util > 30),
            ("Payment Ratio",
            f"Currently {pr:.0f}%",
            "Pay at least 50% of bill",
            pr < 50),
            ("Payment Delays",
            f"Latest: {pay_sep}",
            "Always pay on time",
            "late" in pay_sep.lower()),
        ]

        for title, current, goal, is_bad in targets:
            dot = "🔴" if is_bad else "🟢"
            color = C["red"] if is_bad else C["green"]
            st.markdown(f"""
            <div style='padding:8px 0; border-bottom:1px solid {C["border"]};'>
                <div style='font-size:0.84rem; font-weight:600;
                            color:{C["text"]};'>
                    {dot} {title}
                </div>
                <div style='font-size:0.78rem; color:{C["muted"]};
                            margin-top:2px;'>
                    {current} → <b style='color:{color};'>{goal}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # ─────────────────────────────
    # RIGHT COLUMN — CHAT
    # ─────────────────────────────
    with right_col:

        st.markdown(f"""
        <div class='section-card'>
            <div class='section-title'>💬 AI Loan Advisor</div>
            <div style='font-size:0.82rem; color:{C["muted"]}; margin-bottom:14px;'>
                Ask anything about Customer #{cust_id:04d}.
                The advisor knows their full profile and risk score.
            </div>
        """, unsafe_allow_html=True)

        groq_context = build_groq_context(row, prob, risk_label)

        if not st.session_state.chat_history:
            st.markdown(f"""
            <div style='text-align:center; padding:30px 0;
                        color:{C["muted"]}; font-size:0.86rem;'>
                👋 Hi! I have analyzed Customer #{cust_id:04d}.<br>
                <span style='font-size:0.78rem;'>
                    Ask me: "Is this customer safe to approve?"<br>
                    or "What should this customer do to improve?"
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        st.markdown("</div>", unsafe_allow_html=True)

        # Chat input
        user_input = st.chat_input(
            f"Ask about Customer #{cust_id:04d}..."
        )

        if user_input:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            GROQ_API_KEY = os.getenv("GROQ_API_KEY")
            if not GROQ_API_KEY:
                reply = "Groq API key not found. Add GROQ_API_KEY to your .env file."
            else:
                try:
                    client = Groq(api_key=GROQ_API_KEY)

                    messages = [
                        {
                            "role": "system",
                            "content": groq_context
                        }
                    ]

                    for m in st.session_state.chat_history[-8:]:
                        messages.append({
                            "role": m["role"],
                            "content": m["content"]
                        })

                    response = client.chat.completions.create(
                        model="groq/compound",
                        messages=messages,
                        temperature=0.4,
                        max_tokens=400
                    )
                    reply = response.choices[0].message.content

                except Exception as e:
                    reply = f"Chat error: {str(e)}"

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": reply
            })
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; padding:28px 0 10px;
            font-size:0.74rem; color:{C["muted"]};
            border-top:1px solid {C["border"]}; margin-top:20px;'>
    💳 Credit Risk Scorer &nbsp;·&nbsp;
    Random Forest + SHAP + Groq AI &nbsp;·&nbsp;
    Built with FastAPI & Streamlit
</div>
""", unsafe_allow_html=True)