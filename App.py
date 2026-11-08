import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="centered")

FEATURES = ["study_hours", "attendance", "previous_marks", "sleep_hours"]

# ---------------------------------------------------------------- DESIGN
PRIMARY = "#6C5CE7"
SECONDARY = "#00B894"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Poppins', sans-serif; }}
.stApp {{ background: linear-gradient(180deg, #F5F3FF 0%, #FFFFFF 45%); }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 720px; }}

/* Hero banner */
.hero {{
    background: linear-gradient(135deg, {PRIMARY} 0%, #A29BFE 55%, #74B9FF 100%);
    color: #fff; padding: 28px 24px; border-radius: 22px;
    box-shadow: 0 12px 30px rgba(108, 92, 231, .35); margin-bottom: 18px;
}}
.hero h1 {{ color: #fff; font-size: 1.7rem; font-weight: 700; margin: 0 0 6px 0; padding: 0; line-height: 1.25; }}
.hero p {{ margin: 0; opacity: .92; font-size: .9rem; }}
.badge {{
    display: inline-block; background: rgba(255,255,255,.22); color: #fff;
    padding: 4px 12px; border-radius: 999px; font-size: .72rem; margin-top: 12px; margin-right: 6px;
}}

/* Login card */
.login-card {{
    background: #fff; border-radius: 22px; padding: 26px 22px 8px 22px;
    box-shadow: 0 14px 40px rgba(108, 92, 231, .18); margin-top: 8px;
}}
.login-icon {{ font-size: 3rem; text-align: center; }}
.login-title {{ text-align: center; font-weight: 700; font-size: 1.6rem; color: #2D3436; margin: 0; }}
.login-sub {{ text-align: center; color: #636E72; font-size: .85rem; margin-bottom: 14px; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: none; }}
.stTabs [data-baseweb="tab"] {{
    background: #EFEDFF; border-radius: 12px; padding: 8px 14px; font-weight: 500; color: #4B4B6B;
}}
.stTabs [aria-selected="true"] {{ background: {PRIMARY}; color: #fff !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* Metric cards */
[data-testid="stMetric"] {{
    background: #fff; border-radius: 16px; padding: 14px 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,.06); border-left: 5px solid {PRIMARY};
}}
[data-testid="stMetricValue"] {{ color: {PRIMARY}; font-weight: 700; }}

/* Inputs & buttons */
.stButton > button, .stDownloadButton > button {{
    background: linear-gradient(135deg, {PRIMARY}, #A29BFE); color: #fff; border: none;
    border-radius: 12px; padding: 10px 22px; font-weight: 600; width: 100%;
    box-shadow: 0 6px 16px rgba(108, 92, 231, .35); transition: transform .15s;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ transform: translateY(-2px); color: #fff; }}
.stTextInput input {{ border-radius: 12px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 18px; background: #fff; }}

/* Result card */
.result {{
    border-radius: 20px; padding: 22px; text-align: center; color: #fff; margin-top: 14px;
    box-shadow: 0 10px 26px rgba(0,0,0,.15);
}}
.result .score {{ font-size: 2.8rem; font-weight: 700; line-height: 1.1; }}
.result .grade {{
    display: inline-block; background: rgba(255,255,255,.25); border-radius: 999px;
    padding: 4px 18px; font-weight: 600; margin-top: 8px;
}}
.section-title {{ font-weight: 600; color: #2D3436; margin: 18px 0 6px 0; font-size: 1.05rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- AUTH
def get_users() -> dict:
    try:
        return dict(st.secrets["users"])
    except Exception:
        return {"admin": "admin123"}  # demo only - change this!


def login_page():
    st.markdown(
        """
        <div class="login-card">
            <div class="login-icon">🎓</div>
            <p class="login-title">Welcome Back</p>
            <p class="login-sub">Sign in to the Student Performance Predictor</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    username = st.text_input("👤 Username", placeholder="Enter username")
    password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
    if st.button("Login"):
        users = get_users()
        if username in users and str(users[username]) == password:
            st.session_state["user"] = username
            st.rerun()
        else:
            st.error("Invalid username or password")


if "user" not in st.session_state:
    login_page()
    st.stop()


# ---------------------------------------------------------------- DATA + MODEL
@st.cache_data
def make_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    study = rng.uniform(0, 10, n).round(2)
    attendance = rng.integers(40, 101, n)
    previous = rng.integers(30, 101, n)
    sleep = rng.uniform(4, 9, n).round(1)
    final = 4.0 * study + 0.25 * attendance + 0.40 * previous + 1.0 * sleep + rng.normal(0, 4, n)
    return pd.DataFrame(
        {
            "study_hours": study,
            "attendance": attendance,
            "previous_marks": previous,
            "sleep_hours": sleep,
            "final_score": np.clip(final, 0, 100).round(1),
        }
    )


@st.cache_resource
def train_model(df: pd.DataFrame):
    X, y = df[FEATURES], df["final_score"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "R²": r2_score(y_te, pred),
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
    }
    return model, metrics, y_te.reset_index(drop=True), pd.Series(pred)


def grade(score: float):
    """Returns (letter, gradient colors)."""
    if score >= 85:
        return "A", "linear-gradient(135deg,#00B894,#55EFC4)"
    if score >= 70:
        return "B", "linear-gradient(135deg,#0984E3,#74B9FF)"
    if score >= 55:
        return "C", "linear-gradient(135deg,#F39C12,#FDCB6E)"
    if score >= 40:
        return "D", "linear-gradient(135deg,#E17055,#FAB1A0)"
    return "F", "linear-gradient(135deg,#D63031,#FF7675)"


df = make_data()
model, metrics, y_test, y_pred = train_model(df)

# ---------------------------------------------------------------- SIDEBAR
st.sidebar.markdown(f"### 👤 {st.session_state['user']}")
st.sidebar.caption("Signed in")
if st.sidebar.button("Logout"):
    del st.session_state["user"]
    st.rerun()

# ---------------------------------------------------------------- HERO
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Academic<br>Performance Prediction</h1>
        <p>Predict final scores using Machine Learning</p>
        <span class="badge">Scikit-learn</span><span class="badge">Streamlit</span>
        <span class="badge">Synthetic data</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_dash, tab_pred, tab_data, tab_model = st.tabs(
    ["📊 Dashboard", "🎯 Predict", "🗂️ Dataset", "📈 Model"]
)

# ---------------------------------------------------------------- DASHBOARD
with tab_dash:
    c1, c2 = st.columns(2)
    c1.metric("Students", len(df))
    c2.metric("Avg score", f"{df['final_score'].mean():.1f}")
    c3, c4 = st.columns(2)
    c3.metric("Avg study hrs", f"{df['study_hours'].mean():.1f}")
    c4.metric("Avg attendance", f"{df['attendance'].mean():.0f}%")

    st.markdown('<div class="section-title">Average score by study hours</div>', unsafe_allow_html=True)
    bins = pd.cut(df["study_hours"], bins=[0, 2, 4, 6, 8, 10], include_lowest=True)
    by_hours = df.groupby(bins, observed=True)["final_score"].mean()
    by_hours.index = by_hours.index.astype(str)
    st.bar_chart(by_hours, color=PRIMARY)

    st.markdown('<div class="section-title">Grade distribution</div>', unsafe_allow_html=True)
    grades = df["final_score"].apply(lambda s: grade(s)[0]).value_counts().reindex(list("ABCDF"), fill_value=0)
    st.bar_chart(grades, color=SECONDARY)

# ---------------------------------------------------------------- PREDICT
with tab_pred:
    st.markdown('<div class="section-title">Enter student details</div>', unsafe_allow_html=True)
    with st.container(border=True):
        study_hours = st.slider("📚 Study hours per day", 0.0, 10.0, 4.0, 0.25)
        attendance = st.slider("🏫 Attendance (%)", 0, 100, 80)
        previous_marks = st.slider("📝 Previous exam marks", 0, 100, 65)
        sleep_hours = st.slider("😴 Sleep hours per day", 4.0, 10.0, 7.0, 0.5)

    if st.button("🎯 Predict score"):
        row = pd.DataFrame([[study_hours, attendance, previous_marks, sleep_hours]], columns=FEATURES)
        score = float(np.clip(model.predict(row)[0], 0, 100))
        letter, color = grade(score)
        st.markdown(
            f"""
            <div class="result" style="background:{color}">
                <div>Predicted final score</div>
                <div class="score">{score:.1f}<span style="font-size:1.1rem"> / 100</span></div>
                <div class="grade">Grade {letter}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(int(score))
        if letter in ("D", "F"):
            st.warning("At risk: try more study time and better attendance.")
        elif letter in ("A", "B"):
            st.success("Great work! Keep up the current habits.")
        else:
            st.info("Decent. A bit more study time could raise the grade.")

# ---------------------------------------------------------------- DATASET
with tab_data:
    st.dataframe(df, use_container_width=True, height=350)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "student_data.csv", "text/csv")
    st.markdown('<div class="section-title">Summary statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

# ---------------------------------------------------------------- MODEL
with tab_model:
    st.caption("Random Forest Regressor | 80/20 train-test split")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{metrics['R²']:.3f}")
    m2.metric("MAE", f"{metrics['MAE']:.2f}")
    m3.metric("RMSE", f"{metrics['RMSE']:.2f}")

    st.markdown('<div class="section-title">Feature importance</div>', unsafe_allow_html=True)
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    st.bar_chart(imp, color=PRIMARY)

    st.markdown('<div class="section-title">Actual vs predicted</div>', unsafe_allow_html=True)
    st.scatter_chart(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}), x="Actual", y="Predicted", color=SECONDARY)

        
    