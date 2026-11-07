import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 55%, #e0f2fe 100%);
}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {background: #e8edff;}
.hero {
    padding: 28px 24px; border-radius: 20px; color: white; text-align: center;
    background: linear-gradient(120deg, #1d4ed8, #7c3aed);
    box-shadow: 0 8px 25px rgba(37,99,235,.20); margin-bottom: 20px;
}
.hero h1 {color:white !important; margin-bottom:8px;}
.panel {
    background: white; padding: 20px; border-radius: 16px;
    box-shadow: 0 4px 18px rgba(15,23,42,.08); margin-bottom: 16px;
}
div.stButton > button {
    border-radius: 10px; border: 0; background: #2563eb; color: white;
    font-weight: 600; min-height: 42px;
}
div.stButton > button:hover {background:#1d4ed8; color:white;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN STATE ----------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Demo credentials: change these before using publicly.
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "1234"

# ---------------- LOGIN PAGE ----------------
if not st.session_state.authenticated:
    st.markdown("""
    <div class="hero">
      <h1>🎓 Student Performance Predictor</h1>
      <p>Machine Learning • Student Academic Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.15, 1])
    with center:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 🔐 Welcome back")
        st.caption("Sign in to open your prediction dashboard.")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            if username == DEMO_USERNAME and password == DEMO_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")
    st.info("Demo login: username `admin` · password `1234`")
    st.stop()

# ---------------- DASHBOARD HEADER / THREE DOT MENU ----------------
with st.sidebar:
    st.markdown("## ⋮ Menu")
    st.caption("Student Performance")
    page = st.radio("Navigate", ["Dashboard", "Predict Marks", "Dataset", "About"], label_visibility="collapsed")
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

st.markdown("""
<div class="hero">
  <h1>🎓 Student Performance Dashboard</h1>
  <p>Explore student data and predict academic marks with Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------------- DATASET ----------------
st.sidebar.markdown("### 📁 Dataset")
uploaded = st.sidebar.file_uploader("Upload student CSV", type=["csv"])

@st.cache_data
def load_default_data():
    # Synthetic fallback dataset so the app can run even without a CSV.
    rng = np.random.default_rng(42)
    n = 500
    study_hours = rng.uniform(0, 10, n)
    attendance = rng.uniform(50, 100, n)
    previous_marks = rng.uniform(30, 100, n)
    sleep_hours = rng.uniform(4, 9, n)
    noise = rng.normal(0, 7, n)
    final_marks = np.clip(
        5 + study_hours * 4.2 + attendance * .25 + previous_marks * .35
        + sleep_hours * 1.2 + noise, 0, 100
    )
    return pd.DataFrame({
        "Study_Hours": study_hours.round(1),
        "Attendance": attendance.round(1),
        "Previous_Marks": previous_marks.round(1),
        "Sleep_Hours": sleep_hours.round(1),
        "Final_Marks": final_marks.round(1)
    })

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {len(df)} rows")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = load_default_data()
else:
    df = load_default_data()
    st.sidebar.info("Using demo synthetic data")

# ---------------- MODEL PREPARATION ----------------
# Select target and numeric features. Prefer Final_Marks if present.
target_candidates = ["Final_Marks", "final_marks", "G3", "Marks", "Score", "score"]
target = next((c for c in target_candidates if c in df.columns), None)

numeric_df = df.select_dtypes(include=np.number).copy()
if target is None or target not in numeric_df.columns:
    # If uploaded data has no known target, use the last numeric column as target.
    if numeric_df.shape[1] >= 2:
        target = numeric_df.columns[-1]
    else:
        target = None

if target is not None and numeric_df.shape[1] >= 2:
    model_data = numeric_df.dropna(subset=[target])
    features = [c for c in model_data.columns if c != target]
    X = model_data[features].fillna(model_data[features].median())
    y = model_data[target]
    if len(model_data) >= 10 and len(features) > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions) if len(y_test) > 1 else np.nan
    else:
        model = None
        features = []
        mae = np.nan
        r2 = np.nan
else:
    model = None
    features = []
    mae = np.nan
    r2 = np.nan

# ---------------- PAGES ----------------
if page == "Dashboard":
    st.markdown("### 📊 Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Student records", f"{len(df):,}")
    c2.metric("Numeric columns", numeric_df.shape[1])
    c3.metric("Prediction target", target if target else "Not found")
    if model is not None:
        c4, c5 = st.columns(2)
        c4.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        c5.metric("R² score", f"{r2:.3f}" if pd.notna(r2) else "N/A")
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    if target and target in df.columns:
        st.markdown("#### Marks distribution")
        st.bar_chart(df[target].dropna())

elif page == "Predict Marks":
    st.markdown("### 🤖 Predict student performance")
    if model is None:
        st.warning("Please upload a CSV with at least 10 rows and 2 numeric columns (one target column).")
    else:
        st.caption(f"Model target: **{target}**. Enter values for the numeric input features.")
        input_values = {}
        cols = st.columns(2)
        for i, feature in enumerate(features):
            series = X[feature]
            default = float(series.median())
            min_val = float(series.min())
            max_val = float(series.max())
            # Avoid identical min/max values in widgets.
            if min_val == max_val:
                max_val = min_val + 1.0
            with cols[i % 2]:
                input_values[feature] = st.number_input(
                    feature, min_value=min_val, max_value=max_val,
                    value=min(max(default, min_val), max_val), key=f"input_{feature}"
                )
        if st.button("✨ Predict Marks", use_container_width=True):
            new_data = pd.DataFrame([input_values], columns=features)
            predicted = float(model.predict(new_data)[0])
            st.success(f"Predicted {target}: {predicted:.2f}")
            st.metric("Predicted score", f"{predicted:.2f}")

elif page == "Dataset":
    st.markdown("### 📁 Full dataset")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download loaded dataset as CSV", df.to_csv(index=False), "student_dataset.csv", "text/csv")

elif page == "About":
    st.markdown("""
    ### ℹ️ About this project
    **Student Performance Prediction** uses a Random Forest regression model
    to estimate a numeric student score from other numeric dataset columns.

    - Upload your own CSV from the sidebar.
    - The app selects `Final_Marks`, `G3`, `Marks`, or `Score` as target if available.
    - Otherwise, it uses the last numeric column as target.
    - This demo login is not secure production authentication.
    """)
