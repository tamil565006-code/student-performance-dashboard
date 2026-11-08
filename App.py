 import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Academic Performance Prediction", page_icon="🎓", layout="centered")

FEATURES = ["study_hours", "attendance", "previous_marks", "sleep_hours"]


# ---------------------------------------------------------------- AUTH
def get_users() -> dict:
    """Users from Streamlit secrets ([users] section) or a demo fallback."""
    try:
        return dict(st.secrets["users"])
    except Exception:
        return {"admin": "admin123"}  # demo only - change this!


def login_page():
    st.title("🔐 Login")
    st.caption("Sign in to use the Student Performance app.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login", type="primary"):
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
    final = (
        4.0 * study
        + 0.25 * attendance
        + 0.40 * previous
        + 1.0 * sleep
        + rng.normal(0, 4, n)
    )
    final = np.clip(final, 0, 100).round(1)
    return pd.DataFrame(
        {
            "study_hours": study,
            "attendance": attendance,
            "previous_marks": previous,
            "sleep_hours": sleep,
            "final_score": final,
        }
    )


@st.cache_resource
def train_model(df: pd.DataFrame):
    X, y = df[FEATURES], df["final_score"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "R²": r2_score(y_te, pred),
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
    }
    return model, metrics, y_te.reset_index(drop=True), pd.Series(pred)


def grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


df = make_data()
model, metrics, y_test, y_pred = train_model(df)

# ---------------------------------------------------------------- SIDEBAR
st.sidebar.write(f"👤 Signed in as **{st.session_state['user']}**")
if st.sidebar.button("Logout"):
    del st.session_state["user"]
    st.rerun()

# ---------------------------------------------------------------- HEADER
st.title("🎓 Student Academic Performance Prediction")
st.caption("Machine Learning project | Scikit-learn + Streamlit")
st.caption("Dataset: simulated (synthetic) student data for demonstration.")

tab_dash, tab_pred, tab_data, tab_model = st.tabs(
    ["📊 Dashboard", "🎯 Predict", "🗂️ Dataset", "📈 Model Performance"]
)

# ---------------------------------------------------------------- DASHBOARD
with tab_dash:
    st.header("Overview")
    c1, c2 = st.columns(2)
    c1.metric("Students", len(df))
    c2.metric("Average score", f"{df['final_score'].mean():.1f}")
    c3, c4 = st.columns(2)
    c3.metric("Avg study hrs/day", f"{df['study_hours'].mean():.1f}")
    c4.metric("Avg attendance", f"{df['attendance'].mean():.0f}%")

    st.subheader("Average score by study hours")
    bins = pd.cut(df["study_hours"], bins=[0, 2, 4, 6, 8, 10], include_lowest=True)
    by_hours = df.groupby(bins, observed=True)["final_score"].mean()
    by_hours.index = by_hours.index.astype(str)
    st.bar_chart(by_hours)

    st.subheader("Grade distribution")
    grades = df["final_score"].apply(grade).value_counts().reindex(list("ABCDF"), fill_value=0)
    st.bar_chart(grades)

# ---------------------------------------------------------------- PREDICT
with tab_pred:
    st.header("Enter student details")
    with st.container(border=True):
        study_hours = st.slider("Study hours per day", 0.0, 10.0, 4.0, 0.25)
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        previous_marks = st.slider("Previous exam marks", 0, 100, 65)
        sleep_hours = st.slider("Sleep hours per day", 4.0, 10.0, 7.0, 0.5)

    if st.button("🎯 Predict score", type="primary"):
        row = pd.DataFrame(
            [[study_hours, attendance, previous_marks, sleep_hours]], columns=FEATURES
        )
        score = float(np.clip(model.predict(row)[0], 0, 100))
        g = grade(score)
        st.success(f"Predicted final score: **{score:.1f} / 100**  |  Grade: **{g}**")
        st.progress(int(score))
        if g in ("D", "F"):
            st.warning("At risk: consider more study time and better attendance.")
        elif g in ("A", "B"):
            st.info("Great! Keep up the current habits.")

# ---------------------------------------------------------------- DATASET
with tab_data:
    st.header("Dataset")
    st.dataframe(df, use_container_width=True, height=350)
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="student_data.csv",
        mime="text/csv",
    )
    st.subheader("Summary statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ---------------------------------------------------------------- MODEL PERFORMANCE
with tab_model:
    st.header("Model performance")
    st.caption("Random Forest Regressor | 80/20 train-test split")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{metrics['R²']:.3f}")
    m2.metric("MAE", f"{metrics['MAE']:.2f}")
    m3.metric("RMSE", f"{metrics['RMSE']:.2f}")

    st.subheader("Feature importance")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    st.bar_chart(imp)

    st.subheader("Actual vs predicted")
    st.scatter_chart(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}), x="Actual", y="Predicted")



                         