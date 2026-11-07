import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Performance Predictor",
                   page_icon="🎓", layout="wide")

# Put your CSV in the GitHub repo with one of these names,
# or upload it from the sidebar while the app is running.
DEFAULT_FILES = ["dataset.csv", "student_data.csv", "data.csv"]
TARGET_HINTS = ["final_marks", "final", "g3", "marks", "score", "result"]


def read_csv(src):
    # sep=None lets pandas detect comma, semicolon or tab files
    return pd.read_csv(src, sep=None, engine="python")


def load_data():
    up = st.sidebar.file_uploader("Upload student dataset (CSV)", type="csv")
    if up is not None:
        return read_csv(up), up.name
    for f in DEFAULT_FILES:
        if os.path.exists(f):
            return read_csv(f), f
    return None, None


def get_category(marks, top):
    pct = marks / top * 100
    if pct >= 75:
        return "Excellent"
    if pct >= 60:
        return "Good"
    if pct >= 50:
        return "Average"
    return "Needs Improvement"


@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200,
                                               random_state=42),
    }
    rows, preds = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        preds[name] = p
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, p),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, p))),
            "R2": r2_score(y_test, p),
        })
    results = pd.DataFrame(rows)
    best_name = results.sort_values("R2", ascending=False).iloc[0]["Model"]
    importance = pd.Series(models["Random Forest"].feature_importances_,
                           index=X.columns).sort_values(ascending=False)
    compare = pd.DataFrame({"Actual": y_test.values,
                            "Predicted": preds[best_name]})
    return {
        "best_model": models[best_name],
        "best_name": best_name,
        "results": results,
        "importance": importance,
        "compare": compare,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ---------- Load data ----------
st.title("🎓 Student Academic Performance Prediction")
st.caption("Machine Learning project | Scikit-learn + Streamlit")

raw, source = load_data()
if raw is None:
    st.info("No dataset found. Upload a CSV from the sidebar, or add a file "
            "named **dataset.csv** to your GitHub repository.")
    st.stop()

raw.columns = [str(c).strip() for c in raw.columns]
raw = raw.dropna(how="all").drop_duplicates().reset_index(drop=True)

num_cols = list(raw.select_dtypes("number").columns)
if not num_cols:
    st.error("The dataset needs at least one numeric column (the marks).")
    st.stop()

# ---------- Sidebar settings ----------
st.sidebar.header("Dataset settings")
default_t = num_cols[-1]
for hint in TARGET_HINTS:
    match = [c for c in num_cols if c.lower() == hint]
    if match:
        default_t = match[0]
        break
TARGET = st.sidebar.selectbox("Target column (marks to predict)", num_cols,
                              index=num_cols.index(default_t))

id_like = [c for c in raw.columns
           if raw[c].dtype == object and raw[c].nunique() > 0.9 * len(raw)]
options = [c for c in raw.columns if c != TARGET and c not in id_like]
FEATURES = st.sidebar.multiselect("Input features", options, default=options)
if not FEATURES:
    st.warning("Select at least one input feature in the sidebar.")
    st.stop()

top_default = 100.0 if raw[TARGET].max() > 20 else 20.0
TOP = st.sidebar.number_input("Maximum possible marks", 1.0, 1000.0,
                              top_default)

df = raw[FEATURES + [TARGET]].dropna().reset_index(drop=True)
if len(df) < 30:
    st.error("Not enough rows after removing missing values (need 30+).")
    st.stop()
df["Category"] = df[TARGET].apply(lambda m: get_category(m, TOP))

X = pd.get_dummies(df[FEATURES]).astype(float)
y = df[TARGET]
art = train_models(X, y)

st.caption(f"Dataset: {source} | {len(df):,} rows | "
           f"{len(FEATURES)} input features | target: {TARGET}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard", "🎯 Predict", "🗂️ Dataset",
     "📈 Model Performance", "ℹ️ About"])

# ---------- Dashboard ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(df):,}")
    c2.metric("Average Marks", f"{df[TARGET].mean():.1f}")
    c3.metric("Highest Marks", f"{df[TARGET].max():.1f}")
    c4.metric("Average and above",
              f"{(df[TARGET] / TOP * 100 >= 50).mean() * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Students by Performance Category")
        order = ["Excellent", "Good", "Average", "Needs Improvement"]
        counts = df["Category"].value_counts().reindex(order).fillna(0)
        st.bar_chart(counts)
    with right:
        st.subheader("Distribution of Marks")
        hist, edges = np.histogram(df[TARGET], bins=10)
        labels = [f"{edges[i]:.0f}-{edges[i + 1]:.0f}"
                  for i in range(len(hist))]
        st.bar_chart(pd.Series(hist, index=labels))

    top_num = [c for c in art["importance"].index if c in df.columns]
    if top_num:
        st.subheader(f"{top_num[0]} vs {TARGET}")
        st.scatter_chart(df, x=top_num[0], y=TARGET)

# ---------- Predict ----------
with tab2:
    st.subheader("Enter student details")
    with st.form("predict_form"):
        values = {}
        cols = st.columns(2)
        for i, f in enumerate(FEATURES):
            with cols[i % 2]:
                s = df[f]
                if pd.api.types.is_numeric_dtype(s):
                    lo, hi = s.min(), s.max()
                    if lo == hi:
                        values[f] = lo
                    elif pd.api.types.is_integer_dtype(s):
                        values[f] = st.slider(f, int(lo), int(hi),
                                              int(s.median()))
                    else:
                        values[f] = st.slider(f, float(lo), float(hi),
                                              float(s.median()))
                else:
                    values[f] = st.selectbox(f, sorted(s.astype(str).unique()))
        submitted = st.form_submit_button("Predict Marks", type="primary")

    if submitted:
        row = pd.get_dummies(pd.DataFrame([values])).astype(float)
        row = row.reindex(columns=X.columns, fill_value=0.0)
        pred = float(np.clip(art["best_model"].predict(row)[0], 0, TOP))
        r1, r2 = st.columns(2)
        r1.metric("Predicted Final Marks", f"{pred:.1f} / {TOP:g}")
        r2.metric("Performance Category", get_category(pred, TOP))
        st.progress(min(100, int(round(pred / TOP * 100))))
        st.caption("Prediction by " + art["best_name"] +
                   " trained on the uploaded dataset.")

# ---------- Dataset ----------
with tab3:
    st.subheader("Student Dataset")
    st.caption(f"Source file: {source}")
    st.dataframe(df)
    st.download_button("Download cleaned dataset as CSV",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="student_performance_dataset.csv",
                       mime="text/csv")
    st.subheader("Summary statistics")
    st.dataframe(df.describe().round(2))

# ---------- Model Performance ----------
with tab4:
    st.subheader("Evaluation Metrics (20% test set)")
    st.caption(f"Training rows: {art['n_train']} | "
               f"Test rows: {art['n_test']}")
    st.dataframe(art["results"].round(3))
    st.success("Best model (highest R2): " + art["best_name"])
    st.markdown("**MAE** and **RMSE** are average errors in marks "
                "(lower is better). **R2** shows how much of the variation "
                "the model explains (closer to 1 is better).")

    x1, x2 = st.columns(2)
    with x1:
        st.subheader("Feature Importance (Random Forest)")
        st.bar_chart(art["importance"].head(15))
    with x2:
        st.subheader("Actual vs Predicted Marks")
        st.scatter_chart(art["compare"], x="Actual", y="Predicted")

# ---------- About ----------
with tab5:
    st.subheader("About this project")
    st.write("This app predicts a student's final marks from the input "
             "features in the dataset, then assigns a performance category.")
    st.markdown(
        f"- **Data:** {len(df):,} real records from `{source}`\n"
        "- **Models:** Linear Regression and Random Forest\n"
        "- **Split:** 80% training, 20% testing\n"
        "- **Categories:** Excellent (75%+), Good (60-74%), "
        "Average (50-59%), Needs Improvement (below 50%) of maximum marks\n"
        "- **Limitation:** Results depend on the quality and size of the "
        "dataset, and many factors affecting marks are not recorded."
    )
