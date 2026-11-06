# student-performance-dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------
st.title("🎓 Student Academic Performance Prediction")
st.markdown(
    "### Machine Learning Based Student Analysis Dashboard"
)
st.caption("Data Science Internship Project")

# -----------------------------
# GENERATE DATASET
# -----------------------------
@st.cache_data
def generate_data():

    np.random.seed(42)
    n = 300

    df = pd.DataFrame({
        "Study_Hours": np.random.uniform(1, 8, n),
        "Attendance": np.random.uniform(50, 100, n),
        "Previous_Marks": np.random.uniform(35, 95, n),
        "Assignment_Score": np.random.uniform(30, 100, n)
    })

    df["Final_Marks"] = (
        0.25 * df["Previous_Marks"]
        + 3.2 * df["Study_Hours"]
        + 0.22 * df["Attendance"]
        + 0.18 * df["Assignment_Score"]
        + np.random.normal(0, 6, n)
    ).clip(0, 100)

    df["Performance"] = pd.cut(
        df["Final_Marks"],
        bins=[-1, 49, 69, 100],
        labels=["Needs Support", "Developing", "Strong"]
    )

    return df

df = generate_data()

# -----------------------------
# TRAIN MODEL
# -----------------------------
features = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignment_Score"
]

X = df[features]
y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

test_predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("📝 Student Details")

student_name = st.sidebar.text_input(
    "Student Name", "Student 1"
)

study_hours = st.sidebar.slider(
    "Study Hours per Day", 0.0, 12.0, 6.0
)

attendance = st.sidebar.slider(
    "Attendance (%)", 0, 100, 85
)

previous_marks = st.sidebar.slider(
    "Previous Marks", 0, 100, 70
)

assignment_score = st.sidebar.slider(
    "Assignment Score", 0, 100, 80
)

predict_button = st.sidebar.button(
    "🔮 Predict Performance",
    use_container_width=True
)

# -----------------------------
# DASHBOARD METRICS
# -----------------------------
st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Students", len(df))
c2.metric("Average Marks", round(df["Final_Marks"].mean(), 2))
c3.metric("Model MAE", round(mae, 2))
c4.metric("Model R²", round(r2, 3))

st.divider()

# -----------------------------
# PREDICTION
# -----------------------------
if predict_button:

    student = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Marks": previous_marks,
        "Assignment_Score": assignment_score
    }])

    predicted_marks = float(model.predict(student)[0])

    if predicted_marks < 50:
        performance = "Needs Support"
        priority = "High Support Priority"

    elif predicted_marks < 70:
        performance = "Developing"
        priority = "Moderate Support Priority"

    else:
        performance = "Strong"
        priority = "Low Support Priority"

    st.subheader("🎯 Prediction Results")

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Student",
        student_name
    )

    p2.metric(
        "Predicted Marks",
        f"{predicted_marks:.2f}/100"
    )

    p3.metric(
        "Performance",
        performance
    )

    st.info(f"Support Priority: {priority}")

    if performance == "Needs Support":
        st.warning(
            "Recommendation: Provide additional academic support."
        )
    elif performance == "Developing":
        st.info(
            "Recommendation: Improve study habits and practice."
        )
    else:
        st.success(
            "Recommendation: Maintain consistent performance."
        )

# -----------------------------
# CHARTS
# -----------------------------
st.subheader("📈 Data Visualization")

tab1, tab2, tab3 = st.tabs([
    "Marks Distribution",
    "Study Hours Analysis",
    "Correlation"
])

with tab1:

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        df["Final_Marks"],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Final Marks")
    ax.set_ylabel("Number of Students")

    st.pyplot(fig)
    plt.close(fig)

with tab2:

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.scatterplot(
        data=df,
        x="Study_Hours",
        y="Final_Marks",
        ax=ax
    )

    ax.set_title("Study Hours vs Final Marks")

    st.pyplot(fig)
    plt.close(fig)

with tab3:

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        df.select_dtypes("number").corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("🧠 Important Prediction Factors")

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))

importance.plot(
    kind="bar",
    ax=ax
)

ax.set_ylabel("Importance")

st.pyplot(fig)
plt.close(fig)

# -----------------------------
# DATASET
# -----------------------------
st.subheader("📚 Student Dataset")

st.dataframe(
    df.head(30),
    use_container_width=True
)

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Student Dataset",
    data=csv,
    file_name="student_academic_dataset.csv",
    mime="text/csv"
)

# -----------------------------
# MODEL INFORMATION
# -----------------------------
with st.expander("About the Machine Learning Model"):

    st.write("""
    Algorithm: Random Forest Regression

    Dataset: 300 synthetic student records

    Features:
    - Study Hours
    - Attendance
    - Previous Marks
    - Assignment Score

    Target:
    - Final Marks

    Evaluation:
    - Mean Absolute Error (MAE)
    - R² Score
    """)

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.caption(
    "Student Academic Performance Prediction | "
    "Data Science Internship Project"
)
