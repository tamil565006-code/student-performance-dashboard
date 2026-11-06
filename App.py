 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction")
st.write("Machine Learning Internship Project")

# Generate synthetic dataset
@st.cache_data
def create_data():
    np.random.seed(42)
    n = 300

    data = pd.DataFrame({
        "Study_Hours": np.random.uniform(1, 8, n),
        "Attendance": np.random.uniform(50, 100, n),
        "Previous_Marks": np.random.uniform(35, 95, n),
        "Assignment_Score": np.random.uniform(30, 100, n)
    })

    data["Final_Marks"] = (
        0.25 * data["Previous_Marks"]
        + 3.2 * data["Study_Hours"]
        + 0.22 * data["Attendance"]
        + 0.18 * data["Assignment_Score"]
        + np.random.normal(0, 6, n)
    ).clip(0, 100)

    return data

df = create_data()

# Train machine learning model
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
        n_estimators=50,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

# Sidebar inputs
st.sidebar.header("Student Details")

name = st.sidebar.text_input("Student Name", "Student")

hours = st.sidebar.slider("Study Hours", 0.0, 12.0, 6.0)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 85)
previous = st.sidebar.slider("Previous Marks", 0, 100, 70)
assignment = st.sidebar.slider("Assignment Score", 0, 100, 80)

if st.sidebar.button("Predict Marks"):

    student = pd.DataFrame([{
        "Study_Hours": hours,
        "Attendance": attendance,
        "Previous_Marks": previous,
        "Assignment_Score": assignment
    }])

    result = float(model.predict(student)[0])
    result = max(0, min(100, result))

    st.subheader("Prediction Result")

    st.metric("Student", name)
    st.metric("Predicted Marks", f"{result:.2f}/100")

    if result < 50:
        st.warning("Performance: Needs Support")
    elif result < 70:
        st.info("Performance: Developing")
    else:
        st.success("Performance: Strong")

    st.write("Support Priority:")
    if result < 50:
        st.write("High Support Priority")
    elif result < 70:
        st.write("Moderate Support Priority")
    else:
        st.write("Low Support Priority")

# Dataset information
st.divider()
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Students", len(df))
col2.metric("Average Marks", round(df["Final_Marks"].mean(), 2))
col3.metric("Model MAE", round(mae, 2))

st.subheader("Student Dataset")
st.dataframe(df.head(20), use_container_width=True)

# Simple chart without matplotlib
st.subheader("Marks Distribution")
st.bar_chart(df["Final_Marks"])

# Download dataset
csv = df.to_csv(index=False)

st.download_button(
    "Download Student Dataset",
    data=csv,
    file_name="student_dataset.csv",
    mime="text/csv"
)

st.caption("Demo project using synthetic student data.")
