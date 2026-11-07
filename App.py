import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Academic Performance Prediction", page_icon="🎓", layout="wide")

# --------- CSS ---------
st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem}
.stApp{background:linear-gradient(180deg,#ffffff 0%,#f3f7ff 55%,#e4f5ff 100%)}
.hero{background:linear-gradient(110deg,#254be8,#8737ee);color:white;text-align:center;padding:28px 18px;border-radius:0 0 28px 28px;margin-bottom:20px}
.hero h1{color:white!important;font-size:clamp(26px,5vw,48px);margin:0 0 12px}
.hero p{color:#f0edff;font-size:clamp(14px,2vw,19px);margin:7px 0}
.card{background:rgba(255,255,255,.88);border:1px solid #e0e5ef;border-radius:18px;padding:20px;margin:12px 0;box-shadow:0 5px 18px rgba(30,64,175,.06)}
div.stButton>button{border-radius:10px;min-height:42px;font-weight:600}
[data-testid="stSidebar"]{background:#edf2ff}
</style>
""", unsafe_allow_html=True)

# --------- LOGIN ---------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    st.markdown("""<div class="hero"><h1>🎓 Student Academic Performance Prediction</h1>
    <p>Machine Learning project | Scikit-learn + Streamlit</p></div>""", unsafe_allow_html=True)
    _, mid, _ = st.columns([1,1.2,1])
    with mid:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## 🔐 Welcome back")
        st.write("Sign in to open your prediction dashboard.")
        with st.form("login"):
            username=st.text_input("Username",placeholder="admin")
            password=st.text_input("Password",type="password")
            login=st.form_submit_button("Login",use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
        if login:
            if username.strip()=="admin" and password=="1234":
                st.session_state.logged_in=True
                st.rerun()
            else:
                st.error("Incorrect username or password")
    st.info("Demo login: username **admin** · password **1234**")
    st.stop()

# --------- SIDEBAR / THREE DOT MENU ---------
with st.sidebar:
    st.markdown("## ⋮ Menu")
    tab=st.radio("Pages",["Dashboard","Predict","Dataset","Model Performance"],label_visibility="collapsed")
    st.divider()
    if st.button("Logout",use_container_width=True):
        st.session_state.logged_in=False
        st.rerun()
    st.markdown("### 📂 Dataset")
    upload=st.file_uploader("Upload CSV",type=["csv"])

# --------- DATA ---------
@st.cache_data
def synthetic_data():
    rng=np.random.default_rng(42)
    n=500
    study=rng.uniform(0,10,n)
    attend=rng.uniform(50,100,n)
    previous=rng.uniform(30,100,n)
    sleep=rng.uniform(4,9,n)
    final=np.clip(5+study*4.2+attend*.25+previous*.35+sleep*1.2+rng.normal(0,7,n),0,100)
    return pd.DataFrame({"Study_Hours":study.round(1),"Attendance":attend.round(1),
                         "Previous_Marks":previous.round(1),"Sleep_Hours":sleep.round(1),
                         "Final_Marks":final.round(1)})
if upload is not None:
    try: df=pd.read_csv(upload)
    except Exception as e:
        st.error(f"CSV could not be loaded: {e}")
        df=synthetic_data()
else: df=synthetic_data()

# target/model
num=df.select_dtypes(include=np.number).copy()
candidates=["Final_Marks","final_marks","G3","Marks","Score","score"]
target=next((x for x in candidates if x in num.columns),num.columns[-1] if num.shape[1]>=2 else None)
model=None; features=[]; mae=np.nan; r2=np.nan; X=pd.DataFrame()
if target and num.shape[1]>=2:
    clean=num.dropna(subset=[target])
    features=[x for x in clean.columns if x!=target]
    X=clean[features].fillna(clean[features].median()); y=clean[target]
    if len(clean)>=10 and features:
        xtr,xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
        model=RandomForestRegressor(n_estimators=150,random_state=42).fit(xtr,ytr)
        pred=model.predict(xte); mae=mean_absolute_error(yte,pred); r2=r2_score(yte,pred)

# --------- HEADER ---------
st.markdown("""<div class="hero">
<h1>🎓 Student Academic Performance Prediction</h1>
<p>Machine Learning project | Scikit-learn + Streamlit</p>
<p>Dataset: student performance records for academic analysis.</p>
</div>""",unsafe_allow_html=True)

# horizontal navigation-like tabs
t1,t2,t3,t4=st.tabs(["📊 Dashboard","🎯 Predict","📁 Dataset","📈 Model Performance"])

# --------- DASHBOARD ---------
with t1:
    st.markdown("## 📊 Dataset overview")
    a,b,c,d=st.columns(4)
    a.metric("Student records",len(df))
    b.metric("Numeric columns",num.shape[1])
    c.metric("Prediction target",target or "Not found")
    d.metric("Model", "Random Forest" if model else "Not trained")
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("Dataset preview")
    st.dataframe(df.head(15),use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)
    if target and target in df:
        st.subheader("Marks distribution")
        st.bar_chart(df[target].dropna())

# --------- PREDICT ---------
with t2:
    st.markdown("## Enter student details")
    if model is None:
        st.warning("Upload a CSV with at least 10 rows and 2 numeric columns to train the model.")
    else:
        vals={}
        # use screenshot's familiar fields where present; any other feature gets a numeric input
        st.markdown('<div class="card">',unsafe_allow_html=True)
        for feature in features:
            s=X[feature]; lo=float(s.min()); hi=float(s.max()); default=float(np.clip(s.median(),lo,hi))
            key=feature.lower()
            if "study" in key and ("hour" in key or "time" in key):
                vals[feature]=st.slider("Study hours per day",0.0,12.0,float(np.clip(default,0,12)),step=.5)
            elif "attendance" in key:
                vals[feature]=st.slider("Attendance (%)",0.0,100.0,float(np.clip(default,0,100)),step=1.0)
            elif "previous" in key and ("mark" in key or "score" in key or "grade" in key):
                vals[feature]=st.slider("Previous exam marks",0.0,100.0,float(np.clip(default,0,100)),step=1.0)
            elif "sleep" in key:
                vals[feature]=st.slider("Sleep hours",0.0,12.0,float(np.clip(default,0,12)),step=.5)
            else:
                if lo==hi: hi=lo+1
                vals[feature]=st.number_input(feature,min_value=lo,max_value=hi,value=float(np.clip(default,lo,hi)))
        st.markdown("</div>",unsafe_allow_html=True)
        if st.button("🔮 Predict Marks",use_container_width=True):
            result=float(model.predict(pd.DataFrame([vals],columns=features))[0])
            st.success(f"Predicted {target}: {result:.2f}")
            st.metric("Predicted marks",f"{result:.2f}/100")

# --------- DATASET ---------
with t3:
    st.markdown("## 📁 Dataset")
    st.dataframe(df,use_container_width=True)
    st.download_button("Download CSV",df.to_csv(index=False),"student_dataset.csv","text/csv")

# --------- MODEL PERFORMANCE ---------
with t4:
    st.markdown("## 📈 Model Performance")
    if model is None: st.warning("Model is not trained. Upload a valid CSV.")
    else:
        a,b=st.columns(2); a.metric("Mean Absolute Error (MAE)",f"{mae:.2f}"); b.metric("R² Score",f"{r2:.3f}")
        st.subheader("Feature importance")
        imp=pd.DataFrame({"Feature":features,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False)
        st.bar_chart(imp.set_index("Feature"))
