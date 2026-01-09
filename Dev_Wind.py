import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Dev Patra ML Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<h1 style='text-align:center;'>📊 Dev Patra – ML & Data Analytics Toolkit</h1>
<p style='text-align:center; font-size:18px;'>
EDA • Visualization • Scaling • Regression • AI Models
</p>
<hr>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown("## 🔍 Navigation")
    page = st.radio(
        "",
        ["🏠 Home", "📈 Statistics", "📊 Visualization", "⚖️ Scaling",
         "📉 Linear Regression", "🤖 AI Models"]
    )
    st.markdown("---")
    st.markdown("**Developer:** Dev Patra")
    st.markdown("🎓 IMTech Chemical Engineering")
    st.markdown("🔬 ML • Modeling • Optimization")

# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------
def load_data():
    st.subheader("📂 Upload Dataset")

    file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

    if file:
        df = pd.read_excel(file)
        st.session_state["data"] = df

        st.success("✅ Dataset loaded successfully")

        with st.expander("🔎 Dataset Preview"):
            st.dataframe(df.head(), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", df.isnull().sum().sum())

# ---------------------------------------------------
# STATISTICS
# ---------------------------------------------------
def statistics():
    if "data" not in st.session_state:
        st.warning("Please upload data first.")
        return

    df = st.session_state["data"]

    tab1, tab2 = st.tabs(["📈 Numerical", "🔤 Categorical"])

    with tab1:
        num = df.select_dtypes(include=np.number)
        stats = num.describe().T
        stats["Mode"] = num.mode().iloc[0]
        st.dataframe(stats, use_container_width=True)

        st.download_button(
            "⬇️ Download Statistics",
            stats.to_csv(),
            "numerical_statistics.csv"
        )

    with tab2:
        cat = df.select_dtypes(include=["object", "category"])
        if cat.empty:
            st.info("No categorical columns found.")
        else:
            for col in cat.columns:
                st.write(f"**{col}**")
                st.write(cat[col].value_counts())

# ---------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------
def visualization():
    if "data" not in st.session_state:
        st.warning("Please upload data first.")
        return

    df = st.session_state["data"]
    num = df.select_dtypes(include=np.number)

    st.subheader("📊 Data Visualization")

    plot = st.selectbox(
        "Select Plot Type",
        ["Correlation Heatmap", "Scatter Plot", "Histogram", "Boxplot"]
    )

    if plot == "Correlation Heatmap":
        corr = num.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    else:
        col = st.selectbox("Select Variable", num.columns)

        if plot == "Scatter Plot":
            y = st.selectbox("Select Y variable", num.columns)
            fig = px.scatter(df, x=col, y=y, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)

        elif plot == "Histogram":
            fig = px.histogram(df, x=col, nbins=30)
            st.plotly_chart(fig, use_container_width=True)

        elif plot == "Boxplot":
            fig = px.box(df, y=col)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------
def scaling():
    if "data" not in st.session_state:
        st.warning("Please upload data first.")
        return

    df = st.session_state["data"]
    num = df.select_dtypes(include=np.number)

    scaler_type = st.selectbox("Select Scaler", ["Standard Scaler", "Min-Max Scaler"])

    if st.button("Apply Scaling"):
        scaler = StandardScaler() if scaler_type == "Standard Scaler" else MinMaxScaler()
        scaled = pd.DataFrame(scaler.fit_transform(num), columns=num.columns)

        st.session_state["scaled"] = scaled

        c1, c2 = st.columns(2)
        c1.subheader("Original Data")
        c1.dataframe(num.head())

        c2.subheader("Scaled Data")
        c2.dataframe(scaled.head())

# ---------------------------------------------------
# LINEAR REGRESSION
# ---------------------------------------------------
def linear_regression():
    if "data" not in st.session_state:
        st.warning("Please upload data first.")
        return

    df = st.session_state["data"]
    num = df.select_dtypes(include=np.number)

    X_col = st.multiselect("Select Features (X)", num.columns)
    y_col = st.selectbox("Select Target (y)", num.columns)

    test_size = st.slider("Test Size (%)", 10, 50, 20)

    if st.button("Run Linear Regression"):
        X = df[X_col]
        y = df[y_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )

        model = LinearRegression()

        with st.spinner("Training model..."):
            model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        r2_tr = r2_score(y_train, y_pred_train)
        r2_te = r2_score(y_test, y_pred_test)

        st.metric("Train R²", f"{r2_tr:.3f}")
        st.metric("Test R²", f"{r2_te:.3f}")

        coef = pd.DataFrame({
            "Feature": X_col,
            "Coefficient": model.coef_
        }).sort_values(by="Coefficient", key=abs, ascending=False)

        st.subheader("📌 Feature Importance")
        st.dataframe(coef)

# ---------------------------------------------------
# AI MODELS
# ---------------------------------------------------
def ai_models():
    if "data" not in st.session_state:
        st.warning("Please upload data first.")
        return

    df = st.session_state["data"]
    num = df.select_dtypes(include=np.number)

    X_cols = st.multiselect("Select Features (X)", num.columns)
    y_col = st.selectbox("Select Target (y)", num.columns)

    model_name = st.selectbox(
        "Select Model",
        ["LightGBM", "XGBoost", "Extra Trees"]
    )

    test_size = st.slider("Test Size (%)", 10, 50, 20)

    if st.button("Train Model"):
        X = df[X_cols]
        y = df[y_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )

        if model_name == "LightGBM":
            model = LGBMRegressor(n_estimators=50, learning_rate=0.05)
        elif model_name == "XGBoost":
            model = XGBRegressor(n_estimators=50, learning_rate=0.05)
        else:
            model = ExtraTreesRegressor(n_estimators=100)

        with st.spinner("Training model..."):
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.metric("Test R²", f"{r2_score(y_test, y_pred):.3f}")
        st.metric("Test MSE", f"{mean_squared_error(y_test, y_pred):.3f}")

        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({
                "Feature": X_cols,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig = px.bar(imp, x="Importance", y="Feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# ROUTING
# ---------------------------------------------------
if page == "🏠 Home":
    load_data()
elif page == "📈 Statistics":
    statistics()
elif page == "📊 Visualization":
    visualization()
elif page == "⚖️ Scaling":
    scaling()
elif page == "📉 Linear Regression":
    linear_regression()
elif page == "🤖 AI Models":
    ai_models()
