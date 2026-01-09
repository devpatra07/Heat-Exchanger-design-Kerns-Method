import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
# from eli5.sklearn import PermutationImportance
import numpy as np


# Streamlit app layout
st.set_page_config(page_title="Dev Patra Toolkit", layout="wide")
st.title("Dev Patra Toolkit")

def load_data():
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file:
        data = pd.read_excel(uploaded_file)
        st.session_state["data"] = data
        st.success("Data Loaded Successfully")

def display_statistics():
    if "data" in st.session_state:
        data = st.session_state["data"]

        # Tab Layout
        tab1, tab2 = st.tabs(["Numerical Statistics", "Categorical Statistics"])

        # Numerical Statistics
        with tab1:
            numeric_stats = data.describe(percentiles=[0.25, 0.5, 0.75]).T
            numeric_stats["Mode"] = data.mode().iloc[0]
            st.dataframe(numeric_stats)

        # Categorical Statistics
        with tab2:
            categorical_data = data.select_dtypes(include=['object', 'category'])
            if not categorical_data.empty:
                st.text("Categorical Data Counts:")
                for col in categorical_data.columns:
                    st.text(f"Column: {col}")
                    st.text(categorical_data[col].value_counts().to_string())
            else:
                st.text("No categorical data found.")
    else:
        st.warning("Please load data first.")

def visualize_data():
    if "data" not in st.session_state:
        st.warning("Please load data first.")
        return

    data = st.session_state["data"]
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Filter numerical data

    if numeric_data.empty:
        st.warning("No numerical columns found in the dataset.")
        return

    st.header("Data Visualization")

    # Single plot selection
    plot_type = st.selectbox(
        "Choose a plot to generate:",
        ["Select", "Pair Plot", "Correlation Heatmap", "Histogram", "Boxplot"]
    )

    # Variable selection for applicable plots
    x_var, y_var = None, None
    selected_columns = None
    if plot_type == "Pair Plot":
        x_var = st.selectbox("Select X-axis Variable:", options=numeric_data.columns.tolist())
        y_var = st.selectbox("Select Y-axis Variable:", options=numeric_data.columns.tolist())
    elif plot_type == "Histogram":
        selected_columns = st.selectbox("Select Variable for Histogram:", numeric_data.columns.tolist())
    elif plot_type == "Boxplot":
        selected_columns = st.selectbox("Select Variable for Boxplot:", numeric_data.columns.tolist())
    else:
        selected_columns = None
        x_var, y_var = None, None

    # Generate button
    if st.button("Generate"):
        st.subheader(f"{plot_type}")
        
        if plot_type == "Pair Plot":
            if x_var and y_var:
                fig = px.scatter(data_frame=data, x=x_var, y=y_var, title=f"Scatter Plot: {x_var} vs {y_var}")
                st.plotly_chart(fig)
            else:
                st.warning("Please select both X and Y variables for the Pair Plot.")
            return

        elif plot_type == "Correlation Heatmap":
            # Drop non-numeric columns for the correlation matrix
            numeric_data = data.select_dtypes(include=['float64', 'int64'])
            
            # Create the correlation matrix
            correlation_matrix = numeric_data.corr()

            # Plot the heatmap using Plotly
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu_r',
                colorbar=dict(title="Correlation"),
                hoverongaps=False,
                showscale=True,
                text=correlation_matrix.round(2).values,  # Display the correlation values
                texttemplate="%{text}",
                textfont=dict(size=5),  # Reduced font size for correlation coefficients
            ))

            # Update layout
            fig.update_layout(
                title="Correlation Heatmap",
                xaxis_title="Variables",
                yaxis_title="Variables",
                xaxis=dict(tickangle=90),  # Rotate x-axis labels for better readability
                yaxis=dict(tickangle=0),  # Rotate y-axis labels for better readability
                template="plotly_white",
                width=1000,  # Increase the width of the heatmap
                height=1000,  # Increase the height of the heatmap
            )

            st.plotly_chart(fig)

        elif plot_type == "Histogram":
            if selected_columns:
                col_data = data[selected_columns].dropna()  # Remove missing values
                mean = col_data.mean()
                std_dev = col_data.std()

                # Plot histogram using Plotly
                fig = px.histogram(col_data, nbins=20, title=f"Histogram of {selected_columns} (Mean: {mean:.2f}, Std Dev: {std_dev:.2f})")
                fig.update_traces(marker=dict(color='skyblue', line=dict(color='black', width=1)))
                fig.add_vline(mean, line=dict(color='red', dash='dash'), annotation_text=f"Mean: {mean:.2f}", annotation_position="top left")
                fig.add_vline(mean + std_dev, line=dict(color='green', dash='dash'), annotation_text=f"Std Dev: {std_dev:.2f}", annotation_position="top left")
                fig.add_vline(mean - std_dev, line=dict(color='green', dash='dash'))

                st.plotly_chart(fig)

            else:
                st.warning("Please select a variable for the histogram.")
        
        elif plot_type == "Boxplot":
            if selected_columns:
                fig = px.box(data, y=selected_columns, title=f"Boxplot of {selected_columns}")
                st.plotly_chart(fig)
            else:
                st.warning("Please select a variable for the Boxplot.")

def scaler_options():
    if "data" not in st.session_state:
        st.warning("Please load data first.")
        return

    data = st.session_state["data"]
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Filter numerical data

    if numeric_data.empty:
        st.warning("No numerical columns found in the dataset.")
        return

    # Select Scaler
    scaler_choice = st.selectbox("Select a Scaler:", ["Select", "Standard Scaler", "Min-Max Scaler"])

    # Apply the selected scaler
    if scaler_choice == "Standard Scaler":
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        scaled_data = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        st.session_state["scaled_data"] = scaled_data
        st.success("Data scaled using Standard Scaler.")

    elif scaler_choice == "Min-Max Scaler":
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        scaled_data = pd.DataFrame(scaled_data, columns=numeric_data.columns)
        st.session_state["scaled_data"] = scaled_data
        st.success("Data scaled using Min-Max Scaler.")
        
    elif scaler_choice == "Select":
        st.warning("Please select a scaler.")

def linear_regression_model():
    if "data" not in st.session_state:
        st.warning("Please load data first.")
        return

    data = st.session_state["data"]
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    if numeric_data.empty:
        st.warning("No numerical columns found in the dataset.")
        return

    # Select features (X) and target variable (y)
    all_features = numeric_data.columns.tolist()
    # selected_features = st.multiselect("Select Features (X):", options=all_features, default=all_features)
    selected_features = st.selectbox(
        "Select Features (X):", options=all_features, key="feature_select"
    )
    target = st.selectbox("Select Target Variable (y):", options=numeric_data.columns.tolist())

    # Ensure that only one target variable is selected
    if len(selected_features) == 0 or target == '':
        st.warning("Please select at least one feature and one target variable.")
        return

    # # Hyperparameter tuning options
    # regularization_choice = st.selectbox(
    #     "Select Regularization (Optional):", 
    #     ["None", "Ridge", "Lasso"]
    # )

    # alpha_value = st.slider("Select Regularization Strength (Alpha):", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    alpha_value = 1.0
    # solver_choice = st.selectbox(
    #     "Select Solver (Optional):", 
    #     ["auto", "svd", "lsqr", "saga"]
    # )

    # Split data into training and testing sets
    test_size = st.slider("Test Data Percentage:", min_value=1, max_value=99, value=20)
    train_size = 100 - test_size

    X = data[selected_features]
    y = data[target]
    # Reshape y to be 1D if necessary
    y = y.squeeze()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)
        # Ensure X_train and X_test are 2D arrays
    if X_train.ndim == 1:
        X_train = X_train.values.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.values.reshape(-1, 1)

    # Initialize Linear Regression Model
    
    # if regularization_choice == "None":
    #     model = LinearRegression()  # No 'solver' parameter here
    # elif regularization_choice == "Ridge":
    #     model = Ridge(alpha=alpha_value, solver=solver_choice)
    # elif regularization_choice == "Lasso":
    #     model = Lasso(alpha=alpha_value, solver=solver_choice)
    model = LinearRegression()  # No 'solver' parameter here
    # Run Model and Train
    if st.button("Run Model"):
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate performance metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        # Visualization of metrics
        st.subheader("Model Performance")
        
        # R² and MSE Visualization
        fig = go.Figure()

        # R²
        fig.add_trace(go.Bar(
            x=["Train", "Test"],
            y=[r2_train, r2_test],
            name="R²",
            marker=dict(color='skyblue'),
            text=["{:.2f}".format(r2_train), "{:.2f}".format(r2_test)],
            textposition="auto"
        ))

        # MSE
        fig.add_trace(go.Bar(
            x=["Train", "Test"],
            y=[mse_train, mse_test],
            name="MSE",
            marker=dict(color='orange'),
            text=["{:.2f}".format(mse_train), "{:.2f}".format(mse_test)],
            textposition="auto"
        ))

        fig.update_layout(
            title="Linear Regression Model Performance",
            barmode="group",
            xaxis_title="Dataset",
            yaxis_title="Score",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig)

        # Show training and testing predictions
        st.subheader("Predictions vs Actual Values")

        # Train and test predictions graph
        fig_train_test = go.Figure()

        # Train predictions
        fig_train_test.add_trace(go.Scatter(
            x=y_train.index, y=y_pred_train, mode="markers", name="Train Predictions", marker=dict(color='green')
        ))

        # Actual values for train
        fig_train_test.add_trace(go.Scatter(
            x=y_train.index, y=y_train, mode="markers", name="Actual Train Values", marker=dict(color='red')
        ))

        # Test predictions
        fig_train_test.add_trace(go.Scatter(
            x=y_test.index, y=y_pred_test, mode="markers", name="Test Predictions", marker=dict(color='blue')
        ))

        # Actual values for test
        fig_train_test.add_trace(go.Scatter(
            x=y_test.index, y=y_test, mode="markers", name="Actual Test Values", marker=dict(color='purple')
        ))

        # Combine R² values into one annotation
        r2_values_text = f"R² (Train): {r2_train:.2f} | R² (Test): {r2_test:.2f}"
        fig_train_test.add_annotation(
            x=0.5, y=0.95, xref="paper", yref="paper",
            text=r2_values_text, showarrow=False, font=dict(size=14, color="green"),
            align="center"
        )
        

        fig_train_test.update_layout(
            title="Training and Testing Predictions vs Actual",
            xaxis_title="Index",
            yaxis_title="Target Variable",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig_train_test)

def ai_model():
    if "data" not in st.session_state:
        st.warning("Please load data first.")
        return

    data = st.session_state["data"]
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Filter numerical data

    if numeric_data.empty:
        st.warning("No numerical columns found in the dataset.")
        return

    # Select features (X) and target variable (y)
    all_features = numeric_data.columns.tolist()
    selected_features = st.multiselect("Select Features (X):", options=all_features, default=all_features)
    # selected_features = st.selectbox("Select Target Variable (X):", options=numeric_data.columns.tolist())

    target = st.selectbox("Select Target Variable (y):", options=numeric_data.columns.tolist())

    if len(selected_features) == 0 or not target:
        st.warning("Please select at least one feature and one target variable.")
        return

    # Split data into training and testing sets
    test_size = st.slider("Test Data Percentage:", min_value=1, max_value=99, value=20)
    train_size = 100 - test_size

    X = data[selected_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)
    if X_train.ndim == 1:
        X_train = X_train.values.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.values.reshape(-1, 1)
    # Model selection
    model_choice = st.selectbox("Choose an AI Model:", ["Select", "LightGBM", "XGBoost", "Extra Trees"])

    if model_choice == "LightGBM":
        # LightGBM Parameters
        # learning_rate = st.number_input("Learning Rate:", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        # n_estimators = st.number_input("Number of Estimators:", min_value=1, max_value=500, value=100, step=1)
        # max_depth = st.number_input("Max Depth:", min_value=-1, max_value=50, value=-1, step=1)
        learning_rate = 0.01
        n_estimators = 50
        max_depth = 3

        model = LGBMRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

    elif model_choice == "XGBoost":
        # XGBoost Parameters
        # learning_rate = st.number_input("Learning Rate:", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        # n_estimators = st.number_input("Number of Estimators:", min_value=1, max_value=500, value=100, step=1)
        # max_depth = st.number_input("Max Depth:", min_value=1, max_value=50, value=6, step=1)

        model = XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        learning_rate = 0.01
        n_estimators = 50
        max_depth = 3

    elif model_choice == "Extra Trees":
        # Extra Trees Parameters
        # n_estimators = st.number_input("Number of Estimators:", min_value=1, max_value=500, value=100, step=1)
        # max_depth = st.number_input("Max Depth:", min_value=1, max_value=50, value=None, step=1)
        n_estimators = 50
        max_depth = 3

        model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)


    else:
        st.warning("Please select a valid model.")
        return

    # Run the model
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate performance metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        # Visualization of metrics
        st.subheader("Model Performance")

        # R² and MSE Visualization
        fig = go.Figure()

        # R²
        fig.add_trace(go.Bar(
            x=["Train", "Test"],
            y=[r2_train, r2_test],
            name="R²",
            marker=dict(color='skyblue'),
            text=["{:.2f}".format(r2_train), "{:.2f}".format(r2_test)],
            textposition="auto"
        ))

        # MSE
        fig.add_trace(go.Bar(
            x=["Train", "Test"],
            y=[mse_train, mse_test],
            name="MSE",
            marker=dict(color='orange'),
            text=["{:.2f}".format(mse_train), "{:.2f}".format(mse_test)],
            textposition="auto"
        ))

        fig.update_layout(
            title="AI Model Performance",
            barmode="group",
            xaxis_title="Dataset",
            yaxis_title="Score",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig)
        # Show training and testing predictions
        st.subheader("Predictions vs Actual Values")

        # Train and test predictions graph
        fig_train_test = go.Figure()

        # Train predictions
        fig_train_test.add_trace(go.Scatter(
            x=y_train.index, y=y_pred_train, mode="markers", name="Train Predictions", marker=dict(color='green')
        ))

        # Actual values for train
        fig_train_test.add_trace(go.Scatter(
            x=y_train.index, y=y_train, mode="markers", name="Actual Train Values", marker=dict(color='red')
        ))

        # Test predictions
        fig_train_test.add_trace(go.Scatter(
            x=y_test.index, y=y_pred_test, mode="markers", name="Test Predictions", marker=dict(color='blue')
        ))

        # Actual values for test
        fig_train_test.add_trace(go.Scatter(
            x=y_test.index, y=y_test, mode="markers", name="Actual Test Values", marker=dict(color='purple')
        ))

        # Combine R² values into one annotation
        r2_values_text = f"R² (Train): {r2_train:.2f} | R² (Test): {r2_test:.2f}"
        fig_train_test.add_annotation(
            x=0.5, y=0.95, xref="paper", yref="paper",
            text=r2_values_text, showarrow=False, font=dict(size=14, color="green"),
            align="center"
        )

        # Display the plot
        st.plotly_chart(fig_train_test)

   

def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    options = ["Home", "Show Basic Statistics","Data Visualization","Scaler", "Linear Regression","AI Model"]
    choice = st.sidebar.radio("Go to", options)

    if choice == "Home":
        st.header("Home")
        load_data()

    elif choice == "Show Basic Statistics":
        st.header("Basic Statistics")
        display_statistics()

    elif choice == "Data Visualization":
        visualize_data()

    elif choice == "Scaler":
        st.header("Data Scaling")
        scaler_options()

    elif choice == "Linear Regression":
        st.header("Linear Regression Model")
        linear_regression_model()

    elif choice == "AI Model":
        st.header("AI Models")
        ai_model()

if __name__ == "__main__":
    main()
