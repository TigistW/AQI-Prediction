import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

from utils import load_dataset, null_values, drop_null_values, get_aqi_category
import warnings
warnings.filterwarnings("ignore")
os.makedirs("../model", exist_ok=True)

st.sidebar.header("Navigation")
menu = ["Load Dataset", "Clean Dataset", "Visualization","Column Dropping","Data Splitting", "Model Testing", "Feature Importance", "Lets Predict!"]
choice = st.sidebar.radio("Choose an action", menu)

if choice == "Load Dataset":
    st.header("Dataset loaded successfully!")
    df = load_dataset()
    num_rows = st.number_input("Enter the number of rows to display:", min_value=1, max_value=len(df), value=5, step=1)
    st.dataframe(df.head(num_rows))
    st.session_state['df'] = df

elif choice == "Clean Dataset":
    
    st.header("Clean Dataset")
    if 'df' not in st.session_state:
        st.warning("Please load a dataset first!")
    else:
        df = st.session_state['df']
        st.write("Current Null Values:")
        st.write(null_values(df)) 
        st.write("Dataset Before Dropping Null Values:")
        st.dataframe(df)
        df_cleaned = df.copy() 
        drop_null_values(df_cleaned)
        st.write("Dataset After Dropping Null Values:")
        st.dataframe(df_cleaned)
        st.session_state['df'] = df_cleaned


elif choice == "Visualization":
    st.header("Dataset Visualization")
    if 'df' not in st.session_state:
        st.warning("Please load a dataset first!")
    else:
        df = st.session_state['df']
        selected_column = st.selectbox("Select a column for univariate analysis", df.columns)
        if selected_column:
            st.subheader(f"Univariate Analysis of {selected_column}")
            fig = px.histogram(df, x=selected_column, nbins=30, title=f"Univariate Analysis of {selected_column}")
            fig.update_layout(xaxis_title=selected_column, yaxis_title="Frequency")
            st.plotly_chart(fig)
            
    # Bivariate Analysis - Correlation Heatmap
    st.subheader("Bivariate Analysis - Correlation Heatmap")
    # if st.checkbox("Show Correlation Heatmap"):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")
    
    # Time Series Analysis
    st.title("Time Series Analysis")
    
    df['Date'] = pd.to_datetime(df['Date'])

    # Plot 1: Trend of AQI Over Time
    st.header("Trend of AQI Over Time")
    fig, ax = plt.subplots(figsize=(18, 6))
    df.groupby('Date')['AQI'].mean().plot(ax=ax)
    ax.set_title("Trend of AQI Over Time")
    ax.set_ylabel("Average AQI")
    ax.set_xlabel("Date")
    st.pyplot(fig)

    # Plot 2: Monthly Trends of PM2.5, PM10, and AQI
    st.header("Monthly Trends of PM2.5, PM10, and AQI")
    fig, ax = plt.subplots(figsize=(12, 6))
    df.set_index('Date')[['PM2.5', 'PM10', 'AQI']].resample('M').mean().plot(ax=ax)
    ax.set_title("Monthly Trends of PM2.5, PM10, and AQI")
    ax.set_ylabel("Concentration (µg/m³)")
    ax.set_xlabel("Date")
    st.pyplot(fig)

    # Plot 3: Monthly Trends of Xylene and AQI
    st.header("Monthly Trends of Xylene and AQI")
    fig, ax = plt.subplots(figsize=(12, 6))
    df.set_index('Date')[['Xylene', 'AQI']].resample('M').mean().plot(ax=ax)
    ax.set_title("Monthly Trends of Xylene and AQI")
    ax.set_ylabel("Concentration (µg/m³)")
    ax.set_xlabel("Date")
    st.pyplot(fig)
    
elif choice == "Column Dropping":
    st.header("Drop Columns in Dataset")
    if 'df' not in st.session_state:
        st.warning("Please load a dataset first!")
    else:
        df = st.session_state['df']
        
        st.write("Current Dataset:")
        st.dataframe(df)

        columns_to_drop = st.multiselect("Select columns to drop:", options=df.columns)
        if st.button("Drop Columns"):
            if columns_to_drop:
                df_dropped = df.drop(columns=columns_to_drop)
                st.write("Updated Dataset (after dropping selected columns):")
                st.dataframe(df_dropped)
                st.session_state['df'] = df_dropped
            else:
                st.write("No columns dropped yet. Please select columns above.")
                
elif choice == "Data Splitting":
    st.title("Train-Test Split and Data Scaling")
    if 'df' not in st.session_state:
        st.warning("Please load a dataset first!")
    else:
        df = st.session_state['df']
        st.header("Current Dataset")
        st.dataframe(df)
        st.header("Data Splitting and Scaling")
        target_column = st.selectbox("Select the target column (y):", df.columns, index=df.columns.get_loc("AQI"))

        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            scalar_path = f"../model/standard_scalar.joblib"
            joblib.dump(scaler, scalar_path)

            X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
           
            st.session_state['y_val'] = y_val
            st.session_state['y_test'] = y_test
            st.session_state['y_train'] = y_train
            st.session_state['X_train'] = X_train
            st.session_state['X_val'] = X_val
            st.session_state['X_test'] = X_test
            

            st.subheader("Data Shapes After Splitting")
            st.write(f"Training data shape: {X_train.shape}")
            st.write(f"Validation data shape: {X_val.shape}")
            st.write(f"Testing data shape: {X_test.shape}")
            
            st.subheader("Visualizing Data Splits as Histograms")

            def plot_histograms(data, title):
                fig, ax = plt.subplots(figsize=(10, 6))
                for i in range(data.shape[1]):
                    ax.hist(data[:, i], bins=15, alpha=0.5, label=f"Feature {i+1}")
                ax.set_title(title)
                ax.set_xlabel("Scaled Feature Values")
                ax.set_ylabel("Frequency")
                ax.legend()
                st.pyplot(fig)

            plot_histograms(X_train, "Training Data Histogram")
            plot_histograms(X_val, "Validation Data Histogram")
            plot_histograms(X_test, "Testing Data Histogram")
            
            
elif choice == "Model Testing":
    st.header("Train and Evaluate Models")
    if 'df' not in st.session_state:
        st.warning("Please load a dataset first!")
    else:
        df = st.session_state['df']
        y_val = st.session_state['y_val']
        y_test = st.session_state['y_test']
        y_train = st.session_state['y_train']
        X_train = st.session_state['X_train']
        X_val = st.session_state['X_val']
        X_test = st.session_state['X_test']

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Support Vector Regressor (SVR)": SVR(kernel='rbf'),
        }
        
        st.title("Model Visualizer")

        st.header("Available Models")
        model_names = list(models.keys())
        selected_model = st.selectbox("Select a model to explore:", model_names)

        if selected_model:
            st.subheader(f"Model: {selected_model}")
            model_instance = models[selected_model]
            
            st.write("Model Parameters:")
            st.write(model_instance.get_params())

            descriptions = {
                "Linear Regression": "Linear Regression is a basic regression algorithm that fits a linear relationship between input features and the target.",
                "Ridge Regression": "Ridge Regression adds L2 regularization to linear regression, which helps reduce overfitting by penalizing large coefficients.",
                "Lasso Regression": "Lasso Regression adds L1 regularization to linear regression, which performs feature selection by driving some coefficients to zero.",
                "Random Forest": "Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.",
                "Gradient Boosting": "Gradient Boosting is an ensemble technique that builds trees sequentially, minimizing errors from previous models.",
                "Decision Tree": "Decision Tree splits data into branches based on feature thresholds, forming a tree structure for predictions.",
                "Support Vector Regressor (SVR)": "SVR is a kernel-based regression technique that fits a hyperplane or decision boundary to minimize prediction errors.",
            }
            st.write("Description:")
            st.write(descriptions.get(selected_model, "No description available."))

        results = []
        
        if st.button("Train and Evaluate Models"):
            for name, model in models.items():
                model.fit(X_train, y_train)

                y_val_pred = model.predict(X_val)
                val_mse = mean_squared_error(y_val, y_val_pred)
                val_r2 = r2_score(y_val, y_val_pred)
                val_rmse = np.sqrt(val_mse)

                y_test_pred = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                test_rmse = np.sqrt(test_mse)

                model_path = f"../model/{name.replace(' ', '_')}.joblib"
                joblib.dump(model, model_path)

                results.append({
                    "Model": name,
                    "Validation MSE": val_mse,
                    "Validation R²": val_r2,
                    "Validation RMSE": val_rmse,
                    "Test MSE": test_mse,
                    "Test R²": test_r2,
                    "Test RMSE": test_rmse,
                    "Model Path": model_path,
                })

            results_df = pd.DataFrame(results)
            st.write("Model Evaluation Results:")
            st.dataframe(results_df)
            
elif choice == "Feature Importance":
    def plot_feature_importance(model, model_name, feature_names):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(feature_names)), importance[indices], align="center")
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
        ax.set_title(f"Feature Importance - {model_name}")
        ax.set_ylabel("Importance Score")
        return fig
   
    st.title("Feature Importance Visualization")
    if 'df' not in st.session_state:
        st.warning("No dataset found. Please load a dataset first!")
    else:

        df = st.session_state['df']
        feature_names = list(df.columns.drop("AQI"))
        
        st.header("Select a Model for Feature Importance")
        model_name = st.selectbox(
            "Choose a model:",
            ["Gradient Boosting", "Random Forest"]
        )

        if model_name == "Gradient Boosting":
            gradient_boosting_model = joblib.load("../model/Gradient_Boosting.joblib")
            fig = plot_feature_importance(gradient_boosting_model, "Gradient Boosting", feature_names)
            st.pyplot(fig)

        elif model_name == "Random Forest":
            random_forest_model = joblib.load("../model/Random_Forest.joblib")
            fig = plot_feature_importance(random_forest_model, "Random Forest", feature_names)
            st.pyplot(fig)
            
elif choice == "Lets Predict!":
    
    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
   
    @st.cache_resource
    def load_resources():
        models = {
            "Gradient Boosting": joblib.load("../model/Gradient_Boosting.joblib"),
            "Random Forest": joblib.load("../model/Random_Forest.joblib"),
            "Decision Tree": joblib.load("../model/Decision_Tree.joblib"),
            "Linear Regression": joblib.load("../model/Linear_Regression.joblib"),
            "Ridge Regression": joblib.load("../model/Ridge_Regression.joblib"),
            "Lasso Regression": joblib.load("../model/Lasso_Regression.joblib"),
            "Support Vector Regressor (SVR)": joblib.load("../model/Support_Vector_Regressor_(SVR).joblib"),
        }
        scaler = joblib.load("../model/standard_scalar.joblib")
        return models, scaler

    models, scaler = load_resources()

    model_choices = list(models.keys())
    selected_model = st.selectbox("Select a model for prediction:", model_choices)

    st.header("Input Feature Values")
    st.write("Adjust the sliders below to set feature values:")
    user_inputs = {}
    
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        if i % 2 == 0:
            with col1:
                user_inputs[feature] = st.slider(
                    f"{feature}:", min_value=0.0, max_value=100.0, value=2.0, step=0.01
                )
        else:
            with col2:
                user_inputs[feature] = st.slider(
                    f"{feature}:", min_value=0.0, max_value=100.0, value=5.0, step=0.01
                )

    input_df = pd.DataFrame([user_inputs])  
    scaled_input = scaler.transform(input_df)
    
    st.subheader("Input Values:")
    st.write(input_df)

    # Make prediction
    if st.button("Make Prediction"):
        model = models.get(selected_model)

        if model:
            prediction = model.predict(scaled_input)
            aqi_value = prediction[0]
            aqi_category = get_aqi_category(aqi_value)
            st.success(f"The predicted AQI is: {prediction[0]:.2f}")

            st.subheader("Prediction Output:")
            st.success(f"Predicted AQI Category: **{aqi_category}**")
            