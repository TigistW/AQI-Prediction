import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


def load_dataset():
    file_path = 'dataset/city_day.csv'
    df = pd.read_csv(file_path)
    return df

def null_values(df):
    null_val = df.isna().sum()
    return null_val

def drop_null_values(df):
    df = df.dropna(inplace=True)
    return df

def univariate_analysis(df):
    df.hist(figsize=(20,20))
    plt.show()

def univariate_analysis_seaborn(df, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Univariate Analysis of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    return fig

def get_aqi_category(aqi_value):

    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"