### README: **Streamlit-Based AQI Prediction and Data Analysis App**

---

#### **Overview**

This Streamlit application provides an interactive interface for **Air Quality Index (AQI) analysis and prediction**. Users can load and clean datasets, visualize data, drop unwanted columns, split data, train models, analyze feature importance, and make predictions using trained machine learning models.

---

#### **Features**

1. **Load Dataset**:
   - Upload a dataset in CSV format.
   - Display the dataset with the option to select the number of rows to preview.

2. **Clean Dataset**:
   - Identify and drop rows with null values.
   - View the dataset before and after cleaning.

3. **Visualization**:
   - Univariate analysis using histograms.
   - Correlation heatmap for bivariate analysis.
   - Time series analysis for AQI trends and monthly variations in pollutant concentrations.

4. **Column Dropping**:
   - Dynamically drop columns using a multiselect dropdown.
   - Update the dataset and save changes to session state.

5. **Data Splitting and Scaling**:
   - Split data into training, validation, and test sets.
   - Standardize features using `StandardScaler` and save the scaler for later use.
   - Visualize the distributions of training, validation, and test datasets using histograms.

6. **Model Testing**:
   - Train and evaluate multiple regression models:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - Random Forest
     - Gradient Boosting
     - Decision Tree
     - Support Vector Regressor (SVR)
   - Display model parameters, validation/test metrics, and save trained models.

7. **Feature Importance**:
   - Visualize feature importance for tree-based models (Random Forest, Gradient Boosting).
   - Display bar plots to identify the most influential features.

8. **Prediction**:
   - Input feature values using sliders.
   - Select a trained model for prediction.
   - Scale input features and predict AQI values.
   - Display the predicted AQI and its corresponding category (e.g., "Good", "Moderate").

---

#### **How to Run**

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. **Open in Browser**:
   Once the app starts, a URL will be displayed in the terminal (e.g., `http://localhost:8501`). Open it in your web browser.

---

#### **Project Structure**

```plaintext
.
├── app.py                   # Main Streamlit application file
├── utils.py                 # Helper functions for data cleaning and AQI categorization
├── model/                   # Directory to save trained models and scalers
│   ├── Gradient_Boosting.joblib
│   ├── Random_Forest.joblib
│   ├── standard_scalar.joblib
│   └── ...
├── data/                    # Directory to store datasets
├── requirements.txt         # List of dependencies
└── README.md                # This README file
```

---

#### **Dependencies**

- **Streamlit**: Interactive web application framework.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **Matplotlib**: Data visualization.
- **Seaborn**: Enhanced visualizations.
- **Plotly**: Interactive visualizations.
- **Scikit-learn**: Machine learning and data preprocessing.

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

#### **How to Use**

1. **Load Dataset**:
   - Upload your CSV file and preview the data.

2. **Clean Data**:
   - Handle missing values by dropping null rows.

3. **Visualize Data**:
   - Explore univariate and bivariate relationships, as well as time series trends.

4. **Split Data**:
   - Split and scale data for machine learning.

5. **Train Models**:
   - Train, evaluate, and save multiple machine learning models.

6. **Analyze Feature Importance**:
   - Identify key contributors to AQI using feature importance plots.

7. **Predict AQI**:
   - Use sliders to input feature values and predict AQI using trained models.

---

#### **AQI Categories**

The app categorizes AQI values into the following groups:
| AQI Range         | Category                          |
|--------------------|-----------------------------------|
| 0–50              | Good                              |
| 51–100            | Moderate                         |
| 101–150           | Unhealthy for Sensitive Groups   |
| 151–200           | Unhealthy                        |
| 201–300           | Very Unhealthy                   |
| 301+              | Hazardous                        |

---

#### **Future Enhancements**

- Add support for batch predictions.
- Include model hyperparameter tuning options.
- Provide advanced visualizations for multivariate relationships.

---

#### **Contact**

For queries or contributions, please contact:

- **Name**: [Your Name]
- **Email**: [Your Email Address]
- **GitHub**: [Your GitHub Profile URL]

---

Let me know if you'd like further customizations!