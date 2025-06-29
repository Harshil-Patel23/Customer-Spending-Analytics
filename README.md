# Customer Spending Analytics Dashboard

[![Live App](https://img.shields.io/badge/Live%20Demo-Streamlit-green?logo=streamlit)](https://customer-spending-analytics.streamlit.app/)

A modern, interactive dashboard for predicting and analyzing customer spending behavior using machine learning. Built with Streamlit, scikit-learn, SHAP, and Plotly, this app enables both single and batch predictions, model comparison, feature importance analysis, and exploratory data visualization.

## Features

- **Predict Customer Spending Score**: Input customer details to predict their spending score (1-100) using multiple ML models.
- **Model Comparison**: Compare Random Forest, Gradient Boosting, Linear Regression, and SVR models with cross-validation scores.
- **Feature Importance (SHAP)**: Visualize and interpret model predictions using SHAP values.
- **Exploratory Data Analysis**: Interactive charts for age, income, gender, and spending score distributions, plus correlation heatmaps.
- **Batch Prediction**: Upload a CSV file for batch predictions and download results with spending categories.
- **Custom Styling**: Clean, modern UI with custom CSS and responsive layout.

## Demo

<!-- ![Dashboard Screenshot](demo_screenshot.png) Add your screenshot here -->

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Customer-Spending-Analytics.git
cd Customer-Spending-Analytics
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## File Structure

- `app.py` — Main Streamlit dashboard application
- `customer_spending_prediction.py` — Standalone script for model training and SHAP analysis
- `Mall_Customers (1).csv` — Sample dataset
- `requirements.txt` — Python dependencies
- `customer_spending_prediction.ipynb` — (Optional) Jupyter notebook for experimentation

## Dataset

The app uses the Mall Customers dataset with the following columns:

- `CustomerID`: Unique customer identifier
- `Gender`: Male/Female
- `Age`: Customer age
- `Annual Income (k$)`: Annual income in thousands
- `Spending Score (1-100)`: Spending score assigned by the mall

## Usage

- **Single Prediction**: Use the sidebar to enter customer details and select a model.
- **Batch Prediction**: Upload a CSV file with columns: `Gender`, `Age`, `Annual Income (k$)`.
- **Model Analysis**: Explore model performance, feature importance, and data insights via dashboard tabs.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [SHAP](https://shap.readthedocs.io/)
- [Plotly](https://plotly.com/python/)

## License

This project is licensed under the MIT License.
