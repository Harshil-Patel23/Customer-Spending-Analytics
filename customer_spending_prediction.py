import pandas as pd

# Load dataset
df = pd.read_csv("Mall_Customers (1).csv")

# Drop unnecessary columns
df = df.drop("CustomerID", axis=1)

# Convert categorical variables
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Feature-target split
X = df.drop("Spending Score (1-100)", axis=1)
y = df["Spending Score (1-100)"]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

import shap

# SHAP values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Visualize feature importance
shap.summary_plot(shap_values, X_test)

