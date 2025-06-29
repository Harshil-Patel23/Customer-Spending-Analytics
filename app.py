
# import streamlit as st
# import pandas as pd
# import numpy as np
# import shap
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

# # Load and preprocess data
# df = pd.read_csv("Mall_Customers (1).csv")
# df = df.drop("CustomerID", axis=1)
# df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
# X = df.drop("Spending Score (1-100)", axis=1)
# y = df["Spending Score (1-100)"]
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X, y)

# # App Title
# st.title("💳 Customer Spending Score Prediction App")

# # Sidebar Input
# st.sidebar.header("Enter Customer Details")

# def user_input_features():
#     gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
#     age = st.sidebar.slider("Age", 18, 70, 30)
#     income = st.sidebar.slider("Annual Income (k$)", 15, 150, 60)
#     data = {
#         'Gender': 1 if gender == 'Female' else 0,
#         'Age': age,
#         'Annual Income (k$)': income
#     }
#     return pd.DataFrame(data, index=[0])

# input_df = user_input_features()
# prediction = model.predict(input_df)[0]
# st.write(f"### 🎯 Predicted Spending Score: `{round(prediction, 2)}`")

# # SHAP
# explainer = shap.Explainer(model, X)
# shap_values = explainer(X)
# st.subheader("Feature Importance (SHAP Summary)")
# fig, ax = plt.subplots()
# shap.summary_plot(shap_values, X, plot_type="bar", show=False)
# st.pyplot(fig)

# # Upload feature
# st.subheader("📤 Upload Your Dataset (Optional)")
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# if uploaded_file is not None:
#     user_data = pd.read_csv(uploaded_file)
#     user_data['Gender'] = user_data['Gender'].map({'Male': 0, 'Female': 1})
#     user_data = user_data[X.columns]
#     predictions = model.predict(user_data)
#     user_data['Predicted Spending Score'] = predictions
#     st.write("### Actual vs. Predicted")
#     st.write(user_data)
#     sns.scatterplot(x='Annual Income (k$)', y='Predicted Spending Score', data=user_data)
#     st.pyplot()

# # Custom Styling
# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #F9F9F9;
#         font-family: 'Arial';
#     }
#     </style>
#     """, unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        width: fit-content;
        height:fit-content;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .feature-importance {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset with caching"""
    try:
        df = pd.read_csv("Mall_Customers (1).csv")
    except FileNotFoundError:
        # Generate sample data if file not found
        np.random.seed(42)
        n_samples = 200
        df = pd.DataFrame({
            'CustomerID': range(1, n_samples + 1),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 71, n_samples),
            'Annual Income (k$)': np.random.randint(15, 151, n_samples),
            'Spending Score (1-100)': np.random.randint(1, 101, n_samples)
        })
    
    df = df.drop("CustomerID", axis=1)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    return df

@st.cache_data
def train_models(X_train, y_train):
    """Train multiple models and return them"""
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
    }
    
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        if name == 'SVR':
            # Scale features for SVR
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            model.fit(X_scaled, y_train)
            trained_models[name] = (model, scaler)
        else:
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        model_scores[name] = cv_scores.mean()
    
    return trained_models, model_scores

def create_prediction_confidence_interval(model, X, n_bootstrap=100):
    """Create confidence interval for predictions using bootstrap"""
    predictions = []
    n_samples = len(X)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X.iloc[indices]
        pred = model.predict(X_bootstrap)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    lower_bound = np.percentile(predictions, 2.5, axis=0)
    upper_bound = np.percentile(predictions, 97.5, axis=0)
    
    return lower_bound, upper_bound

def main():
    # Header
    st.title("💳 Customer Spending Analytics Dashboard")
    st.markdown("### Predict customer spending behavior with AI-powered insights")
    
    # Load data
    df = load_and_preprocess_data()
    X = df.drop("Spending Score (1-100)", axis=1)
    y = df["Spending Score (1-100)"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    trained_models, model_scores = train_models(X_train, y_train)
    
    # Sidebar
    st.sidebar.header("🎯 Prediction Settings")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model", 
        list(trained_models.keys()),
        help="Choose the machine learning model for prediction"
    )
    
    selected_model = trained_models[model_choice]
    
    # User inputs
    st.sidebar.subheader("Customer Information")
    
    def user_input_features():
        gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
        age = st.sidebar.slider("Age", 18, 70, 30, help="Customer's age")
        income = st.sidebar.slider(
            "Annual Income (k$)", 
            15, 150, 60, 
            help="Customer's annual income in thousands"
        )
        
        data = {
            'Gender': 1 if gender == 'Female' else 0,
            'Age': age,
            'Annual Income (k$)': income
        }
        return pd.DataFrame(data, index=[0])
    
    input_df = user_input_features()
    
    # Make prediction
    if model_choice == 'SVR':
        model, scaler = selected_model
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
    else:
        prediction = selected_model.predict(input_df)[0]
    
    # Main content area
    col1, col2, col3 = st.columns([2, 1, 2])
    
    
    st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Predicted Score</h2>
            <h1>{round(prediction, 1)}</h1>
            <p>Spending Score (1-100)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Performance Dashboard
    st.subheader("📊 Model Performance Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics for selected model
    if model_choice == 'SVR':
        model, scaler = selected_model
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = selected_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    with col1:
        st.metric("R² Score", f"{r2:.3f}", f"{model_scores[model_choice]:.3f}")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}")
    with col3:
        st.metric("MAE", f"{mae:.2f}")
    with col4:
        st.metric("Cross-Val Score", f"{model_scores[model_choice]:.3f}")
    
    # Visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Model Comparison", 
        "🔍 Feature Analysis", 
        "📊 Data Insights", 
        "🎯 Prediction Analysis",
        "📤 Batch Prediction"
    ])
    
    with tab1:
        st.subheader("Model Performance Comparison")
        
        # Model comparison chart
        comparison_df = pd.DataFrame({
            'Model': list(model_scores.keys()),
            'R² Score': list(model_scores.values())
        })
        
        fig = px.bar(
            comparison_df, 
            x='Model', 
            y='R² Score',
            title="Model Performance Comparison (Cross-Validation R² Score)",
            color='R² Score',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Actual vs Predicted scatter plot
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                x=y_test, 
                y=y_pred,
                title=f"Actual vs Predicted - {model_choice}",
                labels={'x': 'Actual Spending Score', 'y': 'Predicted Spending Score'}
            )
            fig.add_shape(
                type="line",
                x0=y_test.min(), y0=y_test.min(),
                x1=y_test.max(), y1=y_test.max(),
                line=dict(dash="dash", color="red")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residuals plot
            residuals = y_test - y_pred
            fig = px.scatter(
                x=y_pred, 
                y=residuals,
                title="Residuals Plot",
                labels={'x': 'Predicted Values', 'y': 'Residuals'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("SHAP Feature Importance Analysis")
        
        # SHAP analysis
        if model_choice != 'SVR':  # SHAP works better with tree-based models
            explainer = shap.Explainer(selected_model, X_train)
            shap_values = explainer(X_test[:50])  # Limit for performance
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Feature Importance (SHAP Summary)**")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test[:50], plot_type="bar", show=False)
                st.pyplot(fig)
            
            with col2:
                st.write("**SHAP Values Distribution**")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test[:50], show=False)
                st.pyplot(fig)
            
            # Individual prediction explanation
            st.write("**Your Prediction Explanation**")
            if model_choice != 'SVR':
                shap_values_single = explainer(input_df)
                fig, ax = plt.subplots(figsize=(10, 4))
                shap.waterfall_plot(shap_values_single[0], show=False)
                st.pyplot(fig)
        else:
            st.info("SHAP analysis is not available for SVR model. Please select a tree-based model.")
    
    with tab3:
        st.subheader("Dataset Insights & Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Age Distribution', 'Income Distribution', 
                               'Spending Score Distribution', 'Gender Distribution'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Histogram(x=df['Age'], name='Age', nbinsx=20),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(x=df['Annual Income (k$)'], name='Income', nbinsx=20),
                row=1, col=2
            )
            fig.add_trace(
                go.Histogram(x=df['Spending Score (1-100)'], name='Spending Score', nbinsx=20),
                row=2, col=1
            )
            
            gender_counts = df['Gender'].value_counts()
            fig.add_trace(
                go.Bar(x=['Male', 'Female'], y=[gender_counts[0], gender_counts[1]], name='Gender'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text="Data Distribution Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation heatmap
            correlation_matrix = df.corr()
            fig = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.write("**Dataset Summary Statistics**")
            st.dataframe(df.describe(), use_container_width=True)
    
    with tab4:
        st.subheader("Prediction Analysis & Sensitivity")
        
        # Sensitivity analysis
        st.write("**How predictions change with different inputs:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age sensitivity
            ages = np.arange(18, 71, 5)
            age_predictions = []
            
            for age in ages:
                temp_df = input_df.copy()
                temp_df['Age'] = age
                if model_choice == 'SVR':
                    temp_scaled = scaler.transform(temp_df)
                    pred = model.predict(temp_scaled)[0]
                else:
                    pred = selected_model.predict(temp_df)[0]
                age_predictions.append(pred)
            
            fig = px.line(
                x=ages, 
                y=age_predictions,
                title="Spending Score vs Age",
                labels={'x': 'Age', 'y': 'Predicted Spending Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Income sensitivity
            incomes = np.arange(15, 151, 10)
            income_predictions = []
            
            for income in incomes:
                temp_df = input_df.copy()
                temp_df['Annual Income (k$)'] = income
                if model_choice == 'SVR':
                    temp_scaled = scaler.transform(temp_df)
                    pred = model.predict(temp_scaled)[0]
                else:
                    pred = selected_model.predict(temp_df)[0]
                income_predictions.append(pred)
            
            fig = px.line(
                x=incomes, 
                y=income_predictions,
                title="Spending Score vs Income",
                labels={'x': 'Annual Income (k$)', 'y': 'Predicted Spending Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("📤 Batch Prediction & Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch predictions", 
            type="csv",
            help="Upload a CSV file with columns: Gender, Age, Annual Income (k$)"
        )
        
        if uploaded_file is not None:
            try:
                user_data = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = ['Gender', 'Age', 'Annual Income (k$)']
                if all(col in user_data.columns for col in required_cols):
                    # Preprocess
                    user_data['Gender'] = user_data['Gender'].map({'Male': 0, 'Female': 1})
                    user_data_processed = user_data[required_cols]
                    
                    # Make predictions
                    if model_choice == 'SVR':
                        user_data_scaled = scaler.transform(user_data_processed)
                        predictions = model.predict(user_data_scaled)
                    else:
                        predictions = selected_model.predict(user_data_processed)
                    
                    # Add predictions to dataframe
                    user_data['Predicted_Spending_Score'] = predictions
                    user_data['Spending_Category'] = pd.cut(
                        predictions, 
                        bins=[0, 30, 60, 100], 
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    st.success(f"✅ Successfully processed {len(user_data)} records!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Prediction Results**")
                        st.dataframe(user_data, use_container_width=True)
                        
                        # Download button
                        csv = user_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="spending_predictions.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Visualization of batch results
                        fig = px.histogram(
                            user_data, 
                            x='Predicted_Spending_Score',
                            title="Distribution of Predicted Spending Scores",
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Category distribution
                        category_counts = user_data['Spending_Category'].value_counts()
                        fig = px.pie(
                            values=category_counts.values,
                            names=category_counts.index,
                            title="Spending Categories Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                else:
                    st.error("❌ Please ensure your CSV has columns: Gender, Age, Annual Income (k$)")
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        else:
            # Sample data template
            st.info("💡 **Need a template?** Download the sample format below:")
            sample_data = pd.DataFrame({
                'Gender': ['Male', 'Female', 'Male', 'Female'],
                'Age': [25, 35, 45, 28],
                'Annual Income (k$)': [50, 75, 100, 60]
            })
            
            csv_sample = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=csv_sample,
                file_name="sample_customer_data.csv",
                mime="text/csv"
            )
            
            st.dataframe(sample_data, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666; padding: 1rem;'>
        <p>💳 Customer Spending Analytics Dashboard | Built with Streamlit & Machine Learning</p>
        <p>📊 Multiple ML Models | 🔍 SHAP Explanations | 📈 Interactive Visualizations</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()