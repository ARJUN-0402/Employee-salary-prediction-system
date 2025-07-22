import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load trained model
model = joblib.load("salary_model.pkl")  # Make sure this exists
# Load data (optional, for evaluation visuals)
data = pd.read_csv("Salary Data.csv")  # Your file name

# App Title
st.set_page_config(page_title="Salary Predictor", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Salary", "Model Evaluation"])

# ------------------ Home ------------------
if page == "Home":
    st.title("💼 Employee Salary Prediction App")
    def card_html(emoji, title, description):
        return f"""
        <div style="background-color:#f0f2f6; padding:20px; margin-bottom: 20px;
                border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.1);">
        <h5 style="color:#333333;">{emoji} {title}</h5>
        <p style="font-size: 14px;color:#666666;">{description}</p></div>"""
    st.markdown("### 🧪 Models Tried")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(card_html("📊", "Linear Regression", "Baseline model for salary prediction."), unsafe_allow_html=True)
    with col2:
        st.markdown(card_html("🌲", "Random Forest", "Ensemble model, good with nonlinearities."), unsafe_allow_html=True)
    with col3:
        st.markdown(card_html("🚀", "Gradient Boosting", "High accuracy, handles complexity well."), unsafe_allow_html=True)
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    col4, col5 = st.columns([1, 1])
    with col4:
        st.markdown(card_html("👥", "K-Nearest Neighbors", "Distance-based, simple and intuitive."), unsafe_allow_html=True)
    with col5:
        st.markdown(card_html("🎯", "Ridge / Lasso", "Regularized models to prevent overfitting."), unsafe_allow_html=True)
    st.markdown("### Final model used: Linear Regression")
    
# ------------------ Predict Salary ------------------
elif page == "Predict Salary":
    st.title("📊 Predict Employee Salary")

    # Load encoders
    job_encoder = joblib.load('encoders/job_encoder.pkl')
    edu_encoder = joblib.load('encoders/edu_encoder.pkl')
    gender_encoder = joblib.load('encoders/gender_encoder.pkl')
    job_titles = job_encoder.classes_
    education_levels = edu_encoder.classes_
    genders = gender_encoder.classes_

    # Create two columns for input layout
    col1, col2 = st.columns(2)

    with col1:
        job_title = st.selectbox("Select Job Title", job_titles)
        gender = st.selectbox("Gender", genders)
        age = st.slider("Age", 18, 65, 30)

    with col2:
        education = st.selectbox("Education Level", education_levels)
        experience = st.slider("Years of Experience", 0, 40, 5)

    # Encode input
    job_encoded = job_encoder.transform([job_title])[0]
    edu_encoded = edu_encoder.transform([education])[0]
    gender_encoded = gender_encoder.transform([gender])[0]

    if st.button("Predict"):
        input_df = pd.DataFrame([[age, gender_encoded, edu_encoded, job_encoded, experience]],
                                columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Salary: ₹{round(prediction, 2)}")


# ------------------ Model Evaluation ------------------
elif page == "Model Evaluation":
    st.title("📈 Model Evaluation Metrics")
    st.subheader("📊 Model Evaluation Metrics")
    try:
        results_df = pd.read_csv("results_df.csv") 
        np.float=float 
        fig, axs = plt.subplots(2, 2, figsize=(16,12))
        metrics = ["MSE", "MAE", "RMSE", "R2 Score"]
        colors = ["Reds", "Oranges", "coolwarm", "Reds"]  # Valid Seaborn palettes


        for i, metric in enumerate(metrics):
            ax = axs[i // 2, i % 2]
            sns.barplot(x="Model", y=metric, data=results_df, ax=ax, palette=colors[i])
            ax.set_title(f"{metric} by Model")
            ax.tick_params(axis='x', rotation=15)

        st.pyplot(fig)
        fig.tight_layout(pad=4.0)
    except FileNotFoundError:
        st.warning("📁 No evaluation metrics file found. Please upload 'results_df.csv'.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Made with ❤️ by <b>Arjun</b></p>", unsafe_allow_html=True)
