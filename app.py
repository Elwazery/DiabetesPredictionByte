import streamlit as st
import pandas as pd
import joblib
import shap
from catboost import CatBoostClassifier

# Load the trained model
try:
    model = joblib.load(r"C:\DiabetesPredictionByte\diabetes_final_model.pkl")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Verify categorical features
cat_features = model.get_param('cat_features')
if cat_features is None:
    st.warning("⚠️ Model does not specify categorical features. Assuming ['age_bin', 'BMI_Class'].")
    cat_features = ['age_bin', 'BMI_Class']

# Initialize SHAP explainer
try:
    explainer = shap.TreeExplainer(model)
except Exception as e:
    st.error(f"❌ Failed to initialize SHAP explainer: {e}")
    st.stop()

st.title("🔬 Diabetes Prediction & Explanation App")

st.markdown("""
This app uses a machine learning model to predict the probability of having diabetes 
and explains the top 5 most influential features using SHAP values.
""")

# Sidebar input
st.sidebar.header("📝 Enter Patient Information")
def user_input():
    # Age group selection as string
    Age_group = st.sidebar.selectbox("Age Group", [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'
    ])

    # Map age group to numeric Age column
    age_dict = {
        '18-24': 1, '25-29': 2, '30-34': 3, '35-39': 4,
        '40-44': 5, '45-49': 6, '50-54': 7, '55-59': 8,
        '60-64': 9, '65-69': 10, '70-74': 11, '75-79': 12, '80+': 13
    }
    Age = age_dict[Age_group]

    # Other inputs using textual values
    BMI = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    HighBP = st.sidebar.selectbox("High Blood Pressure", ["No", "Yes"])
    HighChol = st.sidebar.selectbox("High Cholesterol", ["No", "Yes"])
    Smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
    Stroke = st.sidebar.selectbox("Stroke", ["No", "Yes"])
    HeartDiseaseorAttack = st.sidebar.selectbox("Heart Disease or Attack", ["No", "Yes"])
    PhysActivity = st.sidebar.selectbox("Physical Activity", ["No", "Yes"])
    Fruits = st.sidebar.selectbox("Eats Fruits", ["No", "Yes"])
    Veggies = st.sidebar.selectbox("Eats Vegetables", ["No", "Yes"])
    GenHlth = st.sidebar.selectbox("General Health", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
    Sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
    DiffWalk = st.sidebar.selectbox("Difficulty Walking", ["No", "Yes"])

    # Default values for features not collected via user input
    CholCheck = 1  # Assume most people had cholesterol checked
    AnyHealthcare = 1  # Assume most have healthcare
    NoDocbcCost = 0  # Assume no cost barrier
    MentHlth = 0  # Assume median mental health days (no issues)
    PhysHlth = 0  # Assume median physical health days (no issues)

    # Map textual inputs to numeric values for model
    binary_map = {"No": 0, "Yes": 1}
    gen_health_map = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
    sex_map = {"Female": 0, "Male": 1}

    # Calculate BMI class
    if BMI < 18.5:
        BMI_Class = '1'  # Underweight
    elif BMI <= 24.9:
        BMI_Class = '2'  # Normal
    elif BMI <= 29.9:
        BMI_Class = '3'  # Overweight
    elif BMI <= 34.9:
        BMI_Class = '4'  # Obesity Class 1
    elif BMI <= 39.9:
        BMI_Class = '5'  # Obesity Class 2
    else:
        BMI_Class = '6'  # Obesity Class 3

    # Build input data with exact column order
    data = {
        'HighBP': binary_map[HighBP],
        'HighChol': binary_map[HighChol],
        'CholCheck': CholCheck,
        'BMI': BMI,
        'Smoker': binary_map[Smoker],
        'Stroke': binary_map[Stroke],
        'HeartDiseaseorAttack': binary_map[HeartDiseaseorAttack],
        'PhysActivity': binary_map[PhysActivity],
        'Fruits': binary_map[Fruits],
        'Veggies': binary_map[Veggies],
        'AnyHealthcare': AnyHealthcare,
        'NoDocbcCost': NoDocbcCost,
        'GenHlth': gen_health_map[GenHlth],
        'MentHlth': MentHlth,
        'PhysHlth': PhysHlth,
        'DiffWalk': binary_map[DiffWalk],
        'Sex': sex_map[Sex],
        'Age': Age,
        'age_bin': Age_group,  # String
        'BMI_Class': BMI_Class  # String
    }

    # Define expected column order (excluding HvyAlcoholConsump, Education, Income)
    columns = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
        'PhysActivity', 'Fruits', 'Veggies', 'AnyHealthcare', 'NoDocbcCost',
        'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'age_bin', 'BMI_Class'
    ]
    return pd.DataFrame([data])[columns]

# Get input
input_df = user_input()

# Predict and explain
if st.button("🔍 Predict Now"):
    try:
        # Ensure categorical features are strings
        input_df['age_bin'] = input_df['age_bin'].astype(str)
        input_df['BMI_Class'] = input_df['BMI_Class'].astype(str)

        # Verify input data matches expected features
        expected_features = model.feature_names_
        if set(input_df.columns) != set(expected_features):
            st.error(f"❌ Input features do not match model expectations.\nExpected: {expected_features}\nGot: {list(input_df.columns)}")
            st.stop()

        # Predict probability
        prob = model.predict_proba(input_df)[0][1] * 100
        st.success(f"📈 Probability of Having Diabetes: **{prob:.2f}%**")

        # SHAP explanation
        shap_values = explainer.shap_values(input_df)

        # Handle SHAP values for binary classification (use class 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take values for positive class

        st.subheader("📌 Top 5 Influential Features:")
        shap_df = pd.DataFrame({
            'Feature': input_df.columns,
            'SHAP_Impact': shap_values[0]
        })
        shap_df['Impact_%'] = (shap_df['SHAP_Impact'].abs() / shap_df['SHAP_Impact'].abs().sum()) * 100
        shap_df = shap_df.sort_values(by='Impact_%', ascending=False).head(5)

        for idx, row in shap_df.iterrows():
            st.write(f"- **{row['Feature']}**: Impact = {row['Impact_%']:.2f}%")

        # Bar chart for SHAP impact
        st.subheader("📊 SHAP Impact Bar Chart:")
        chart_data = shap_df.set_index('Feature')['Impact_%']
        st.bar_chart(chart_data)

    except Exception as e:
        st.error(f"❌ Error during prediction or SHAP computation:\n\n{e}")