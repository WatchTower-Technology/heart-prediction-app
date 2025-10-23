import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('heart.csv')

df = load_data()

# Prepare data
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Streamlit UI
st.title("Heart Disease Prediction App")

st.write("""
Enter the following patient data to predict the likelihood of heart disease.
""")

def user_input_features():
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex (1 = male; 0 = female)', [1, 0])
    cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', [1, 0])
    restecg = st.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
    thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', [1, 0])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (0-4)', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)', [1, 2, 3])
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame(data, index=[0])
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


input_df = user_input_features()

if st.button('Predict'):
    prediction = clf.predict(input_df)[0]
    probability = clf.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.error(f"High likelihood of heart disease ({probability:.2%} probability).")
    else:
        st.success(f"Low likelihood of heart disease ({probability:.2%} probability).")

# Show model accuracy
st.write(f"Model accuracy on test set: {accuracy_score(y_test, clf.predict(X_test)):.2%}")