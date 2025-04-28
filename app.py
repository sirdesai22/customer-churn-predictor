# --- Flask App ---
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)
curr_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(curr_dir, 'model.pkl')
scaler_file_path = os.path.join(curr_dir, 'scaler.pkl')

model = pickle.load(open(model_file_path, 'rb'))
scaler = pickle.load(open(scaler_file_path, 'rb')) # Load the scaler as well

# List of all features the model expects
feature_list = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# List of categorical columns
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

numerical_cols_for_form = ['tenure', 'MonthlyCharges', 'TotalCharges']

@app.route('/')
def home():
    return render_template('index.html', categorical_cols=categorical_cols, numerical_cols=numerical_cols_for_form)

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    input_df = pd.DataFrame([form_data])

    # Convert numerical features
    for col in numerical_cols_for_form:
        #print(col)
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

    # Handle missing TotalCharges (if it somehow becomes NaN after conversion)
    if pd.isna(input_df['TotalCharges']).any():
        input_df['TotalCharges'] = input_df['MonthlyCharges'] * input_df['tenure']

    # One-hot encode categorical features
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Align columns with the training data and fill missing with 0
    input_encoded = input_encoded.reindex(columns=feature_list, fill_value=0)

    # Scale numerical features
    numerical_features = input_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']]
    scaled_features = scaler.transform(numerical_features)
    input_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaled_features

    # Make prediction
    prediction = model.predict(input_encoded)
    probability = model.predict_proba(input_encoded)[0][1] # Probability of Churn (Yes)

    if prediction[0] == 1:
        result = f"This customer is likely to churn (Probability: {probability:.2f})"
    else:
        result = f"This customer is not likely to churn (Probability: {probability:.2f})"

    print("Predict------->",prediction)
    print("Probab------->",probability)

    return render_template('index.html', categorical_cols=categorical_cols, numerical_cols=numerical_cols_for_form, prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)