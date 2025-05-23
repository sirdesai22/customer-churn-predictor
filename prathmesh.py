# -*- coding: utf-8 -*-
"""Prathmesh.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DkZ4gP-3xaQZWBVROtFL7CUBJlzZatcJ
"""

# Predict if a customer will leave ("churn") based on their personal and account data.
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv("/content/telco-customer-churn.csv")

df.head()

df.info()

df.shape

df.isnull().sum()

df['TotalCharges']

# Convert 'TotalCharges' to numeric (invalid parsing will be set as NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df['TotalCharges'].dtype

df['TotalCharges'].isnull().sum()

# Fill missing 'TotalCharges' values with the median
if df['TotalCharges'].isnull().any():
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)

# Drop 'customerID' column if it exists
if 'customerID' in df.columns:
    data = df.drop('customerID', axis=1, inplace=True)
else:
    print("Column 'customerID' not found")

df.columns

# List of categorical columns
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod'
                ]

# Check if all categorical columns exist before encoding
for col in categorical_cols:
    if col not in df.columns:
            print(f"Warning: {col} not found in data!")

df.shape

# Encode categorical variables
data_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

data_encoded.shape

# Encode target 'Churn' (Yes -> 1, No -> 0)
if 'Churn' in data_encoded.columns:
    data_encoded['Churn'] = data_encoded['Churn'].map({'No': 0, 'Yes': 1})
else:
     raise ValueError("Target column 'Churn' is missing from the dataset!")

data_encoded

data_encoded.columns

# Separate features and target
X = data_encoded.drop('Churn', axis=1, )
y = data_encoded['Churn']

y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

for col in numerical_cols:
      if col not in X_train.columns:
              raise ValueError(f"Expected numerical column {col} not found!")
      X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
      X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Print X_train to check (optional)
print(X_train)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model using pickle
# model_filename = os.path.join(script_dir, 'model.pkl')
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Trained model saved as '{model_filename}'")