# Customer Churn Predictor

A machine learning application that predicts customer churn for a telecommunications company using historical customer data.

## Project Overview

This project implements a machine learning model to predict whether a customer is likely to churn (discontinue services) based on various customer attributes and behaviors. The application includes both the model training pipeline and a web interface for making predictions.

## Features

- Machine learning model trained on telecom customer data
- Web interface for real-time predictions
- Pre-trained model and scaler for immediate use
- Data preprocessing and feature engineering pipeline

## Project Structure

- `app.py`: Flask web application for serving predictions
- `model.ipynb`: Jupyter notebook containing model development and training
- `model.pkl`: Serialized trained machine learning model
- `scaler.pkl`: Serialized data scaler for feature normalization
- `telco-customer-churn.csv`: Dataset used for training
- `templates/`: HTML templates for the web interface

## Technologies Used

- Python
- Scikit-learn
- Flask
- Pandas
- NumPy
- Jupyter Notebook

## Getting Started

1. Clone the repository
2. Install the required dependencies
3. Run the Flask application using `python app.py`
4. Access the web interface at `http://localhost:5000`

## Model Information

The model is trained on historical customer data and considers various features such as:
- Customer demographics
- Service subscriptions
- Account information
- Usage patterns

## License

MIT License