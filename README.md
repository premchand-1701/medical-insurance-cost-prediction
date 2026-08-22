# Medical Insurance Cost Prediction

A machine learning project that predicts medical insurance charges using demographic, lifestyle, and health-related features.

## Problem Statement

Medical insurance costs can vary significantly based on factors such as age, BMI, number of children, smoking status, sex, and geographic region.

This project explores these relationships and compares multiple regression algorithms to predict insurance charges.

## Dataset

The dataset contains 1,339 records and 7 features.

### Features

- Age
- Sex
- BMI
- Children
- Smoker
- Region
- Charges — target variable

## Machine Learning Models

The project compares five regression models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regressor (SVR)

## Workflow

Dataset
↓
Data Exploration
↓
Data Preprocessing
↓
Feature Encoding
↓
Train/Test Split
↓
Feature Scaling
↓
Model Training
↓
Model Evaluation
↓
Model Comparison

## Evaluation Metrics

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Exploratory Data Analysis

The project analyzes:

- Age distribution
- BMI distribution
- Insurance charge distribution
- Smoker distribution
- Regional distribution
- Relationship between age and insurance charges
- Feature correlations

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Repository Structure

medical-insurance-cost-prediction/
│
├── insurance_cost_prediction.py
├── insurance_dataset.csv
├── Untitled2.ipynb
├── Insurance_Cost_Prediction_ML.pptx
└── README.md

## How to Run

Clone the repository:

git clone https://github.com/premchand-1701/medical-insurance-cost-prediction.git

Install the required Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

Run the project:

python insurance_cost_prediction.py

## Current Limitations

This project is primarily focused on learning and comparing regression models. Future improvements can include:

- Cross-validation
- Improved preprocessing pipelines
- Hyperparameter tuning
- More systematic model selection
- Better experiment tracking
- Model deployment through an API

## Future Improvements

- Build a reusable machine learning pipeline
- Add hyperparameter optimization
- Add cross-validation
- Create a prediction API
- Build a simple web interface
- Deploy the trained model

## Author

**Surabu Premchand**

B.Tech CSE Student  
SR University, Warangal
