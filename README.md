# Medical Insurance Cost Prediction

A machine learning project that predicts medical insurance charges based on demographic, lifestyle, and health-related features.

## 🎯 Project Objective

Medical insurance costs can vary significantly depending on factors such as age, BMI, number of children, smoking status, sex, and geographic region.

The objective of this project is to:

- Explore the factors affecting insurance charges
- Perform data preprocessing and feature encoding
- Train multiple regression models
- Compare model performance
- Select a suitable model for insurance cost prediction
- Build a simple prediction program for new patient data

## 📊 Dataset

The project uses a medical insurance dataset containing **1,339 records** and **7 features**.

### Features

| Feature | Description |
|---|---|
| Age | Age of the individual |
| Sex | Gender of the individual |
| BMI | Body Mass Index |
| Children | Number of children/dependents |
| Smoker | Smoking status |
| Region | Geographic region |
| Charges | Medical insurance cost (target variable) |

## 🤖 Machine Learning Models

The project compares five regression algorithms:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regressor (SVR)

## 🔄 Machine Learning Workflow

```text
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
   ↓
Best Model Selection
   ↓
Insurance Cost Prediction
## 📈 Evaluation Metrics

The models are evaluated using the following metrics:

- MAE (Mean Absolute Error) – measures the average absolute difference between actual and predicted values.
- RMSE (Root Mean Squared Error) – measures the magnitude of prediction errors, giving more weight to larger errors.
- R² Score – measures how well the model explains the variation in insurance charges.

Lower MAE and RMSE values indicate better prediction performance, while a higher R² score indicates a better-fitting model.

## 🔬 Exploratory Data Analysis

The project performs exploratory analysis of:

- Age distribution
- BMI distribution
- Insurance charge distribution
- Smoker distribution
- Regional distribution
- Relationship between age and insurance charges
- Feature correlations
- Feature importance

Generated visualizations include:

- feature_importance.png
- model_comparison.png

## 🧠 Prediction System

A trained machine learning model is saved as:

insurance_model.pkl

The predict.py program allows users to enter:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The program then predicts the estimated medical insurance cost.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib

## 📁 Project Structure

medical-insurance-cost-prediction/
│
├── results/
├── .gitignore
├── feature_importance.png
├── insurance_analysis.ipynb
├── insurance_cost_prediction.py
├── insurance_dataset.csv
├── insurance_model.pkl
├── model_comparison.png
├── predict.py
├── project-presentation.pptx
├── README.md
└── requirements.txt

## ▶️ How to Run

### 1. Clone the repository

git clone https://github.com/premchand-1701/medical-insurance-cost-prediction.git

### 2. Open the project folder

cd medical-insurance-cost-prediction

### 3. Install the required libraries

pip install -r requirements.txt

### 4. Run the machine learning project

python insurance_cost_prediction.py

### 5. Run the prediction program

python predict.py

## 💡 Example Prediction

Example input:

Age: 25
Sex: male
BMI: 22.5
Children: 0
Smoker: no
Region: southeast

Example output:

Predicted Insurance Cost: ₹4598.99

The prediction is a machine learning estimate and should not be considered an actual insurance quote.

## 🚧 Current Limitations

The current project is primarily focused on learning and comparing regression models.

Possible improvements include:

- Cross-validation
- Hyperparameter tuning
- Improved preprocessing pipelines
- More systematic model selection
- Better experiment tracking
- API-based model deployment

## 🚀 Future Improvements

- Build a reusable machine learning pipeline
- Add hyperparameter optimization
- Add cross-validation
- Create a prediction API
- Build a simple web interface
- Deploy the trained model

## 👨‍💻 Author

Surabu Premchand

B.Tech CSE Student
SR University, Warangal
