import joblib
import pandas as pd
from pathlib import Path

model_path = Path(__file__).resolve().parent / "insurance_model.pkl"

model = joblib.load(model_path)

print("Medical Insurance Cost Prediction")
print("-" * 40)

age = int(input("Enter age: "))
sex = input("Enter sex (male/female): ").lower()
bmi = float(input("Enter BMI: "))
children = int(input("Enter number of children: "))
smoker = input("Are you a smoker? (yes/no): ").lower()
region = input("Enter region (southwest/southeast/northwest/northeast): ").lower()

data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

prediction = model.predict(data)[0]

print("\nEstimated Medical Insurance Cost: ${:,.2f}".format(prediction))