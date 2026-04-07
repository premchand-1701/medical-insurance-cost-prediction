"""
Medical Insurance Cost Prediction Using Machine Learning
=========================================================
This script implements and compares 5 ML regression models:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regressor (SVR)

Dataset: insurance_dataset.csv (1338 records, 7 features)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── 1. Load Dataset ───────────────────────────────────────────────
df = pd.read_csv('insurance_dataset.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Info:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# ─── 2. Exploratory Data Analysis ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Feature Distributions', fontsize=16)

axes[0, 0].hist(df['age'], bins=20, color='#065A82', edgecolor='white')
axes[0, 0].set_title('Age Distribution')

axes[0, 1].hist(df['bmi'], bins=20, color='#1C7293', edgecolor='white')
axes[0, 1].set_title('BMI Distribution')

axes[0, 2].hist(df['charges'], bins=30, color='#21295C', edgecolor='white')
axes[0, 2].set_title('Charges Distribution')

df['smoker'].value_counts().plot(kind='bar', ax=axes[1, 0], color=['#065A82', '#F96167'])
axes[1, 0].set_title('Smoker Count')

df['region'].value_counts().plot(kind='bar', ax=axes[1, 1], color='#1C7293')
axes[1, 1].set_title('Region Count')

axes[1, 2].scatter(df['age'], df['charges'], alpha=0.5, c=df['smoker'].map({'yes': '#F96167', 'no': '#065A82'}), s=10)
axes[1, 2].set_title('Age vs Charges')
axes[1, 2].set_xlabel('Age')
axes[1, 2].set_ylabel('Charges')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150)
plt.show()
print("EDA plots saved to eda_plots.png")

# ─── 3. Data Preprocessing ─────────────────────────────────────────
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])          # female=0, male=1
df['smoker'] = le_smoker.fit_transform(df['smoker'])  # no=0, yes=1
df['region'] = le_region.fit_transform(df['region'])  # alphabetical encoding

print("\nEncoded Dataset:")
print(df.head())

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.show()

# ─── 4. Train-Test Split ───────────────────────────────────────────
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Scale features for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─── 5. Model Training & Evaluation ────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'Support Vector Regressor': SVR(kernel='rbf', C=10000, gamma=0.001)
}

results = []
print("\n" + "="*70)
print("MODEL EVALUATION RESULTS")
print("="*70)

for name, model in models.items():
    if name == 'Support Vector Regressor':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R² Score': r2})

    print(f"\n{name}")
    print(f"  MAE:      ${mae:,.2f}")
    print(f"  RMSE:     ${rmse:,.2f}")
    print(f"  R² Score: {r2:.4f}")

# ─── 6. Results Comparison ─────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n\nComparison Table:")
print(results_df.to_string(index=False))

# Bar chart of R² scores
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#065A82', '#1C7293', '#21295C', '#02C39A', '#F96167']
bars = ax.bar(results_df['Model'], results_df['R² Score'], color=colors, edgecolor='white', linewidth=1.5)
ax.set_ylabel('R² Score')
ax.set_title('Model Comparison - R² Score')
ax.set_ylim(0, 1)
for bar, val in zip(bars, results_df['R² Score']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', ha='center', fontsize=10)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

print(f"\nBest Model: {results_df.loc[results_df['R² Score'].idxmax(), 'Model']} "
      f"(R² = {results_df['R² Score'].max():.4f})")
print("\nAll plots saved. Project complete!")
