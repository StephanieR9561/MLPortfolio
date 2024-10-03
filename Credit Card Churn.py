# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('/mnt/data/your_dataset.csv')

# Explore the dataset
print(data.head())
print(data.info())
print(data.describe())

# Visualize the dataset
sns.countplot(x='Attrition_Flag', data=data)
plt.show()

# Encode categorical variables
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Education_Level'] = le.fit_transform(data['Education_Level'])
data['Marital_Status'] = le.fit_transform(data['Marital_Status'])
data['Income_Category'] = le.fit_transform(data['Income_Category'])
data['Card_Category'] = le.fit_transform(data['Card_Category'])

# Separate features and target variable
X = data.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1)
y = data['Attrition_Flag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to build, evaluate and optimize models
def build_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f'Best Parameters: {grid.best_params_}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return best_model

# Define models and parameters for GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC()
}

params = {
    'LogisticRegression': {'C': [0.01, 0.1, 1, 10, 100]},
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

# Build and evaluate models on original data
for model_name in models:
    print(f'Evaluating {model_name} on original data')
    build_evaluate_model(models[model_name], params[model_name], X_train, y_train, X_test, y_test)

# Build and evaluate models on oversampled data
sm = SMOTE(random_state=42)
X_train_oversample, y_train_oversample = sm.fit_resample(X_train, y_train)

for model_name in models:
    print(f'Evaluating {model_name} on oversampled data')
    build_evaluate_model(models[model_name], params[model_name], X_train_oversample, y_train_oversample, X_test, y_test)

# Build and evaluate models on undersampled data
rus = RandomUnderSampler(random_state=42)
X_train_undersample, y_train_undersample = rus.fit_resample(X_train, y_train)

for model_name in models:
    print(f'Evaluating {model_name} on undersampled data')
    build_evaluate_model(models[model_name], params[model_name], X_train_undersample, y_train_undersample, X_test, y_test)

# Based on the evaluations, choose the best three models and tune them further
# Example: Assuming RandomForest performed well

best_models = [
    ('RandomForest', RandomForestClassifier(), {'n_estimators': [100, 200], 'max_depth': [None, 20, 30]}),
    ('LogisticRegression', LogisticRegression(), {'C': [1, 10]}),
    ('SVM', SVC(), {'C': [1, 10], 'kernel': ['linear']})
]

for model_name, model, param_grid in best_models:
    print(f'Tuning {model_name}')
    build_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test)

# Generate insights and recommendations
# This part involves interpreting the results and suggesting actionable items for the bank
