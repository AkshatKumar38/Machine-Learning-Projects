import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('train.csv')

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Mlle', 'Ms', 'Lady'], 'Miss')
df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
df['Title'] = df['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Countess'], 'Other')

# Encoding categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Embarked'] = le.fit_transform(df['Embarked'])
df['Title'] = le.fit_transform(df['Title'])

# Filling missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Binning Age and Fare
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3])
df['AgeBin'] = df['AgeBin'].astype(int)
df['FareBin'] = df['FareBin'].astype(int)

# Selecting features
features = ['Pclass', 'Sex', 'AgeBin', 'FareBin', 'Embarked', 'FamilySize', 'Title']
X = df[features]
y = df['Survived']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling class imbalance
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scaling numerical features
scaler = StandardScaler()
X_train[['FamilySize']] = scaler.fit_transform(X_train[['FamilySize']])
X_test[['FamilySize']] = scaler.transform(X_test[['FamilySize']])

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Training with best parameters
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Best Parameters:", grid_search.best_params_)
print("Test Accuracy:", accuracy)
