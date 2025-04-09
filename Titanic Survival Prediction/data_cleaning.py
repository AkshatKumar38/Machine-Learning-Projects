import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

df = pd.read_csv('Titanic Survival Prediction/Titanic-Dataset.csv')

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Remove inplace=True

df['Deck'] = df['Cabin'].str.slice(0,1) 
df['Deck'] = df['Deck'].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F":6,"G":7, "T":8})
df['Deck'] = df['Deck'].fillna(0)
df['Deck'] = df['Deck'].astype(np.int64)

df.drop(columns=['Cabin'], inplace=True)

df['Sex'] = df['Sex'].map({'male': 0,'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0,'C': 1, 'Q': 2})
df['Embarked'] = df['Embarked'].astype(int)

df.dropna(inplace=True)

df_num_data = df.select_dtypes(include=[np.number])
sns.heatmap(df_num_data.corr(), cmap='YlGnBu')
plt.show()

# Function to extract title from name
def extract_title(name):
    title_search = re.search(r'(\w+)\.', name)
    return title_search.group(1) if title_search else "Unknown"

if 'Name' in df.columns:
    df['Title'] = df['Name'].apply(extract_title)

title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Mme": "Mrs", "Countess": "Rare", "Sir": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Lady": "Rare", "Capt": "Rare"
}

df['Title'] = df['Title'].map(title_map)
df['Title'] = df['Title'].astype('category').cat.codes 

df['Family_size'] = df['SibSp'] + df['Parch'] + 1

df['Age_group'] = pd.cut(df['Age'], bins=[0, 12, 18, 40, 60, 100], labels=[0, 1, 2, 3, 4])
df['Age_group'] = df['Age_group'].astype(int)

df['Fare_group'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3])
df['Fare_group'] = df['Fare_group'].astype(int)

df = df.drop(columns=['Fare', 'Ticket', 'Parch', 'SibSp', 'Age', 'Name'])

df.to_csv('Titanic Survival Prediction/Cleaned-Titanic-Dataset.csv')

features = ['Pclass', 'Sex', 'Deck', 'Age_group', 'Family_size', 'Fare_group', 'Embarked', 'Title']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,  # Prevent overfitting
    gamma=0.2,  # Regularization
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).plot(kind='bar', title="Feature Importance")
print(importance)
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Get predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()