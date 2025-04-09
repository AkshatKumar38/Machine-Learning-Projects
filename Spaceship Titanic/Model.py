import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df, is_train=True):
    df = df.copy()
    
    if not is_train:
        passenger_ids = df['PassengerId'].copy()
    
    df['Cabin'] = df['Cabin'].fillna('Unknown/Unknown/Unknown').astype(str)
    df[['Deck', 'Number', 'Side']] = df['Cabin'].apply(lambda x: x.split('/') if '/' in x else ['Unknown', 'Unknown', 'Unknown']).apply(pd.Series)

    df['Number'] = pd.to_numeric(df['Number'], errors='coerce').fillna(0).astype(int)
    df['Number_Scaled'] = (df['Number'] - df['Number'].min()) / (df['Number'].max() - df['Number'].min())
    df.drop(columns=['Number'], inplace=True)
    
    deck_mapping = {deck: idx for idx, deck in enumerate(df['Deck'].dropna().unique())}
    df['Deck'] = df['Deck'].map(deck_mapping)
    df['Side'] = df['Side'].map({'P': 1, 'S': 0}).fillna(-1)
    
    df['HomePlanet'] = df.groupby('Deck')['HomePlanet'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Earth'))
    df['Destination'] = df.groupby('HomePlanet')['Destination'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'TRAPPIST-1e'))
    df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(int)
    df['VIP'] = df['VIP'].fillna(False).astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)

    df['GroupID'] = df['Cabin'].apply(lambda x: x.split('/')[1] if '/' in x else 'Unknown')
    df['GroupSize'] = df.groupby('GroupID')['GroupID'].transform('count')
    df.drop(columns=['GroupID'], inplace=True)
    
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[spend_cols] = df[spend_cols].fillna(0)
    df['TotalSpending'] = df[spend_cols].sum(axis=1).astype(int)
    df['HighSpender'] = (df['TotalSpending'] > df['TotalSpending'].median()).astype(int)
    df['TotalSpending_Log'] = np.log1p(df['TotalSpending'])
    df.drop(columns=['TotalSpending'], inplace=True)
    
    df['CryoSleep_Spent'] = ((df['CryoSleep'] == 1) & (df['TotalSpending_Log'] > 0)).astype(int)
    df.drop(columns=['Cabin', 'Name'] + spend_cols, inplace=True)
    df = pd.get_dummies(df, columns=['Deck', 'HomePlanet', 'Destination'], drop_first=True)
    
    if is_train and 'Transported' in df.columns:
        df['Transported'] = df['Transported'].astype(int)
    
    if not is_train:
        df['PassengerId'] = passenger_ids
    
    return df

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data, is_train=False)

train_data['Age_Number'] = train_data['Age'] * train_data['Number_Scaled']
train_data['Spender_Cryo'] = train_data['HighSpender'] * train_data['CryoSleep']
train_data['Side_Deck'] = train_data['Side'] * train_data['Deck_1'] 

test_data['Age_Number'] = test_data['Age'] * test_data['Number_Scaled']
test_data['Spender_Cryo'] = test_data['HighSpender'] * test_data['CryoSleep']
test_data['Side_Deck'] = test_data['Side'] * test_data['Deck_1']

X = train_data.drop(columns=['Transported'])
y = train_data['Transported']

categorical_features = [col for col in X.columns if col.startswith(('HomePlanet_', 'Destination_', 'Deck_'))] + ['Side']
numeric_features = ['Age', 'Number_Scaled', 'TotalSpending_Log']

preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X)
print(f"Training Accuracy: {accuracy_score(y, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))

test_features = test_data.drop(columns=['PassengerId'])
test_predictions = best_model.predict(test_features)

submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': test_predictions.astype(bool)
})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")     