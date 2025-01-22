import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

model_path = "./models/model.pkl"

def train_model(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(['UDI', 'Product ID', 'Type', "Target", 'Failure Type'],axis = 1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(criterion='gini', max_depth=10, max_features=None, min_samples_leaf=1, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {"accuracy": accuracy, "f1_score": f1}

def predict_downtime(input_data):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    confidence = max(model.predict_proba(df)[0])

    return {"Machine Failure": "Yes" if prediction else "No", "Confidence": round(confidence, 2)}
