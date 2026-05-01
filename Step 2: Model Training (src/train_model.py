import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from data_preprocessing import load_data, preprocess_data

df = load_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

with open('models/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
