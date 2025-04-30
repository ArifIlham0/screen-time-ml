import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load processed data
df = pd.read_csv('data/processed_data.csv')

X = df[['Daily_Screen_Time_Hours', 'Age', 'Total_App_Usage_Hours']]
y = df['kategori_penggunaan']

# Encode labels
y = y.astype('category').cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'SVM': SVC()
}

best_model = None
best_score = 0
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {score}")
    
    if score > best_score:
        best_score = score
        best_model = model

# Save best model and scaler
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Model terbaik disimpan ke models/best_model.pkl")
