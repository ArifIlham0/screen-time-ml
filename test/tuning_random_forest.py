import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = joblib.load('data/preprocessed_data.pkl')

print("Distribusi label pada y_train:", Counter(y_train))

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

best_rf = grid_search.best_estimator_

preds = best_rf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, preds))

joblib.dump(best_rf, 'models/best_rf_model.pkl')

importances = best_rf.feature_importances_

indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for i in range(len(importances)):
    print(f"{i+1}. Feature {indices[i]} ({importances[indices[i]]:.4f})")

top_features = indices[:20]

X_train_selected = X_train[:, top_features]
X_test_selected = X_test[:, top_features]

best_rf.fit(X_train_selected, y_train)
preds_selected = best_rf.predict(X_test_selected)

acc_selected = accuracy_score(y_test, preds_selected)
print(f"Test Accuracy dengan fitur terpilih: {acc_selected:.4f}")
print("Classification Report dengan fitur terpilih:")
print(classification_report(y_test, preds_selected))