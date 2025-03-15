import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file_path = "dataset.csv"  
df = pd.read_csv(file_path)

df = df.dropna()

X = df.drop(columns=['price_range'])  
y = df['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)


print(f"Model Accuracy: {accuracy:.2f}")
print("Best Parameters:", grid_search.best_params_)


new_phone = [[1500, 1, 2.0, 1, 5, 1, 64, 0.5, 140, 4, 8, 1280, 1920, 4000, 12, 6, 15, 1, 1, 1]]
new_phone_scaled = scaler.transform(new_phone)
predicted_price_range = best_model.predict(new_phone_scaled)
print("Predicted Price Range:", predicted_price_range)
