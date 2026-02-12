# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
df = pd.read_csv("ragasense_train.csv")

# 2. Select features & label
target_col = "label"     # or "label" if you prefer
exclude_cols = ["id", "label"]
X = df.drop(columns=exclude_cols, errors="ignore")
y = df[target_col]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Normalize (optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Simple model — Random Forest (easy to interpret)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Quick prediction example
sample = X_test.iloc[0:1]
predicted_rasa = model.predict(scaler.transform(sample))[0]
print("\nExample Prediction:")
print(f"Predicted Rasa: {predicted_rasa}")

import joblib

joblib.dump(model, "model/rasa_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
