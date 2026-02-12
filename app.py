from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model + scaler
MODEL_PATH = os.path.join("model", "rasa_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Simple rasa→raga mapping
RAGA_MAP = {
    "Shanta": "Revati",
    "Karuna": "Nilambari",
    "Veera": "Shankarabharanam",
    "Raudra": "Simhendramadhyamam",
    "Hasya": "Mohanam",
    "Bhayanaka": "Todi",
    "Bibhatsa": "Vakulabharanam",
    "Adbhuta": "Hamsadhwani",
    "Shringara": "Kalyani"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    profile_name = data.get("profile")

    # Load matching CSV (each profile file you generated earlier)
    csv_path = f"profiles/{profile_name.lower()}_profile.csv"
    df = pd.read_csv(csv_path)

    # Extract numeric features (exclude profile_name if present)
    features = df.select_dtypes("number").mean().to_frame().T
    X_scaled = scaler.transform(features)

    rasa_pred = model.predict(X_scaled)[0]
    raga = RAGA_MAP.get(rasa_pred, "Unknown")

    return jsonify({"rasa": rasa_pred, "raga": raga})

@app.route("/dataset_preview")
def dataset_preview():
    df = pd.read_csv("ragasense_train.csv").head(10)
    return jsonify({
        "columns": list(df.columns),
        "rows": df.values.tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)
