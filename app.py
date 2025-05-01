from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from api.v1.details import details_bp

# --------------------
# Model & Data Loading
# --------------------
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("final_version.csv", low_memory=False)

categorical_features = ["Transmission", "Carburant", "marque", "modele", "premierMain"]
numerical_features = ["Kilométrage", "Année", "CV"]

# Label encode and scale
df_encoded = df.copy()
df_encoded[categorical_features] = df_encoded[categorical_features].astype(str)

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# -----------------
# Prediction Method
# -----------------
def predict_price(input_data, model, scaler, label_encoders, numerical_features, categorical_features):
    input_df = pd.DataFrame([input_data])
    for col in categorical_features:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col].astype(str))

    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    X_input = input_df[categorical_features + numerical_features]

    predicted_price = model.predict(X_input)[0]
    return predicted_price

# ----------------------
# Flask App Initialization
# ----------------------
app = Flask(__name__)
app.register_blueprint(details_bp, url_prefix="/v1/details")
CORS(app, resources={r"/api/v1/*": {"origins": "*"}})
CORS(app, origins=["http://localhost:8080"])
CORS(app, origins=["https://car-prediction-price-maroc.vercel.app"])

# ----------------------
# UI Form Route
# ----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    price = None
    form_data = {}

    if request.method == "POST":
        try:
            form_data = {
                "Transmission": request.form["Transmission"],
                "Carburant": request.form["Carburant"],
                "marque": request.form["marque"],
                "modele": request.form["modele"],
                "premierMain": request.form["premierMain"],
                "Kilométrage": request.form["Kilométrage"],
                "Année": request.form["Année"],
                "CV": request.form["CV"]
            }

            input_data = {
                "Transmission": form_data["Transmission"],
                "Carburant": form_data["Carburant"],
                "marque": form_data["marque"],
                "modele": form_data["modele"],
                "premierMain": int(form_data["premierMain"]),
                "Kilométrage": float(form_data["Kilométrage"]),
                "Année": int(form_data["Année"]),
                "CV": int(form_data["CV"])
            }

            price = predict_price(input_data, model, scaler, label_encoders, numerical_features, categorical_features)

        except Exception as e:
            print("Prediction error:", e)

    return render_template("index.html",
                           marques=sorted(df["marque"].dropna().unique()),
                           carburants=df["Carburant"].dropna().unique(),
                           transmissions=df["Transmission"].dropna().unique(),
                           price=price,
                           form_data=form_data)

# -----------------------
# API Endpoint: Predict
# -----------------------
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return '', 200

    try:
        form_data = request.get_json()
        print("Received data:", form_data)

        input_data = {
            "Transmission": form_data["selectedGear"],
            "Carburant": form_data["selectedFuel"],
            "marque": form_data["marques"],
            "modele": form_data["models"],
            "premierMain": int(form_data["premierMain"]),
            "Kilométrage": float(form_data["kilometrage"]),
            "Année": int(form_data["annee"]),
            "CV": int(form_data["cv"])
        }

        price = predict_price(input_data, model, scaler, label_encoders, numerical_features, categorical_features)
        return jsonify({'price': round(float(price), 2)})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'Prediction failed'}), 500

def get_models_by_marque(marque):
    return df[df["marque"] == marque]["modele"].dropna().unique().tolist()

@app.route("/get_models/<marque>", methods=["GET"])
def get_models(marque):
    models = get_models_by_marque(marque)
    return jsonify(models)


# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)