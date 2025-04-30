import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load the model, encoder, and scaler
with open(r"C:\Users\Othman SALAHI\Desktop\PFM\xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)


df = pd.read_csv("Last_version.csv", low_memory=False)
df.shape


# Define features
categorical_features = ["Transmission", "Carburant", "marque", "modele", "premierMain"]
numerical_features = ["Kilométrage", "Année", "CV"]

# Make a copy of the original DataFrame (optional, for safety)
df_encoded = df.copy()

# Ensure categorical columns are strings
df_encoded[categorical_features] = df_encoded[categorical_features].astype(str)

# Apply Label Encoding to each categorical column
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le  # Optional: Save encoder for inverse_transform or inference

# Standardize numerical features
scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# Final feature matrix and target
X = df_encoded[categorical_features + numerical_features]
y = df_encoded["prix"]





# Define the categorical and numerical features
categorical_features = ["Transmission", "Carburant", "marque", "modele", "premierMain"]
numerical_features = ["Kilométrage", "Année", "CV"]

def predict_price(input_data, model, scaler, label_encoders, numerical_features, categorical_features):

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features using saved LabelEncoders
    for col in categorical_features:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col].astype(str))

    # Scale numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # Ensure column order matches training data
    X_input = input_df[categorical_features + numerical_features]

    # Predict
    predicted_price = model.predict(X_input)[0]
    return predicted_price # Return both log and original price



def get_models_by_marque(marque):
    return df[df["marque"] == marque]["modele"].dropna().unique().tolist()

# def get_marque_by_carburant(marque):
#     return df[df["Carburant"] == marque]["marque"].dropna().unique().tolist()

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

# Flask App
app = Flask(__name__)
CORS(app)
CORS(app, origins=["http://localhost:8080"])

# Get unique values for select fields
unique_transmission = df["Transmission"].dropna().unique()
unique_carburant = df["Carburant"].dropna().unique()
unique_marque = sorted(df["marque"].dropna().unique())
unique_modele = sorted(df["modele"].dropna().unique())

@app.route("/", methods=["GET", "POST"])
def index():
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
                           marques=unique_marque, 
                           carburants=unique_carburant, 
                           transmissions=unique_transmission, 
                           price=price if 'price' in locals() else None, 
                           form_data=form_data if 'form_data' in locals() else None)


@app.route("/get_models/<marque>", methods=["GET"])
def get_models(marque):
    models = get_models_by_marque(marque)
    return jsonify(models)

# @app.route("/get_marques/<Carburant>", methods=["GET"])
# def get_Carburant(Carburant):
#     marques = get_marque_by_carburant(Carburant)
#     return jsonify(marques)

from flask import Flask, request, jsonify
import numpy as np  # just in case your model returns numpy types

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return '', 200  # CORS preflight

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

        # your prediction function likely returns np.float32 or similar
        price = predict_price(input_data, model, scaler, label_encoders, numerical_features, categorical_features)

        # Cast it to float (native Python) before returning
        return jsonify({'price': round(float(price),2)})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': 'Prediction failed'}), 500




if __name__ == "__main__":
    app.run(debug=True)