from flask import Blueprint, jsonify
import pandas as pd

# Create a blueprint for v1/details
details_bp = Blueprint('details', __name__, url_prefix='/v1/details')

# Sample data
data = pd.read_csv('Last_version.csv')
df = pd.DataFrame(data)


@details_bp.route('/marques', methods=['GET'])
def get_marques():
    unique_marque = sorted(df["marque"].dropna().unique().tolist())
    return jsonify(unique_marque)

@details_bp.route('/gear', methods=['GET'])
def get_gear():
    unique_gear = sorted(df["Transmission"].dropna().unique().tolist())
    return jsonify(unique_gear)

@details_bp.route('/<marque>/models', methods=['GET'])
def get_modele(marque):
    unique_modele = sorted(df[df["marque"] == marque]["modele"].dropna().unique().tolist())
    return jsonify(unique_modele)