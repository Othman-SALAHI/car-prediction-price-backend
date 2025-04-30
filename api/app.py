from flask import Flask
from api.v1 import details_bp  # Import the blueprint from api/v1
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for all routes and origins
app.register_blueprint(details_bp)
CORS(app, resources={r"/api/v1/*": {"origins": "*"}})
CORS(app, origins=["http://localhost:8080"])

# Register the blueprint

# Log incoming requests for debugging
# @app.before_request
# def log_request():
#     print(f"Request Headers: {request.headers}")
#     print(f"Request Body: {request.get_data()}")

if __name__ == "__main__":
    app.run(port=5001,debug=True)
