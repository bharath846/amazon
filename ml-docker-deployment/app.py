
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "ML Model is running in Docker!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = data.get("features")

    if not isinstance(features, list) or len(features) != 3:
        return jsonify({
            "error": "Provide exactly 3 numerical features"
        }), 400

    try:
        prediction = model.predict([features])
        return jsonify({
            "prediction": str(prediction[0])
        })
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
