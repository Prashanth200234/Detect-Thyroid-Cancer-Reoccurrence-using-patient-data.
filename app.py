from flask import Flask, request, jsonify,render_template,redirect,url_for
import joblib  # To load the trained model
import numpy as np
from flask_cors import CORS  # Allow cross-origin requests

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load your trained model
model = joblib.load("model.pkl")  # Ensure the correct path

# Home route
@app.route("/")
def home():
    return redirect(url_for("index"))  # Redirect to index route
@app.route("/index")
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert inputs into the correct format for prediction
        features = [
            int(data["Age"]),
            1 if data["Gender"] == "Male" else 0,
            1 if data["Smoking"] == "Yes" else 0,
            1 if data["Hx_Smoking"] == "Yes" else 0,
            1 if data["Hx_Radiotherapy"] == "Yes" else 0,
            1 if data["Thyroid_Function"] == "Abnormal" else 0,
            1 if data["Physical_Examination"] == "Abnormal" else 0,
            1 if data["Adenopathy"] == "Yes" else 0,
            {"Papillary": 0, "Follicular": 1, "Medullary": 2, "Anaplastic": 3}[data["Pathology"]],
            1 if data["Focality"] == "Multifocal" else 0,
            {"Low": 0, "Intermediate": 1, "High": 2}[data["Risk"]],
            int(data["T"]),
            int(data["N"]),
            int(data["M"]),
            int(data["Stage"]),
            1 if data["Response"] == "Non-Responsive" else 0
        ]

        # Convert features into NumPy array and reshape for prediction
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Convert numeric output into meaningful text
        prediction_text = "High risk of recurrence" if prediction == 1 else "Low risk of recurrence"

        return jsonify({"prediction": prediction_text})  # Send descriptive response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
