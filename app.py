from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained artifacts
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from the HTML form
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Create feature array
        features = np.array([
            [sepal_length, sepal_width, petal_length, petal_width]
        ])

        # Apply the same scaler used during training
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Decode numeric prediction back to species name
        species = encoder.inverse_transform(prediction)[0]

        return render_template(
            "index.html",
            prediction=species
        )

    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numeric values."
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )


# Optional REST API endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]])

        # Scale features
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)

        # Decode prediction
        species = encoder.inverse_transform(prediction)[0]

        return jsonify({
            "success": True,
            "species": species
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
