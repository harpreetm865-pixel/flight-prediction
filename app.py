
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained Linear Regression Pipeline
model = joblib.load(
    os.path.join(BASE_DIR, "model.pkl")
)

# Load original feature names
feature_columns = joblib.load(
    os.path.join(BASE_DIR, "feature_columns.pkl")
)


@app.route("/")
def home():

    return render_template(
        "index.html",
        prediction=None,
        error=None
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get input values from HTML
        airline = request.form["airline"]
        flight = request.form["flight"]

        source_city = request.form["source_city"]
        destination_city = request.form["destination_city"]

        departure_time = request.form["departure_time"]
        arrival_time = request.form["arrival_time"]

        stops = request.form["stops"]
        flight_class = request.form["class"]

        duration = float(request.form["duration"])
        days_left = int(request.form["days_left"])

        # Prepare input in the same format as training
        input_data = pd.DataFrame([{
            "airline": airline,
            "flight": flight,
            "source_city": source_city,
            "departure_time": departure_time,
            "stops": stops,
            "arrival_time": arrival_time,
            "destination_city": destination_city,
            "class": flight_class,
            "duration": duration,
            "days_left": days_left
        }])

        # Maintain exact original column order
        input_data = input_data[feature_columns]

        # Predict flight price
        prediction = model.predict(input_data)[0]

        prediction = round(float(prediction), 2)

        return render_template(
            "index.html",
            prediction=prediction,
            source=source_city,
            destination=destination_city,
            airline=airline,
            error=None
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=None,
            error=str(e)
        )


if __name__ == "__main__":

    app.run(debug=True)
