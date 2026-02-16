import joblib
import pandas as pd
import json
import os
from django.shortcuts import render
from django.conf import settings

# Paths
MODEL_PATH = os.path.join(settings.BASE_DIR, "predictor", "random_forest_model.pkl")
FEATURE_PATH = os.path.join(settings.BASE_DIR, "predictor", "feature_names.json")

# Load model
model = joblib.load(MODEL_PATH)

# Load exact feature order
with open(FEATURE_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)


def home(request):
    result = None
    risk = None  # ✅ define risk safely

    if request.method == "POST":
        try:
            # User inputs
            cycle = float(request.POST.get("cycle", 0))
            temperature = float(request.POST.get("temperature", 0))
            pressure = float(request.POST.get("pressure", 0))
            vibration = float(request.POST.get("vibration", 0))

            # Initialize ALL features with 0
            data = {feature: 0.0 for feature in FEATURE_NAMES}

            # Fill known values (mapping to model sensors)
            data["cycle"] = cycle
            data["sensor_2"] = temperature
            data["sensor_3"] = pressure
            data["sensor_4"] = vibration

            # Estimate RUL from cycle
            if "RUL" in data:
                data["RUL"] = max(0, 400 - cycle)

            # Create DataFrame with exact column order
            input_df = pd.DataFrame(
                [[data[f] for f in FEATURE_NAMES]],
                columns=FEATURE_NAMES
            )

            # Predict probability
            prob_failure = model.predict_proba(input_df)[0][1]

            # Convert to percentage
            risk = round(prob_failure * 100, 2)

            # Custom threshold
            THRESHOLD = 0.30

            if prob_failure >= THRESHOLD:
                result = f"⚠ FAILURE SOON (Risk: {risk}%)"
            else:
                result = f"✅ NORMAL (Risk: {risk}%)"

        except Exception as e:
            result = f"Error: {str(e)}"
            risk = None

    return render(request, "home.html", {
        "result": result,
        "risk": risk
    })
