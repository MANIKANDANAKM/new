from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import google.generativeai as genai

app = Flask(__name__)

# Load your pre-trained model
model = pickle.load(open("model.pkl", "rb"))

# Configure Google Generative AI
genai.configure(api_key="AIzaSyBDbBi8Z2LBbcWQt3L3bfMhAapHtJfaGjQ")

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]

modelai = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

@app.route("/")
def index():
    return render_template("index1.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Initialize precautions to avoid undefined variable error
        precautions = ""

        # Collect all 22 features from the form
        features = [
            float(request.form["mdvp_fo"]),
            float(request.form["mdvp_fhi"]),
            float(request.form["mdvp_flo"]),
            float(request.form["mdvp_jitter"]),
            float(request.form["mdvp_jitter_abs"]),
            float(request.form["mdvp_rap"]),
            float(request.form["mdvp_ppq"]),
            float(request.form["jitter_ddp"]),
            float(request.form["mdvp_shimmer"]),
            float(request.form["mdvp_shimmer_db"]),
            float(request.form["shimmer_apq3"]),
            float(request.form["shimmer_apq5"]),
            float(request.form["mdvp_apq"]),
            float(request.form["shimmer_dda"]),
            float(request.form["nhr"]),
            float(request.form["hnr"]),
            float(request.form["rpde"]),
            float(request.form["dfa"]),
            float(request.form["spread1"]),
            float(request.form["spread2"]),
            float(request.form["d2"]),
            float(request.form["ppe"]),
        ]

        # Prepare input for prediction
        features = [features]
        prediction = model.predict(features)

        if prediction[0] == 1:  # If the model predicts Parkinson's Disease
             result = "The Person does not have Parkinson's Disease"

    # Use Google Generative AI to get precautions for Parkinson's Disease
             data = "Provide general health tips for a person who does not have Parkinson's Disease."
             
        else:
             result = "The Person has Parkinson's Disease"

    # Use Google Generative AI to provide health tips for a person without Parkinson's
             data = "Provide precautions for Parkinson's Disease."

# Common logic to call Google Generative AI for content generation
             prompt_parts = [data]
             response = modelai.generate_content(prompt_parts)

# Fetch and display generated response
             precautions = response.text if response else "No recommendations available."


        # Render the result template with prediction outcome and precautions
        return render_template("result.html", result=result, precautions=precautions)

    except Exception as e:
        # Return error details for debugging
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
