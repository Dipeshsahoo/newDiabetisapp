from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model (and scaler if used)
model = joblib.load('diabetes_model.pkl')
# scaler = joblib.load('scaler.pkl')  # uncomment if used

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    features = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['bloodpressure']),
        float(request.form['skinthickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    ]

    final_features = np.array([features])

    # If you used scaling
    # final_features = scaler.transform(final_features)

    prediction = model.predict(final_features)

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    return render_template('index.html', prediction_text=result)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



