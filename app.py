import os
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model and Scaler
try:
    model = pickle.load(open('stroke_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    import os
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Capture 10 basic inputs from the form
    age = float(request.form['age'])
    gender = int(request.form['gender'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    married = int(request.form['married'])
    work = int(request.form['work'])
    residence = int(request.form['residence'])
    glucose = float(request.form['glucose'])
    bmi = float(request.form['bmi'])
    smoking = int(request.form['smoking'])

    # 2. Create the 3 "Improved Data" features (from Step 4 of our lab)
    log_glucose = np.log1p(glucose)
    clinical_risk_score = (age / 20) + hypertension + heart_disease
    metabolic_risk = (1 if glucose > 150 else 0) + (1 if bmi > 30 else 0) + (1 if age > 55 else 0)

    # 3. Combine into the final 13-feature array
    final_features = [
        gender, age, hypertension, heart_disease, married, 
        work, residence, glucose, bmi, smoking, 
        log_glucose, clinical_risk_score, metabolic_risk
    ]
    
    # 4. Scale and Predict
    final_features_arr = np.array([final_features])
    features_scaled = scaler.transform(final_features_arr)
    
    # Get probability
    prob = model.predict_proba(features_scaled)[:, 1][0]

    prob_percent = round(prob * 100, 2)
    
    # Use our Tuned Threshold (0.35)
    if prob >= 0.35:
        res_text = "⚠️ HIGH RISK PROFILE"
    else:
        res_text = "✅ LOW RISK PROFILE"

    return render_template('result.html', prob=prob_percent)

if __name__ == "__main__":
      port = int(os.environ.get('PORT', 5000))
      app.run(host='0.0.0.0', port=port, debug=False)