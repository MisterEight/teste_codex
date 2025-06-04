from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

MODEL_PATH = 'model.pkl'
USER_DATA_PATH = 'user_data.csv'

app = Flask(__name__)

# Load model at startup
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle['model']
sport_enc = model_bundle['sport_enc']
eye_enc = model_bundle['eye_enc']

# Ensure user data file exists
if not os.path.exists(USER_DATA_PATH):
    pd.DataFrame(columns=['sport', 'predicted_eye_color']).to_csv(USER_DATA_PATH, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sport = request.form.get('sport')
    if not sport:
        return jsonify({'error': 'sport is required'}), 400

    if sport not in sport_enc.classes_:
        return jsonify({'error': 'sport not found'}), 400

    sport_num = sport_enc.transform([sport])[0]
    proba = model.predict_proba([[sport_num]])[0]
    pred_num = proba.argmax()
    eye_color = eye_enc.inverse_transform([pred_num])[0]
    probability = float(proba[pred_num])

    df = pd.DataFrame([[sport, eye_color, probability]], columns=['sport', 'predicted_eye_color', 'probability'])
    df.to_csv(USER_DATA_PATH, mode='a', header=False, index=False)

    return jsonify({'eye_color': eye_color, 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)
