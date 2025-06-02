from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model once globally
MODEL_FILE = 'health_risk_model.pkl'
with open(MODEL_FILE, 'rb') as f:
    model_package = pickle.load(f)

feature_columns = model_package['feature_columns']
label_encoders = model_package['label_encoders']

# Generate feature metadata
feature_metadata = []
for feature in feature_columns:
    if feature in label_encoders:
        options = list(label_encoders[feature].classes_)
        feature_metadata.append({'name': feature, 'type': 'categorical', 'options': options})
    else:
        feature_metadata.append({'name': feature, 'type': 'numerical', 'min': 0, 'max': 100})

def predict_health_risk(age, gender, bmi, smoking_status, alcohol_intake_per_week, 
                        exercise_frequency_per_week, diet_quality, sleep_hours_per_night,
                        stress_level, family_history, blood_pressure, cholesterol_level,
                        glucose_level, heart_rate, mental_health_score, model_data):
    input_data = {
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'smoking_status': smoking_status,
        'alcohol_intake_per_week': alcohol_intake_per_week,
        'exercise_frequency_per_week': exercise_frequency_per_week,
        'diet_quality': diet_quality,
        'sleep_hours_per_night': sleep_hours_per_night,
        'stress_level': stress_level,
        'family_history': family_history,
        'blood_pressure': blood_pressure,
        'cholesterol_level': cholesterol_level,
        'glucose_level': glucose_level,
        'heart_rate': heart_rate,
        'mental_health_score': mental_health_score
    }

    for col in ['gender', 'smoking_status', 'diet_quality', 'stress_level', 'family_history']:
        input_data[col + '_encoded'] = model_data['label_encoders'][col].transform([input_data[col]])[0]

    feature_vector = np.array([input_data[col] for col in model_data['feature_columns']]).reshape(1, -1)
    feature_vector_scaled = model_data['scaler'].transform(feature_vector)

    prediction = model_data['model'].predict(feature_vector_scaled)[0]
    probabilities = model_data['model'].predict_proba(feature_vector_scaled)[0]
    predicted_class = model_data['target_encoder'].inverse_transform([prediction])[0]

    return predicted_class, probabilities, model_data['target_classes']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html', features=feature_metadata)

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    try:
        predicted_class, probabilities, target_classes = predict_health_risk(
            age=int(form['age']),
            gender=form['gender'],
            bmi=float(form['bmi']),
            smoking_status=form['smoking_status'],
            alcohol_intake_per_week=int(form['alcohol_intake_per_week']),
            exercise_frequency_per_week=int(form['exercise_frequency_per_week']),
            diet_quality=form['diet_quality'],
            sleep_hours_per_night=float(form['sleep_hours_per_night']),
            stress_level=form['stress_level'],
            family_history=form['family_history'],
            blood_pressure=int(form['blood_pressure']),
            cholesterol_level=int(form['cholesterol_level']),
            glucose_level=int(form['glucose_level']),
            heart_rate=int(form['heart_rate']),
            mental_health_score=float(form['mental_health_score']),
            model_data=model_package
        )
        return render_template('result.html', prediction=predicted_class, probabilities=probabilities)
    except Exception as e:
        return render_template('result.html', prediction='Error', probabilities=[str(e)])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
