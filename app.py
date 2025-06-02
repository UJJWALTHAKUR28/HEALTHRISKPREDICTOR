from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import sys

app = Flask(__name__)

# Global variables
model_package = None
feature_columns = []
label_encoders = {}
feature_metadata = []

def load_model_with_fallback():
    """Load model with multiple fallback methods for deployment compatibility"""
    global model_package, feature_columns, label_encoders, feature_metadata
    
    MODEL_FILE = 'health_risk_model.pkl'
    
    # Print debug info
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    loading_methods = [
        ("Standard pickle", lambda: pickle.load(open(MODEL_FILE, 'rb'))),
        ("Pickle with protocol fix", lambda: pickle.load(open(MODEL_FILE, 'rb'), fix_imports=True)),
        ("Pickle with encoding", lambda: pickle.load(open(MODEL_FILE, 'rb'), encoding='latin1')),
    ]
    
    for method_name, load_func in loading_methods:
        try:
            print(f"Trying {method_name}...")
            model_package = load_func()
            print(f"✅ Success with {method_name}")
            
            # Extract components
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
            
            return True
            
        except Exception as e:
            print(f"❌ {method_name} failed: {str(e)}")
            continue
    
    print("🚨 All loading methods failed")
    return False

def predict_health_risk(age, gender, bmi, smoking_status, alcohol_intake_per_week, 
                        exercise_frequency_per_week, diet_quality, sleep_hours_per_night,
                        stress_level, family_history, blood_pressure, cholesterol_level,
                        glucose_level, heart_rate, mental_health_score, model_data):
    
    if model_data is None:
        raise Exception("Model not available")
    
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

    # Encode categorical variables
    for col in ['gender', 'smoking_status', 'diet_quality', 'stress_level', 'family_history']:
        try:
            input_data[col + '_encoded'] = model_data['label_encoders'][col].transform([input_data[col]])[0]
        except KeyError as e:
            raise Exception(f"Invalid value for {col}: {input_data[col]}")

    # Create feature vector
    feature_vector = np.array([input_data[col] for col in model_data['feature_columns']]).reshape(1, -1)
    feature_vector_scaled = model_data['scaler'].transform(feature_vector)

    # Make prediction
    prediction = model_data['model'].predict(feature_vector_scaled)[0]
    probabilities = model_data['model'].predict_proba(feature_vector_scaled)[0]
    predicted_class = model_data['target_encoder'].inverse_transform([prediction])[0]

    return predicted_class, probabilities, model_data['target_classes']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    if model_package is None:
        return render_template('index.html', features=[], error="Model loading failed. Predictions unavailable.")
    return render_template('index.html', features=feature_metadata)

@app.route('/predict', methods=['POST'])
def predict():
    if model_package is None:
        return render_template('result.html', prediction='Service Unavailable', 
                             probabilities=['Model failed to load on server'])
    
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
        error_msg = f"Prediction failed: {str(e)}"
        print(error_msg)  # Log to server
        return render_template('result.html', prediction='Error', probabilities=[error_msg])

@app.route('/health')
def health_check():
    """Health check endpoint for deployment platforms"""
    status = "healthy" if model_package is not None else "model_not_loaded"
    return {'status': status, 'numpy_version': np.__version__}

if __name__ == '__main__':
    print("🚀 Starting Health Risk Predictor...")
    
    # Try to load model
    model_loaded = load_model_with_fallback()
    
    if model_loaded:
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model failed to load - app will run with limited functionality")
    
    # Start server
    port = int(os.environ.get('PORT', 4000))
    print(f"🌐 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
