from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained KNN model and the scaler parameters (mean and scale)
with open("stroke_knn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_params.pkl", "rb") as f:
    scaler_params = pickle.load(f)

# Function to preprocess the input features and scale them
def preprocess_features(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status):
    # Convert categorical variables to numeric (as done during training)
    gender_numeric = 0 if gender == 'Male' else 1
    ever_married_numeric = 0 if ever_married == 'No' else 1
    work_type_numeric = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}[work_type]
    residence_type_numeric = 0 if residence_type == 'Urban' else 1
    smoking_status_numeric = {'never smoked': 0, 'smokes': 1, 'formerly smoked': 2}[smoking_status]
    
    # Create the feature array (10 numerical features)
    features = np.array([gender_numeric, age, hypertension, heart_disease, ever_married_numeric, 
                         work_type_numeric, residence_type_numeric, avg_glucose_level, bmi, smoking_status_numeric])
    
    # Manually scale the features using the saved mean and scale values
    mean = np.array(scaler_params['mean'])
    scale = np.array(scaler_params['scale'])

    # Ensure that you have the correct number of features (10 features)
    if len(features) != 10:
        raise ValueError(f"Expected 10 features, but got {len(features)} features.")
    
    # Apply scaling
    features_scaled = (features - mean) / scale  # Apply scaling to the features
    
    return features_scaled

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the features from the form
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        residence_type = request.form['residence_type']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status']

        # Preprocess and scale the features
        features_scaled = preprocess_features(gender, age, hypertension, heart_disease, ever_married, work_type,
                                              residence_type, avg_glucose_level, bmi, smoking_status)
        
        # Make prediction using the trained KNN model
        prediction = model.predict([features_scaled])  # Ensure it's 2D for prediction
        
        # Interpret prediction
        result = "Stroke Risk" if prediction[0] == 1 else "No Stroke Risk"
        
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
