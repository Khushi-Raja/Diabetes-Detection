from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessing objects
try:
    model = joblib.load('model/diabetes_model.pkl')
    scaler = joblib.load('model/diabetes_scaler.pkl')
    imputer = joblib.load('model/diabetes_imputer.pkl')
except Exception as e:
    print(f"Error loading model/scaler: {str(e)}")
    raise e

# Define columns that need imputation
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'Pregnancies': float(request.form['Pregnancies']),
            'Glucose': float(request.form['Glucose']),
            'BloodPressure': float(request.form['BloodPressure']),
            'SkinThickness': float(request.form['SkinThickness']),
            'Insulin': float(request.form['Insulin']),
            'BMI': float(request.form['BMI']),
            'DiabetesPedigreeFunction': float(request.form['DiabetesPedigreeFunction']),
            'Age': float(request.form['Age'])
        }

        # Convert to DataFrame
        sample_df = pd.DataFrame([data])

        # Preprocess the data
        sample_df[columns_to_impute] = imputer.transform(sample_df[columns_to_impute])
        scaled_samples = scaler.transform(sample_df)

        # Make prediction
        prediction = model.predict(scaled_samples)
        probabilities = model.predict_proba(scaled_samples)

        # Prepare result
        result = {
            'prediction': 'Diabetic' if prediction[0] == 1 else 'Non Diabetic',
            'confidence': round(np.max(probabilities) * 100, 2),
            'probabilities': {
                'Non Diabetic': round(probabilities[0][0], 4),
                'Diabetic': round(probabilities[0][1], 4)
            }
        }

        return render_template('result.html', result=result)

    except Exception as e:
        error_message = f"Error: {str(e)}. Please check your inputs."
        return render_template('result.html', 
                             result={'prediction': error_message, 'confidence': 0})

if __name__ == '__main__':
    app.run(debug=True)