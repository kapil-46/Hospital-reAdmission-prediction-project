from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model
model = joblib.load('hospital_readmission_model.pkl')

# Pre-load LabelEncoders for categorical fields (fit on full data)
# For demo, fitting on example categories; replace with more robust approach in production
categorical_cols = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                    'glucose_test', 'A1Ctest', 'change', 'diabetes_med']

label_encoders = {}
example_data = {
    'age': ['[70-80]', '[50-60]', '[40-50]'],
    'medical_specialty': ['Missing', 'Other', 'InternalMedicine'],
    'diag_1': ['Circulatory', 'Respiratory', 'Digestive'],
    'diag_2': ['Other', 'Circulatory', 'Digestive'],
    'diag_3': ['Other', 'Diabetes', 'Respiratory'],
    'glucose_test': ['no', 'yes'],
    'A1Ctest': ['no', 'normal', 'high'],
    'change': ['yes', 'no'],
    'diabetes_med': ['yes', 'no']
}

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(example_data[col])
    label_encoders[col] = le

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Preprocess input data
    try:
        features = []

        # Numeric features
        numeric_features = ['time_in_hospital', 'n_lab_procedures', 'n_procedures',
                            'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']

        for f in numeric_features:
            features.append(float(data.get(f, 0)))

        # Categorical features encoded
        for col in categorical_cols:
            val = data.get(col, 'Missing')
            le = label_encoders.get(col)
            if le and val in le.classes_:
                features.append(le.transform([val])[0])
            else:
                # Handle unseen category
                features.append(-1)

        features = np.array(features).reshape(1, -1)

        pred = model.predict(features)[0]
        result = 'yes' if pred == 1 else 'no'

        return jsonify({'readmitted': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
