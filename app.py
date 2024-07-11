import pickle
from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

CATEGORICAL_FEATURES = ['property_type', 'furnishing', 'power_backup', 'water_supply', 'crime_rate', 'dust_and_noise']


def load_model_and_encoders():
    model = pickle.load(open('model/regression_model.pkl', 'rb'))
    label_encoders = {}
    for feature in CATEGORICAL_FEATURES:
        label_encoders[feature] = LabelEncoder()
    return model, label_encoders


model, label_encoders = load_model_and_encoders()

def preprocess_input_data(request_form):
    try:
        property_area = float(request_form['property_area'])
        freq_powercuts = float(request_form['frequency_of_powercuts'])
        traffic_density = float(request_form['traffic_density_score'])
        air_quality_index = float(request_form['air_quality_index'])
        neighborhood_review = float(request_form['neighborhood_review'])

        features = []
        for feature in CATEGORICAL_FEATURES:
            label_encoder = label_encoders[feature]
            encoded_value = label_encoder.transform([request_form[feature]])[0]
            features.append(encoded_value)

        features.extend([property_area, freq_powercuts, traffic_density, air_quality_index, neighborhood_review])
        return np.array(features).reshape(1, -1)
    except ValueError:
        return None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features_array = preprocess_input_data(request.form)
    if features_array is None:
        return "Invalid input", 400
    habitability_score = model.predict(features_array)[0]
    return render_template('result.html', habitability_score=habitability_score)


if __name__ == '__main__':
    dummy_data = {'property_type': ['Apartment', 'Bungalow', 'Single-family home', 'Duplex', 'Container Home'],
                  'furnishing': ['Semi_Furnished', 'Unfurnished', 'Fully Furnished'],
                  'power_backup': ['No', 'Yes'],
                  'water_supply': ['Once in a day - Morning', 'Once in a day - Evening', 'All time',
                                   'Once in two days'],
                  'crime_rate': ['Slightly below average', 'Well below average', 'Well above average',
                                 'Slightly above average'],
                  'dust_and_noise': ['Medium', 'High', 'Low']}

    for feature, values in dummy_data.items():
        label_encoders[feature].fit(values)

    app.run(debug=True)