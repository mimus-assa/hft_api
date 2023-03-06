from flask import Flask, request, make_response
import pandas as pd
import joblib
from utils import *

# Load the scaler
scaler = joblib.load('scaler.save')
print("loaded scaler")

# Load the trained autoencoder
autoencoder = tf.keras.models.load_model('autoencoder.h5')
print("loaded model")

# Leer los datos del archivo CSV
data_local = pd.read_csv("imb.csv")
imbs = data_local["imb"]
reconstructed_imbs = data_local["reconstructed_imb"]
reconstruction_errors = data_local["reconstructed_error"]


# Flask Server
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def flask_predict():
    data = request.get_json()
    response = predict(data)
    return response

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')