from flask import Flask, request, make_response
import pandas as pd

from utils import *

# Flask Server
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def flask_predict():
    data = request.get_json()
    response = predict(data)
    return response

if __name__ == '__main__':
    
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)


