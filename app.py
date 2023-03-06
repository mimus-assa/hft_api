from flask import Flask, request, make_response
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import requests
import time


app = Flask(__name__)

# Load the scaler
scaler = joblib.load('scaler.save')
print("loaded scaler")
# Load the trained autoencoder
autoencoder = tf.keras.models.load_model('autoencoder.h5')
print("loaded model")

@app.route('/predict', methods=['POST'])
def predict():

    # Leer los datos del archivo CSV
    data_local = pd.read_csv("imb.csv")
    imbs = data_local["imb"]
    reconstructed_imbs = data_local["reconstructed_imb"]
    reconstruction_errors = data_local["reconstructed_error"]
    
    # Obtener los datos de la petición
    data = request.get_json()
    imb = np.array([[data['imb']]])
    ts = data['timestamp']

    #round the ts for timing
    ts_round_30= (ts // 30) * 30

    #choosing side
    side = "buy" if imb[0][0].astype(np.float64) > 0 else "sell" if imb[0][0].astype(np.float64) <= 0 else 0
    imb = scaler.transform(imb)
    imbs = pd.concat([imbs, pd.Series([imb[0][0]])], ignore_index=True)

    # Make the prediction and anomaly detection
    reconstructed_imb = autoencoder.predict(imb)
    reconstructed_imbs = pd.concat([reconstructed_imbs, pd.Series([reconstructed_imb[0][0]])], ignore_index=True)
    reconstruction_error = np.mean(np.square(reconstructed_imb - imb))
    reconstruction_errors = pd.concat([reconstruction_errors, pd.Series([reconstruction_error])], ignore_index=True)    
    threshold = 2 * np.mean(reconstruction_errors)
    is_anomaly = reconstruction_error - threshold > 0
    counts = pd.read_csv("counts.csv")

    if "Unnamed: 0" in counts.columns:
        counts=counts.drop(["Unnamed: 0"], axis=1)#this should be rethinked
    # Generating new row for the count
    if is_anomaly:
        if side == "buy":
            new_row={"timestamp":ts,"buys":1,"sells":0}
        elif side=="sell":
            new_row={"timestamp":ts,"buys":0,"sells":1}
    else:
        new_row={"timestamp":ts,"buys":0,"sells":0}
    
    # Adding the new row to counts
    new_df = pd.DataFrame(new_row, index=[0])
    counts = pd.concat([counts, new_df], ignore_index=True)
    print("counts")
    print(counts)

    # counting anomalies
    buys=counts["buys"].sum()
    sells=counts["sells"].sum()

    # Configurations for the Graphql post when the 5 min candel close
    if (ts_round_30+60)%300==0:
        ts = ts-300+60   
        counts = pd.DataFrame({"timestamp":[],"buys":[],"sells":[]})
        time.sleep(10)
        sv="true"
    
    else:
        sv="false" 
       
    
    # Setings for the grapql payload
    counts.to_csv("counts.csv")   
    reqUrl = "https://pd.paguertrading.com/graphql"
    headersList = {
        "Accept": "*/*",
        "Content-Type": "application/json"
    }
    payload = '{"query": "mutation AddHFT($exchange: Exchange!, $symbol: Symbol!, $interval: Interval!, $secret: String!, $data: [HFTInput!]!) {   addHFT(exchange: $exchange, symbol: $symbol, interval: $interval, secret: $secret, data: $data) {     exchange     symbol     interval     hft {       ts       b       s       sv     }   } }", "variables": {"exchange":"BYBIT","symbol":"BTCUSDT","interval":"M5","secret":"EXaN2XyuM2WakhtmAqHqbITV","data":[{"ts":%s,"b":%s,"s":%s,"sv":%s}]}}' % (ts, buys, sells, sv)

    #sending the grapql post
    response = requests.request("POST", reqUrl, data=payload,  headers=headersList)
    print(response.text)
    
    #adding the new row to the error and deleting the first one
    new_data = pd.DataFrame({"imb": imbs, "reconstructed_imb": reconstructed_imbs, "reconstructed_error": reconstruction_errors})
    new_data = new_data.iloc[1:]
    new_data.to_csv("imb.csv", index=False)

    response = make_response("Success ")
    response.status = "200 OK"
    return response


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')