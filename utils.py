import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import joblib
import time
from flask import  make_response
import logging



logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# Load the trained autoencoders
autoencoder_imb60 = tf.keras.models.load_model('autoencoder_imb.h5')#make a folder for the weights in order to be loaded on a loop
autoencoder_imb5 = tf.keras.models.load_model('autoencoder_imb_5s.h5')
autoencoder_imb60sz = tf.keras.models.load_model('autoencoder_vol.h5')

logging.debug('loaded model')

data_local_imb60 = pd.read_csv("imb60.csv")#this too should be laoded on a loop
data_local_imb5 = pd.read_csv("imb5.csv")
data_local_imb60sz = pd.read_csv("imb60sz.csv")


def predict(data,
            autoencoder_imb60=autoencoder_imb60,
            autoencoder_imb5=autoencoder_imb5,
            autoencoder_imb60sz=autoencoder_imb60sz,
            data_local_imb60=data_local_imb60,
            data_local_imb5=data_local_imb5,
            data_local_imb60sz=data_local_imb60sz):

    # Obtener los datos de la petición
    imb60, imb60sz, imb5, ts = np.array([[data['imb60']]]), np.array([[data['imb60sz']]]),np.array([[data['imb5']]]), data['ts']
    logging.debug("imb60 %s", imb60)
    logging.debug("imb60sz %s", imb60sz)
    logging.debug("imb5 %s", imb5)
    logging.debug("ts %s", ts)
    print("ts: ", ts)
    # Load the scalers
    scaler60 = joblib.load('scaler_imb_5s.save')#this should be laoded 
    scaler60sz = joblib.load('scaler_imb.save')
    scaler5 = joblib.load('scaler_vol.save')

    # Leer los datos del archivo CSV##this also on a loop
    imbs60 = data_local_imb60["imb"]
    reconstructed_imbs60 = data_local_imb60["reconstructed_imb"]
    reconstruction_errors_imbs60 = data_local_imb60["reconstructed_error"]

    imbs5 = data_local_imb5["imb"]
    reconstructed_imbs5 = data_local_imb5["reconstructed_imb"]
    reconstruction_errors_imbs5 = data_local_imb5["reconstructed_error"]

    imbs60sz = data_local_imb60sz["imb"]
    reconstructed_imbs60sz = data_local_imb60sz["reconstructed_imb"]
    reconstruction_errors_imbs60sz = data_local_imb60sz["reconstructed_error"]
    
  

    #choosing side
    side60 = "buy" if imb60[0][0].astype(np.float64) > 0 else "sell" if imb60[0][0].astype(np.float64) <= 0 else 0
    side60sz = "buy" if imb60sz[0][0].astype(np.float64) > 0 else "sell" if imb60sz[0][0].astype(np.float64) <= 0 else 0
    side5 = "buy" if imb5[0][0].astype(np.float64) > 0 else "sell" if imb5[0][0].astype(np.float64) <= 0 else 0

    #scaling
    imb60 = scaler60.transform(imb60)
    imb60sz = scaler60sz.transform(imb60sz)
    imb5 = scaler5.transform(imb5)

    #concat new row to the dataframe
    imbs60 = pd.concat([imbs60, pd.Series([imb60[0][0]])], ignore_index=True)
    imbs60sz = pd.concat([imbs60sz, pd.Series([imb60sz[0][0]])], ignore_index=True)
    imbs5 = pd.concat([imbs5, pd.Series([imb5[0][0]])], ignore_index=True)

    # Make the prediction and anomaly detection 
    reconstructed_imb5 = autoencoder_imb5.predict(imb5)
    reconstructed_imbs5 = pd.concat([reconstructed_imbs5, pd.Series([reconstructed_imb5[0][0]])], ignore_index=True)
    reconstruction_error_imb5 = np.mean(np.square(reconstructed_imb5 - imb5))
    reconstruction_errors_imbs5 = pd.concat([reconstruction_errors_imbs5, pd.Series([reconstruction_error_imb5])], ignore_index=True)    
    threshold_imb5 = 50 * np.mean(reconstruction_errors_imbs5)
    is_anomaly5 = reconstruction_error_imb5 - threshold_imb5 > 0

    counts_5s = pd.read_csv("counts_5s.csv")
   
    if "Unnamed: 0" in counts_5s.columns:
        counts_5s=counts_5s.drop(["Unnamed: 0"], axis=1)#this should be rethinked

    if is_anomaly5:
        new_row={"ts":ts,"anomaly":1}
    else:
        new_row={"ts":ts,"anomaly":0}

    new_df_5s = pd.DataFrame(new_row, index=[0])
    counts_5s = pd.concat([counts_5s, new_df_5s], ignore_index=True)
    
    if (ts+5)%60==0:
        reconstructed_imb60 = autoencoder_imb60.predict(imb60)
        reconstructed_imbs60 = pd.concat([reconstructed_imbs60, pd.Series([reconstructed_imb60[0][0][0]])], ignore_index=True)
        reconstruction_error_imb60 = np.mean(np.square(reconstructed_imb60 - imb60))
        reconstruction_errors_imbs60 = pd.concat([reconstruction_errors_imbs60, pd.Series([reconstruction_error_imb60])], ignore_index=True)    
        threshold_imb60 = 30 * np.mean(reconstruction_errors_imbs60)
        is_anomaly = reconstruction_error_imb60 - threshold_imb60 > 0

        reconstructed_imb60sz = autoencoder_imb60sz.predict(imb60sz)
        reconstructed_imbs60sz = pd.concat([reconstructed_imbs60sz, pd.Series([reconstructed_imb60sz[0][0][0]])], ignore_index=True)
        reconstruction_error_imb60sz = np.mean(np.square(reconstructed_imb60sz - imb60sz))
        reconstruction_errors_imbs60sz = pd.concat([reconstruction_errors_imbs60sz, pd.Series([reconstruction_error_imb60sz])], ignore_index=True)
        
        threshold_imb60sz = 50 * np.mean(reconstruction_errors_imbs60sz)
        is_anomalysz = reconstruction_error_imb60sz - threshold_imb60sz > 0     
        if counts_5s.anomaly.sum()==0:
            is_anomaly5_1min=False
        else:
            is_anomaly5_1min=True
        counts_5s = pd.DataFrame({"ts":[],"anomaly":[]})
    else:
        is_anomaly=False
        is_anomalysz=False

    counts = pd.read_csv("counts.csv")
    


    if "Unnamed: 0" in counts.columns:
        counts=counts.drop(["Unnamed: 0"], axis=1)#this should be rethinked
#this is not printing the 5s dataframe
    # Generating new row for the count
    if is_anomaly and is_anomalysz and is_anomaly5_1min:
        if side60 == "buy" and side60sz == "buy" and side5 == "buy":
            new_row={"ts":ts,"buys":1,"sells":0}
        if side60 == "sell" and side60sz == "sell" and side5 == "sell":
            new_row={"ts":ts,"buys":0,"sells":1}
    else:
        new_row={"ts":ts,"buys":0,"sells":0}
    
    # Adding the new row to counts
    new_df = pd.DataFrame(new_row, index=[0])
    counts = pd.concat([counts, new_df], ignore_index=True)

    # counting anomalies
    buys=counts["buys"].sum()
    sells=counts["sells"].sum()

    # Configurations for the Graphql post when the 5 min candel close
    if (ts+5)%300==0:
        ts = ts-300+5   
        counts = pd.DataFrame({"ts":[],"buys":[],"sells":[]})
        time.sleep(10)
        sv="true"
    else:
        sv="false" 
       
    
    # Setings for the grapql payload
    counts.to_csv("counts.csv", index=False) 
    counts_5s.to_csv("counts_5s.csv", index=False)
    print("counts 5s ")
    print(counts_5s)
    print("counts ")
    print(counts)
    reqUrl = "https://pd.paguertrading.com/graphql"
    headersList = {
        "Accept": "*/*",
        "Content-Type": "application/json"
    }
    payload = '{"query": "mutation AddHFT($exchange: Exchange!, $symbol: Symbol!, $interval: Interval!, $secret: String!, $data: [HFTInput!]!) {   addHFT(exchange: $exchange, symbol: $symbol, interval: $interval, secret: $secret, data: $data) {     exchange     symbol     interval     hft {       ts       b       s       sv     }   } }", "variables": {"exchange":"BYBIT","symbol":"BTCUSDT","interval":"M5","secret":"EXaN2XyuM2WakhtmAqHqbITV","data":[{"ts":%s,"b":%s,"s":%s,"sv":%s}]}}' % (ts, buys, sells, sv)

    #sending the grapql post
    response = requests.request("POST", reqUrl, data=payload,  headers=headersList)
    print(response.text)
    logging.debug(response.text)

    #adding the new row to the error and deleting the first one
    new_data_60 = pd.DataFrame({"imb": imbs60, "reconstructed_imb": reconstructed_imbs60, "reconstructed_error": reconstruction_errors_imbs60})
    new_data_60 = new_data_60.iloc[1:]
    new_data_60.to_csv("imb60.csv", index=False)

    new_data_60sz = pd.DataFrame({"imb": imbs60sz, "reconstructed_imb": reconstructed_imbs60sz, "reconstructed_error":reconstruction_errors_imbs60sz})
    new_data_60sz = new_data_60sz.iloc[1:]
    new_data_60sz.to_csv("imb60sz.csv", index=False)

    new_data_5 = pd.DataFrame({"imb": imbs5, "reconstructed_imb": reconstructed_imbs5, "reconstructed_error": reconstruction_errors_imbs5})
    new_data_5 = new_data_5.iloc[1:]
    new_data_5.to_csv("imb5.csv", index=False)

    response = make_response("Success ")
    response.status = "200 OK"
    return response
