import pandas as pd
import numpy as np
import sklearn
import flask
import flask_cors
import joblib
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from data_ingestion.data_loader import load_data
# from model_building.reducing_features import  clusters
# from xgboost import XGBRegressor
# from MapPrediction.get_map import Map
# from PredictionsToDB.PredictionsToDB import createTable , enterTable
# import matplotlib.pyplot as plt
from Logging.Logger import logger


app = flask.Flask(__name__)
logs = logger()

# createTable()

@app.route('/' , methods =  ['GET' , "POST"])
@cross_origin()
def home_page():
    logs.info("Someone entered site, rendering Homepage !")
    logs.sd()
    return render_template("form.html")

@app.route('/prediction' , methods =  ['GET' , 'POST'])
@cross_origin()
def pred_page():
    try:
        if request.method=='POST':
            global LOCATION , output , input1
            Precipitation = float(request.form["Precipitation"])
            Dry_Bulb_Temp = float(request.form["Dry_Bulb_Temp"])
            Wet_Bulb_Temp=  float(request.form["Wet_Bulb_Temp"])
            Relative_Humidity = float(request.form["Relative_Humidity"])
            Wind_Speed=  float(request.form["Wind_Speed"])
            Wind_Direction = float(request.form["Wind_Direction"])
            Sea_level_Pressure = float(request.form["Sea_level_Pressure"])
            LOCATION = str(request.form["location"])

            input1= np.array([Dry_Bulb_Temp, Wet_Bulb_Temp, Relative_Humidity, Wind_Speed,Wind_Direction, Sea_level_Pressure, Precipitation]).reshape(1,7)

            # enterTable(Dry_Bulb_Temp, Wet_Bulb_Temp, Relative_Humidity, Wind_Speed,Wind_Direction, Sea_level_Pressure, Precipitation)
    except :
        return render_template('Error.html' , result = "Wrong Input, Enter integers only!")


    model , scaler = get_serialized_objects()

    output = model.predict(np.array(scaler.transform(input1)).reshape(1,7))
    # save_map(LOCATION , output)

    return render_template('prediction.html' , result = str(round(output[0],2)))

'''
#@app.route('/map' , methods = ["GET" , "POST"])
def save_map(LOCATION , radius):

        final_map = Map()
        map = final_map.map_from_location_visibility(LOCATION, radius)
        print('started')
        # map.tolist()
        #map.save(outfile ='Map10.html')



@app.route('/map', methods = ["GET" , "POST"])
def t():
    final_map = Map()
    map = final_map.map_from_location_visibility(LOCATION, output)
    return jsonify(map)
'''
def get_serialized_objects():
    path = 'Final_Model//'
    randomforest_obj = joblib.load(path+"RandomForest.pickle")
    scaler_obj = joblib.load(path+"scaler_obj.pickle")
    return randomforest_obj , scaler_obj

if __name__ == "__main__":
    app.run()


'''
    cluster1_data = data.query("cluster==0")
    cluster2_data = data.query("cluster==1")
    cluster3_data = data.query("cluster==2")
    cluster4_data = data.query("cluster==3")
'''