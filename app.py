import numpy as np
import flask
import flask_cors
import joblib
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from PredictionToDatabase.PredictionsToDB import createTable, enterTable
from Logging.Logger import logger
import warnings

warnings.filterwarnings("ignore")

app = flask.Flask(__name__)
logs = logger()


@app.route('/', methods=['GET', "POST"])
@cross_origin()
def home_page():
    logs.info("Someone entered site, rendering Homepage !")
    logs.sd()
    return render_template("form.html")


@app.route('/prediction', methods=['GET', 'POST'])
@cross_origin()
def pred_page():
    try:
        if request.method == 'POST':
            logs.info("Left Home Page.")
            createTable()
            global LOCATION, output, input1, Precipitation, Dry_Bulb_Temp, Relative_Humidity, Wind_Speed, Sea_level_Pressure
            Precipitation = float(request.form["Precipitation"])
            Dry_Bulb_Temp = float(request.form["Dry_Bulb_Temp"])
            Relative_Humidity = float(request.form["Relative_Humidity"])
            Wind_Speed = float(request.form["Wind_Speed"])
            Wind_Direction = float(request.form["Wind_Direction"])
            Sea_level_Pressure = float(request.form["Sea_level_Pressure"])

            logs.info("Successfully taken input values from user")

            input1 = np.array([Dry_Bulb_Temp, Relative_Humidity, Wind_Speed, Wind_Direction, Sea_level_Pressure,
                               Precipitation]).reshape(1, 6)

        logs.info("Calling saved Serialized  model objects fro predictions")
        clustering_obj, scaler = get_serialized_objects()

        logs.info("Creating object of custom model made for the predictions.")
        model = final_model(clustering_obj)

        logs.info("Predicting Visibility from the given input")
        output = model.predict((scaler.transform(input1)))
        print(output)
        enterTable(Dry_Bulb_Temp, Relative_Humidity, Wind_Speed, Wind_Direction, Sea_level_Pressure, Precipitation,
                   output)
        logs.info("Saved the input in the MySQL Prediction Database !")
        print("A")
        logs.info("Rendering Prediction-page Output displayed !!")
        return render_template('prediction.html', result=str(round(output[0], 2)))

    except ValueError as e:
        logs.info("Something went wrong :" + str(e))
        return render_template('Error.html', result= "Invalid Input , \n Enter Numbers Only !")


    except Exception as e:
        logs.info("Something went wrong :" + str(e))
        return render_template('Error.html', result=str(e))


def get_serialized_objects():
    path = 'Models//'
    scaler_obj = joblib.load(path + "scalar_object.pickle")
    clustering_object = joblib.load(path + "Clustering_model.pickle")
    return clustering_object, scaler_obj


class final_model:
    def __init__(self, Clustering_model):
        logs.info("Entered final_model class.")
        self.Clustering_model = Clustering_model

    def predict(self, input):
        logs.info("Calling PCA object for dimensionality reduction.")
        pca = joblib.load("Models//pca_obj.pickle")
        reduced_input = pca.transform(input.reshape(-1, 6))

        cluster = self.Clustering_model.predict(reduced_input)

        if cluster == 0:
            model = joblib.load("Models//RandomForestModel_cluster0_.pickle")
            output = model.predict(input.reshape(-1, 6))
        if cluster[0] == 1:
            # Here the model should be used as model selection is RandomForest for cluster1 but its size is more, so could not upload on github

            model = joblib.load("Models//XGBoostModel_cluster1_.pickle")
            output = model.predict(input.reshape(-1, 6))
        return output


if __name__ == "__main__":
    app.run()
