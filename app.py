import numpy as np
import flask
import flask_cors
import joblib
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from PredictionToDatabase.PredictionsToDB import createTable , enterTable
from Logging.Logger import logger
import warnings
warnings.filterwarnings("ignore")


app = flask.Flask(__name__)
logs = logger()

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
            logs.info("Left Home Page.")
            createTable()
            global LOCATION , output , input1, Precipitation, Dry_Bulb_Temp, Relative_Humidity, Wind_Speed, Sea_level_Pressure
            Precipitation = float(request.form["Precipitation"])
            Dry_Bulb_Temp = float(request.form["Dry_Bulb_Temp"])
            Relative_Humidity = float(request.form["Relative_Humidity"])
            Wind_Speed=  float(request.form["Wind_Speed"])
            Wind_Direction = float(request.form["Wind_Direction"])
            Sea_level_Pressure = float(request.form["Sea_level_Pressure"])

            logs.info("Successfully taken input values from user")

            input1= np.array([Dry_Bulb_Temp, Relative_Humidity, Wind_Speed,Wind_Direction, Sea_level_Pressure, Precipitation]).reshape(1,6)



        logs.info("Calling saved Serialized  model objects fro predictions")
        clustering_obj , scaler = get_serialized_objects()

        logs.info("Creating object of custom model made for the predictions.")
        model = final_model(clustering_obj)

        logs.info("Predicting Visibility from the given input")
        output = model.predict((scaler.transform(input1)))
        print(output)
        enterTable(Dry_Bulb_Temp, Relative_Humidity, Wind_Speed, Wind_Direction, Sea_level_Pressure, Precipitation,  output )
        logs.info("Saved the input in the MySQL Prediction Database !")
        print("A")
        logs.info("Rendering Prediction-page Output displayed !!")
        return render_template('prediction.html' , result = str(round(output[0],2)))
    
    except Exception as e:
        logs.info("Something went wrong :" + str(e))
        return render_template('Error.html' , result = str(e))   


def get_serialized_objects():
    path = 'Models//'
    scaler_obj = joblib.load(path+"scalar_object.pickle")
    clustering_object = joblib.load(path+"Clustering_model.pickle")
    return clustering_object  , scaler_obj


class final_model:
    def __init__(self ,  Clustering_model):
        logs.info("Entered final_model class.")
        self.Clustering_model = Clustering_model
    def predict(self , input):
        logs.info("Calling PCA object for dimensionality reduction.")
        pca = joblib.load("Models//pca_obj.pickle")
        reduced_input = pca.transform(input.reshape(-1,6))

        cluster = self.Clustering_model.predict(reduced_input)

        if cluster ==0 :
            model = joblib.load("Models//RandomForestModel_cluster0_.pickle")
            output = model.predict(input.reshape(-1,6))
        if cluster[0] == 1:
            
            # Here the model should be used as model selection is RandomForest for cluster1 but its size is more, so could not upload on github 
            
            model = joblib.load("Models//XGBoostModel_cluster1_.pickle")
            output = model.predict(input.reshape(-1,6))
        return output


if __name__ == "__main__":
    app.run()

'''
Cluster0 XGBoost :
XGBoost Best Parameters :
 {'gamma': 0.4, 'learning_rate': 0.1, 'min_child_weight': 10}
XGBoost Best Estimators :
 XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0.4, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.1, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=10,
             missing=nan, monotone_constraints='()', n_estimators=100, n_jobs=0,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, ...)
XGBoost Test RMSE :
 0.5893064625381013
XGBoost Test R2  :
 0.779091270788543
XGBoost Train RMSE :
 0.41653195799762327
XGBoost Train R2 :
 0.877509161900869


 Cluster0 Randomforest :

RandomForest Best Parameters :
 {'criterion': 'squared_error', 'n_estimators': 400}
RandomForest Best Estimators :
 RandomForestRegressor(n_estimators=400)
RandomForest Test RMSE :
 0.5861119722894502
RandomForest Test R2  :
 0.7814797668769428
RandomForest Train RMSE :
 0.2195708564718053
RandomForest Train R2 :
 0.9659626028283578



cluster1 XGBoost :
XGBoost Best Parameters :
 {'gamma': 0.1, 'learning_rate': 0.17, 'min_child_weight': 5}
XGBoost Best Estimators :
 XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0.1, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.17, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=5,
             missing=nan, monotone_constraints='()', n_estimators=100, n_jobs=0,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, ...)
XGBoost Test RMSE :
 1.7178308865011365
XGBoost Test R2  :
 0.5655753425170551
XGBoost Train RMSE :
 1.4775374822192073
XGBoost Train R2 :
 0.6786386117905491


 cluster1 Randomforest :

RandomForest Best Parameters :
 {'criterion': 'squared_error', 'n_estimators': 200}
RandomForest Best Estimators :
 RandomForestRegressor(n_estimators=200)
RandomForest Test RMSE :
 1.6732025016119063
RandomForest Test R2  :
 0.5878544038468825
RandomForest Train RMSE :
 0.6174675570538666
RandomForest Train R2 :
 0.9438764713323344


FullData XGBoost :/n
XGBoost Best Parameters :
 {'gamma': 0.4, 'learning_rate': 0.17, 'min_child_weight': 10}
XGBoost Best Estimators :
 XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0.4, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.17, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=10,
             missing=nan, monotone_constraints='()', n_estimators=100, n_jobs=0,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, ...)
XGBoost Test RMSE :
 1.3437126649646927
XGBoost Test R2  :
 0.6168362609976628
XGBoost Train RMSE :
 1.211424447350926
XGBoost Train R2 :
 0.7010534089585627




 FullData Randomforest :

RandomForest Best Parameters :
 {'criterion': 'squared_error', 'n_estimators': 400}
RandomForest Best Estimators :
 RandomForestRegressor(n_estimators=400)
RandomForest Test RMSE :
 1.312994297024733
RandomForest Test R2  :
 0.6341548845912297
RandomForest Train RMSE :
 0.4957536934439992
RandomForest Train R2 :
 0.9499351574372015

Process finished with exit code 0


'''
