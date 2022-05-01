import joblib
from data_ingestion.data_loader import load_data
from sklearn.metrics import mean_absolute_error , mean_squared_error
from Clustered_Data.saving_Clustered_data import  saving_clusters_Assigning_labels_y
from Integrating_Final_Training_Data import get_final_data
from sklearn.model_selection import cross_val_score
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from data_ingestion.data_loader import load_data


model1 = joblib.load('RandomForestModel_cls1_.pickle')
model2 = joblib.load('RandomForestModel_cls2_.pickle')
model3 = joblib.load('RandomForestModel_cls3_.pickle')
model4 = joblib.load('RandomForestModel_cls4_.pickle')
fullMODEL = joblib.load("RandomForest.pickle")

df1 , df2 , df3 , df4 = get_final_data()

def eval(model_object ,X , y):
    X_train,X_test,y_train,y_test = train_test_split(X , y , random_state=5 , test_size=0.3)
    model_object.fit(X_train,  y_train)
    predictions = model_object.predict(X_test)
    model_socre = r2_score(y_test , predictions)
    return model_socre


score1 = eval(model1 , df1.drop("VISIBILITY",axis=1) , df1['VISIBILITY'])
score2 = eval(model2 , df2.drop("VISIBILITY",axis=1) , df2['VISIBILITY'])
score3 = eval(model3 , df3.drop("VISIBILITY",axis=1) , df3['VISIBILITY'])
score4 = eval(model4 , df4.drop("VISIBILITY",axis=1) , df4['VISIBILITY'])

print("score1:",score1 , "\nscore2:",score2 ,"\nscore3:", score3 ,"\nscore4:", score4)


load = load_data()
fullDF = load.get_data()
fullDF.drop("DATE" , inplace=True , axis=1)
fullDF.drop(['WETBULBTEMPF','DewPointTempF','StationPressure'] , inplace=True , axis=1)
fullDF_score = eval(fullMODEL , fullDF.drop("VISIBILITY",axis=1) , fullDF['VISIBILITY'])
print("\nfullDF_score:",fullDF_score)


print("model1 :/n",model1.best_params_)
print("model2 :/n",model2.best_params_)
print("model3 :/n",model3.best_params_)
print("model4 :/n",model4.best_params_)



