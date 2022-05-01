from Clustering.reducing_features import clusters
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import    train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import joblib
from data_ingestion.data_loader import load_data
from sklearn.metrics import mean_absolute_error , mean_squared_error
from Clustered_Data.saving_Clustered_data import  saving_clusters_Assigning_labels_y



def get_final_data():
    load = load_data()
    # df1 , df2 , df3 , df4 = load.load_clustered_data()
    all_y = load.get_data()
    all_y = all_y[['VISIBILITY']]


    df1 , df2 , df3 , df4 , index1 , index2 , index3 , index4 = saving_clusters_Assigning_labels_y()



    y1 = all_y.iloc[index1 , :]
    y2 = all_y.iloc[index2 , :]
    y3 = all_y.iloc[index3 , :]
    y4 = all_y.iloc[index4 , :]




    # print(df1.shape , df2.shape , df3.shape , df4.shape)
    # print(y1.shape , y2.shape , y3.shape , y4.shape)

    df1 = pd.concat([df1 , y1] , axis=1)
    df2 = pd.concat([df2 , y2] , axis=1)
    df3 = pd.concat([df3 , y3] , axis=1)
    df4 = pd.concat([df4 , y4] , axis=1)

    df1.drop("cluster" , axis=1 , inplace=True)
    df2.drop("cluster" , axis=1 , inplace=True)
    df3.drop("cluster" , axis=1 , inplace=True)
    df4.drop("cluster" , axis=1 , inplace=True)

    # print(df1 , df2 , df3 , df4 , sep='\n\n\n')
    return df1 , df2 , df3 , df4


'''


RF_grid = GridSearchCV(RF_model , {'n_estimators':[80 , 100 , 125 , 150] , "criterion":['squared_error'] , "verbose":[3] } , verbose=3)
XG_grid  = GridSearchCV(XG_model , {'learning_rate':[0.01 , 0.05 , 0.1 ,0.17, 0.3   ],  'gamma':[0.1,0.2,0.4] , 'min_child_weight':[1,5,10] , "verbose":[3]} ,verbose=3)
'''