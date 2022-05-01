import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score

from sklearn.model_selection import train_test_split


class FindBestModel:
    '''
    This clss is created to find th best model out of all
    '''

    def __init__(self ,X ,y, cluster_number):
        self.X = X
        self.y = y
        self.cluster_number = cluster_number

    def RandomForest(self, grid):
        '''
        Performs Randomforset Regression on given dataset after splitting it into train and test set.
        Checks for best parameters of the model according to the dataset
        returns
         1. The best parameters of the model
         2. Root Mean Squaed error of the model with best parameters on the test dataset
        '''
        print("Random Forest for ",self.cluster_number)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=1, test_size=0.3)
        RF_model = RandomForestRegressor()
        rf_gridSearch = GridSearchCV(RF_model, param_grid=grid , verbose=3)
        rf_gridSearch.fit(X_train, y_train)
        rf_prediction = rf_gridSearch.predict(X_test)
        RF_RMSE = np.sqrt(mean_squared_error(y_test, rf_prediction))
        RF_r2=  r2_score(y_test , rf_prediction)
        joblib.dump(rf_gridSearch, "RandomForestModel_"+self.cluster_number+"_.pickle")
        return rf_gridSearch.best_params_, rf_gridSearch.best_estimator_, RF_RMSE , RF_r2

    def XGBoost(self, grid):
        '''
        Performs XGBoost Regression on given dataset after splitting it into train and test set.
        Checks for best parameters of the model according to the dataset
        returns
         1. The best parameters of the model
         2. Root Mean Squaed error of the model with best parameters on the test dataset
        '''
        print("XGBOOST for ",self.cluster_number)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=1, test_size=0.3)
        XG_model = XGBRegressor()
        XG_gridSearch = GridSearchCV(XG_model, param_grid=grid , verbose=3)
        XG_gridSearch.fit(X_train, y_train)
        XG_prediction = XG_gridSearch.predict(X_test)
        XG_RMSE = np.sqrt(mean_squared_error(y_test, XG_prediction))
        XG_r2 = r2_score(y_test , XG_prediction)
        joblib.dump(XG_gridSearch, "XGBoostModel_"+self.cluster_number+"_.pickle")
        return XG_gridSearch.best_params_, XG_gridSearch.best_estimator_, XG_RMSE , XG_r2

    def best_model(self, rf_grid): # , xgboost_grid
        '''
        This functions finds the best model out of all models preformed
        and returms best parameters and Root Mean Squared Erro of the best Model out of all
        '''
        RF = self.RandomForest(rf_grid)
        # XG = self.XGBoost(xgboost_grid)
        print("\n\nCluster :" , self.cluster_number)
        print("Random Forest:", RF)
        print("\nRandom Forest Error" , RF[2])
        print("\n RF r2 score : ",RF[3])
        # print("\n\nXGBOOST:", XG)
        # print("\nXGBOOST Error", XG[2])
        # print("\n XGB r2 score : ",XG[3])
        return [RF ]


