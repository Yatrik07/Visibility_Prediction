import joblib
'''
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

load = load_data()
data = load.get_data()


df = joblib.load("..//TrainingData//FinalData.pickle")
df = pd.DataFrame(df)
# print(df)



X = df
y = data['VISIBILITY']

RF_grid = joblib.load('RandomForestGrid.pickle')
# svr = joblib.load('SVRgrid.pickle')
XG_grid = joblib.load('XGBoost.pickle')

cvs_rf = cross_val_score(RF_grid , X , y ,cv = 5)
# cvs_svm = cross_val_score(SVR_grid , X , y ,cv = 5)
cvs_xgb = cross_val_score(XG_grid , X , y ,cv = 5)

X_train , X_test , y_train,  y_test = train_test_split(X , y , test_size=0.3)

print("RF cross_val_score",np.mean(cvs_rf))
# print("svm cross_val_score",np.mean(cvs_svm))
print("xgb cr`  `oss_val_score",np.mean(cvs_xgb))

print("R^2 rf:",(RF_grid.score(X_test , y_test)))
print("R^2 xg :",(XG_grid.score(X_test , y_test)))


print()
'''
# int('a')
int('')
