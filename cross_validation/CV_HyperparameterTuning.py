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


RF_model = RandomForestRegressor()
XG_model = XGBRegressor()
SVR_model = SVR()

RF_grid = GridSearchCV(RF_model , {'n_estimators':[80 , 100 , 125 , 150] , "criterion":['squared_error'] , "verbose":[3] } , verbose=3)
XG_grid  = GridSearchCV(XG_model , {'learning_rate':[0.01 , 0.05 , 0.1 ,0.17, 0.3   ],  'gamma':[0.1,0.2,0.4] , 'min_child_weight':[1,5,10] , "verbose":[3]} ,verbose=3)

X = df
y = data['VISIBILITY']
X_train , X_test , y_train,  y_test = train_test_split(X , y , test_size=0.3)
# RF_model = RandomForestRegressor()
# RF_grid = GridSearchCV(RF_model , {'n_estimators':[80 , 100 , 125 , 150] , "criterion":['squared_error'], 'verbose':[3] } , verbose=3)
# RF_grid.fit(X_train , y_train)


RF_grid.fit(X_train , y_train)
XG_grid.fit(X_train , y_train)


rf_pred = RF_grid.predict(X_test)
xg_pred = XG_grid.predict(X_test)



# RF_model.fit(X_train , y_train)
# rf_pred=RF_model.predict(X_test)


print('random forest accuracy RMSE :',np.sqrt(mean_squared_error(y_test , rf_pred)) )
print('xgboost accuracy rmse :',np.sqrt(mean_squared_error(y_test , xg_pred)))


print("R^2 rf:",(RF_grid.score(X_test , y_test)))
print("R^2 xg :",(XG_grid.score(X_test , y_test)))

joblib.dump(RF_grid , "RandomForestGrid.pickle")
joblib.dump(XG_grid , "XGBoost.pickle")
joblib.dump(SVR_grid , "SVRgrid.pickle")


cvs_rf = cross_val_score(RF_grid , X , y ,cv = 6)
cvs_svm = cross_val_score(SVR_grid , X , y ,cv = 6)
cvs_xgb = cross_val_score(XG_grid , X , y ,cv = 6)

print("RF cross_val_score",np.mean(cvs_rf))
print("svm cross_val_score",np.mean(cvs_svm))
print("xgb cr`  `oss_val_score",np.mean(cvs_xgb))