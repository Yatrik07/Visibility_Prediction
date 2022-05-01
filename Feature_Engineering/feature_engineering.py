from sklearn.preprocessing import StandardScaler
import pandas as pd
from data_ingestion.data_loader import  load_data
import joblib
load = load_data()
data = load.get_data()

scaler=StandardScaler()

data.drop(['DATE'], axis=1, inplace=True)

data.drop(['StationPressure'], axis=1, inplace=True)

data.drop(['DewPointTempF'], axis=1, inplace=True)
data.drop(['VISIBILITY'], axis=1, inplace=True)
scaler.fit(data)
scaled_data =scaler.transform(data)

print(scaled_data)

joblib.dump(scaled_data , 'FinalData.pickle')
# In Training Data

print(data.columns)
joblib.dump(scaler , "scaler_obj.pickle")