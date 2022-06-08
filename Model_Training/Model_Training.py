
from data_ingestion.data_loader import load_data
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

def create_model():

    loadData = load_data

    cluster0 , cluster1 = loadData.load_clustered_data()

    model_for_cluster_0 = RandomForestRegressor( criterion='squared_error', n_estimators= 400)
    model_for_cluster_1 = RandomForestRegressor(criterion= 'squared_error', n_estimators=200)

    def get_x_y(data):
        '''
        Function for getting X and y from the data
        '''
        return [data.drop("VISIBILITY", axis=1), data['VISIBILITY']]

    model_for_cluster_0.fit(get_x_y(cluster0[0]) , get_x_y(cluster0[1]) )
    model_for_cluster_1.fit(get_x_y(cluster1[0]) , get_x_y(cluster1[1]))

    joblib.dump(model_for_cluster_0 , "model_for_cluster_0.pickle")
    joblib.dump(model_for_cluster_1, "model_for_cluster_1.pickle")



class final_model:
    def __init__(self ,  Clustering_model):

        self.Clustering_model = Clustering_model
    def predict(self , input):
        model0 , model1 = joblib.load("Models//RandomForestModel_cluster0_.pickle") , joblib.load("Models//RandomForestModel_cluster1_.pickle")
        cluster = self.Clustering_model.predict(input)
        if cluster ==0 :
            output = model0.predict(input)
        else:
            output = model1.predict(input)
        return output

model_to_use = final_model(joblib.load("..//Models//Clustering_model.pickle"))
joblib.dump(model_to_use , "FINALMODEL.pickle")





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