from data_ingestion.data_loader import load_data
from  best_model_finder import *
load = load_data()
data = load.get_data()

# Getting Data
full_df = load.data_for_mdoel_training()
cluster0 , cluster1 = load_data.load_clustered_data()


def get_x_y(data):
    '''
    Function for getting X and y from the data
    '''
    return [data.drop("VISIBILITY" , axis=1) , data['VISIBILITY'] ]


best_model_cluster0 = FindBestModel(get_x_y(cluster0)[0] , get_x_y(cluster0)[1] , 'cluster0')
best_model_cluster1 = FindBestModel(get_x_y(cluster1)[0] , get_x_y(cluster1)[1] , 'cluster1')
best_model_fullData = FindBestModel(get_x_y(full_df)[0] , get_x_y(full_df)[1] , 'fullData')

RF_grid =  {'n_estimators':[125 , 150,200, 300, 400] , "criterion":['squared_error']  }
XG_grid  = {'learning_rate':[0.01 , 0.05 , 0.1 ,0.17, 0.3   ],  'gamma':[0.1,0.2,0.4] , 'min_child_weight':[1,5,10]   }

full_output_cluster0_XG,  full_output_cluster0_RF = best_model_cluster0.best_model(XG_grid , RF_grid)
full_output_cluster1_XG, full_output_cluster1_RF = best_model_cluster1.best_model(XG_grid , RF_grid)
full_output_fullData_XG, full_output_fullData_RF = best_model_fullData.best_model(XG_grid , RF_grid)


def print_details(XG , RF , name):
    '''
    Function for printing all the necessary details of the model performance
    '''
    print(name ,"XGBoost :/n")
    print("XGBoost Best Parameters :\n",XG[0])
    print("XGBoost Best Estimators :\n", XG[1])
    print("XGBoost Test RMSE :\n", XG[2])
    print("XGBoost Test R2  :\n", XG[3])
    print("XGBoost Train RMSE :\n", XG[4])
    print("XGBoost Train R2 :\n", XG[5])

    # return rf_gridSearch.best_params_, rf_gridSearch.best_estimator_, RF_RMSE, RF_r2, RF_train_RMSE, RF_train_r2
    print("\n\n\n",name ,"Randomforest :\n")
    print("RandomForest Best Parameters :\n",RF[0])
    print("RandomForest Best Estimators :\n", RF[1])
    print("RandomForest Test RMSE :\n", RF[2])
    print("RandomForest Test R2  :\n", RF[3])
    print("RandomForest Train RMSE :\n", RF[4])
    print("RandomForest Train R2 :\n", RF[5])
    # return XG_gridSearch.best_params_, XG_gridSearch.best_estimator_, XG_RMSE, XG_r2, XG_train_RMSE, XG_train_r2


print_details(full_output_cluster0_XG,  full_output_cluster0_RF , "Cluster0")
print_details(full_output_cluster1_XG,  full_output_cluster1_RF , "cluster1")
print_details(full_output_fullData_XG, full_output_fullData_RF , "FullData")