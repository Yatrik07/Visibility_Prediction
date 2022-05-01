from Integrating_Final_Training_Data import get_final_data
from cross_validation.best_model_finder import FindBestModel
from data_ingestion.data_loader import load_data

df1 , df2 , df3 , df4 = get_final_data()

# print(df1,  df2 , df3 , df4)

def finding_best_parameters_for_all_clusters():
    FINDING_MODEL_CLS1 = FindBestModel(df1.drop("VISIBILITY" , axis=1) , df1['VISIBILITY'] , "cls1")

    rf_grid =  {'n_estimators':[150,200,250,350 , 500 , 700] , 'n_jobs':[-1] }
    # xg_grid = {'learning_rate':[0.01 , 0.05 , 0.1 , 0.3 , 0.5 ,0.8  ],  'gamma':[0.1,0.15,0.2,0.4] , 'min_child_weight':[1,5,10] }
    FINDING_MODEL_CLS1.best_model(rf_grid ) #, xg_grid


    FINDING_MODEL_CLS2 = FindBestModel(df2.drop("VISIBILITY" , axis=1) , df2['VISIBILITY'] , "cls2")

    rf_grid =  {'n_estimators':[150,200,250,350 , 500 , 700] , 'n_jobs':[-1] }
    # xg_grid = {'learning_rate':[0.01 , 0.05 , 0.1 , 0.3 , 0.5 ,0.8  ],  'gamma':[0.1,0.15,0.2,0.4] , 'min_child_weight':[1,5,10] }
    FINDING_MODEL_CLS2.best_model(rf_grid ) # , xg_grid


    FINDING_MODEL_CLS3 = FindBestModel(df3.drop("VISIBILITY" , axis=1) , df3['VISIBILITY'] , "cls3")

    rf_grid =  {'n_estimators':[150,200,250,350 , 500 , 700] , 'n_jobs':[-1] }
    # xg_grid = {'learning_rate':[0.01 , 0.05 , 0.1 , 0.3 , 0.5 ,0.8  ],  'gamma':[0.1,0.15,0.2,0.4] , 'min_child_weight':[1,5,10] }
    FINDING_MODEL_CLS3.best_model(rf_grid ) # , xg_grid


    FINDING_MODEL_CLS4 = FindBestModel(df4.drop("VISIBILITY" , axis=1) , df4['VISIBILITY'] , "cls4")

    rf_grid =  {'n_estimators':[150,200,250,350 , 500 , 700] , 'n_jobs':[-1] }
    # xg_grid = {'learning_rate':[0.01 , 0.05 , 0.1 , 0.3 , 0.5 ,0.8  ],  'gamma':[0.1,0.15,0.2,0.4] , 'min_child_weight':[1,5,10] }
    FINDING_MODEL_CLS4.best_model(rf_grid ) # , xg_grid

    load= load_data()
    full_df = load.data_for_mdoel_training()
    FULL_RANDOMFOREST = FindBestModel(full_df.drop("VISIBILITY", axis=1), full_df['VISIBILITY'], "FULL RANDOMFOREST")
    rf_grid =  {'n_estimators':[150,200,250,350 , 500 , 700] , 'n_jobs':[-1] }
    # xg_grid = {'learning_rate':[0.01 , 0.05 , 0.1 , 0.3 , 0.5 ,0.8  ],  'gamma':[0.1,0.15,0.2,0.4] , 'min_child_weight':[1,5,10] }
    FULL_RANDOMFOREST.best_model(rf_grid)  # , xg_grid




finding_best_parameters_for_all_clusters()
# cl1 - rf --> 0.76
# cls2 - rf --> 0.58
# cls3 = rf --> 0.60
# cls4 - rf --> 0.61