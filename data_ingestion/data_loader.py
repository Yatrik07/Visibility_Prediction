from TrainingData import  *
from Clustered_Data import *
import pandas as pd
class load_data:
    """
        class name : load_data

        parameters : None

        methods : 1. get_data
                  2. data_for_mdoel_training
                  3. load_clustered_data
    """

    def get_data(self):
        """
        method name : get_data

        parameters : None

        return : raw data ( for data preparation, EdA and Feature Engineering )
        """
        filename = 'InputFile.csv'
        data = pd.read_csv("..//TrainingData//"+filename)
        # data = pd.read_csv(filename)
        return data

    def data_for_mdoel_training(self):
        """
        method name : data_for_mdoel_training

        parameters : None

        return : Final Data in csv format for Model Training
        """
        filename = 'SData.csv'
        data = pd.read_csv("..//TrainingData//"+filename )
        return data


    def load_clustered_data():
        """
        method name : load_clustered_data

        parameters : None

        return : returns clustered data in csv format for Model Training
        """

        file_path = "..//TrainingData//Clusrtered_Data//"
        filename0 = 'CLUSTER0.csv'
        filename1 = 'CLUSTER1.csv'


        df0 = pd.read_csv(file_path + filename0)
        df1 = pd.read_csv(file_path + filename1)

        return df0 , df1


