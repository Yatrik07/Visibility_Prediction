from TrainingData import  *
from Clustered_Data import *
import pandas as pd
class load_data:
    def get_data(self):
         filename = 'InputFile.csv'
         data = pd.read_csv("..//TrainingData//"+filename)
         # data = pd.read_csv(filename)
         return data

    def data_for_mdoel_training(self):
        filename = 'SData.csv'
        data = pd.read_csv("..//TrainingData//"+filename )
        return data


    def load_clustered_data():
        file_path = "..//Clustered_Data//"
        filename1 = 'CLUSTER1.csv'
        filename2 = 'CLUSTER2.csv'
        filename3 = 'CLUSTER3.csv'
        filename4 = 'CLUSTER4.csv'

        df1 = pd.read_csv(file_path + filename1)
        df2 = pd.read_csv(file_path + filename2)
        df3 = pd.read_csv(file_path + filename3)
        df4 = pd.read_csv(file_path + filename4)

        return df1 , df2 , df3 , df4



