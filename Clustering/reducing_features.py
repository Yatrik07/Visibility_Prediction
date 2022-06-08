from pyexpat import model
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import joblib
import pandas as pd

class clusters:
    '''
    This class is for creating clusters from the data

    parameters:
        df : The DataFrame to create clusters
    '''

    def __init__(self, df):
        self.df = df

    def applyPCA(self, df, no_of_features):
        """
        method name : applyPCA

        work : Applying PCA for Clustering

        parameters : DataFrame , Number Of Features to reduce

        return : returns reduced DataFrae after appluing Principal Component Analysis
        """

        pca = PCA(n_components=no_of_features)
        reduced_df = pca.fit_transform(df)
        joblib.dump(pca , 'pca_obj.pickle')
        return reduced_df

    def makeclusters(self, no_of_clusters, no_of_features):
        """
        method name : makeclusters

        work : Make clusters from the data

        parameters :
            no_of_clusters :
             Number of clusters to create in clustering
            no_of_features :
             Number of features to keep in Principal Component Analysis
             (Reducing the data for better result in clustering)

        return : returns reduced DataFrame and its respective lables [ of clusters]
        """

        reduced_df = self.applyPCA(self.df, no_of_features)
        kmeans_model = KMeans(n_clusters=no_of_clusters , random_state=10)
        kmeans_model.fit(reduced_df)
        labels = kmeans_model.labels_
        joblib.dump(kmeans_model, "Clustering_model.pickle")
        return pd.DataFrame(reduced_df), labels

    def Cluster_accuracy(self):
        """
        method name : Cluster_accuracy

        work : Check clustering accuracy using silhouette score and save the graph showing the result

        parameters : None

        return :None
        """
        reduced_df = self.applyPCA(self.df, 4)
        silhouette = []
        for i in range(2,7):
            km_model = KMeans(n_clusters=i)
            km_model.fit(reduced_df)
            km_labels = km_model.predict(reduced_df)
            silhouette.append(silhouette_score(reduced_df , km_labels))
        plt.plot(range(2,7),  silhouette)
        plt.xlabel("No. of clusters")
        plt.xlabel("silhouette score")
        plt.title("silhouette score")
        plt.savefig("silhouette score.jpg")

'''
from data_ingestion.data_loader import load_data
LOADER = load_data()
data = LOADER.data_for_mdoel_training()
# CLUSTERING = clusters(data.drop("VISIBILITY" , axis=1))
c=clusters(data.drop("VISIBILITY" , axis=1))
c.Cluster_accuracy()
'''