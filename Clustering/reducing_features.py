from pyexpat import model
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class clusters:
    '''
    This class is for creating clusters from the data

    parameters:

    df :
     The DataFrame to create clusters
    '''

    def __init__(self, df):
        self.df = df

    def applyPCA(self, df, no_of_features):
        '''Applying PCA for Clustering'''
        pca = PCA(n_components=no_of_features)
        reduced_df = pca.fit_transform(df)
        return reduced_df

    def makeclusters(self, no_of_clusters, no_of_features):
        '''
         Make clusters from the data

        parameters:

        no_of_clusters :
         Number of clusters to create in clustering
        no_of_features :
         Number of features to keep in Principal Component Analysis
         (Reducing the data for better result in clustering)

        '''
        reduced_df = self.applyPCA(self.df, no_of_features)
        kmeans_model = KMeans(n_clusters=no_of_clusters)
        kmeans_model.fit(reduced_df)
        labels = kmeans_model.labels_
        return reduced_df, labels
