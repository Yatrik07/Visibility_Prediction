from Clustering.reducing_features import clusters
from data_ingestion.data_loader import load_data

# No. of Reduced Features = 4 because more than 90% variance retained within 4 fatures 
# Saved photo in Clustering Folder --> PCA_variance_retained.jpg

# No. of clusters = 4 because maximum silhouette score is achieved with number of clusters = 4
# Saved photo in Clustering Folder --> silhouette score.jpg
#

def saving_clusters_Assigning_labels_y():
    '''
    method name : saving_clusters_Assigning_labels_y

    parameters : None

    work : Gets and saves proper clustered data in csv format from method:makeclusters->class:clusters->file:reducing_features.py

    return: Clustered Data and their respective cluster labels
    '''
    LOADER = load_data()
    data = LOADER.data_for_mdoel_training()
    CLUSTERING = clusters(data.drop("VISIBILITY" , axis = 1 ))

    reduced_data , labels = CLUSTERING.makeclusters(2 , 4)
    reduced_data.to_csv("sample_reduced_data.csv")
    data['cluster'] = labels

    cluster0_data = data.query("cluster==0")
    cluster1_data = data.query("cluster==1")

    index0 = cluster0_data.index
    index1 = cluster1_data.index

    cluster0_data.drop("cluster" , axis=1 , inplace=True)
    cluster1_data.drop("cluster" , axis=1 , inplace=True)

    cluster0_data.to_csv("CLUSTER0_1.csv" , index = False)
    cluster1_data.to_csv("CLUSTER1_1.csv" , index = False)

    return cluster0_data, cluster1_data, index0, index1


saving_clusters_Assigning_labels_y()
