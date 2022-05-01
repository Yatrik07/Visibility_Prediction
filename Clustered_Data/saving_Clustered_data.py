from Clustering.reducing_features import clusters
from data_ingestion.data_loader import load_data

# No. of Reduced Features = 4 because more than 90% variance retained within 4 fatures 
# Saved photo in Clustering Folder --> PCA_variance_retained.jpg

# No. of clusters = 4 because maximum silhouette score is achieved with number of clusters = 4
# Saved photo in Clustering Folder --> silhouette score.jpg
#

def saving_clusters_Assigning_labels_y():
    LOADER = load_data()
    data = LOADER.data_for_mdoel_training()
    CLUSTERING = clusters(data)

    reduced_data , labels = CLUSTERING.makeclusters(4 , 4)

    data['cluster'] = labels

    # print(labels.shape)
    #
    # print(data)
    # print('\n',data.columns)
    # print("\n",data.cluster.unique())

    cluster1_data = data.query("cluster==0")
    cluster2_data = data.query("cluster==1")
    cluster3_data = data.query("cluster==2")
    cluster4_data = data.query("cluster==3")

    # print("cluster1_data",cluster1_data.shape)
    # print("cluster2_data", cluster2_data.shape)
    # print("cluster3_data", cluster3_data.shape)
    # print("cluster4_data", cluster4_data.shape)

    index1 = cluster1_data.index
    index2 = cluster2_data.index
    index3 = cluster3_data.index
    index4 = cluster4_data.index

    # print("index1", index1.shape)
    # print("index2", index2.shape)
    # print("index3", index3.shape)
    # print("index4", index4.shape)

    # print("c1:\n",cluster1_data)
    # print("c2:\n",cluster2_data)
    # print("c3:\n",cluster3_data)
    # print("c4:\n",cluster4_data)

    #cluster1_data.drop("cluster" , axis=1 , inplace=True)
    #cluster2_data.drop("cluster" , axis=1 , inplace=True)
    #cluster3_data.drop("cluster" , axis=1 , inplace=True)
    #cluster4_data.drop("cluster" , axis=1 , inplace=True)

    # print("c1:\n",cluster1_data)
    # print("c2:\n",cluster2_data)
    # print("c3:\n",cluster3_data)
    # print("c4:\n",cluster4_data)



    # cluster1_data.to_csv("CLUSTER1.csv" , index = False)
    # cluster2_data.to_csv("CLUSTER2.csv" , index = False)
    # cluster3_data.to_csv("CLUSTER3.csv" , index = False)
    # cluster4_data.to_csv("CLUSTER4.csv" , index = False)

    return cluster1_data , cluster2_data,  cluster3_data , cluster4_data , index1 , index2 , index3 , index4