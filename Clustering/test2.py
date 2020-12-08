import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from scipy.linalg import eigvals as eigs
from numpy import linalg as LA
from sklearn.cluster import KMeans
import sys

# reading data_nodes and its layout
def read_data(path):
    x_tp = []
    y_tp = ()
    label = []
    k = 0
    pos = {}
    with open(path) as f:
        for line in f:
            x, y = line.split()
            x = float(x)
            y = float(y)
            x_tp.append( (x,y) )
            pos[str(k)] = (x,y)
            k = k+1
    data_knn = np.array(x_tp)

    return data_knn, pos

# reading Edges
def read_edges(path):
    x_tp = []
    y_tp = ()
    label = []
    k = 0
    pos = {}
    with open(path) as f:
        for line in f:
            x, y = line.split()
            x = float(x)
            y = float(y)
            x_tp.append( (x,y) )
            #pos[str(k)] = (x,y)
            #k = k+1
    #data_knn = np.array(x_tp)
    return x_tp



# plotting the Clusters
def Figure_plot_clusters(data, G, name):
    km = KMeans(n_clusters=4,init='k-means++',random_state=0)
    km.fit_predict(G)
    plt.scatter(data[:,0], data[:,1], c=km.labels_, cmap='rainbow', alpha=1, edgecolors='b',s = 400)
    print(km.labels_)
    
   
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return None


if __name__ == "__main__":
    # Reading the Nodes
    data, pos = read_data(sys.argv[1])
    # creating graph object using networkx
    Graph = nx.Graph()
    # Adding nodes and its positions to the graph object created above
    for i in range(0,data.shape[0]):
        Graph.add_node(i, pos = (data[i][0], data[i][1]))
    # reading the edges
    edges = read_edges(sys.argv[2])
    # Adding edges to the grpah
    Graph.add_edges_from(edges)    
    pos=nx.get_node_attributes(Graph,'pos')
    #Graph = nx.read_edgelist('D:/MSCS/GraphMining/GM_VE/assig_5/Dataset2Edges.txt')
    fig = plt.figure()
    #nx.draw_networkx_nodes(Graph,pos)
    #nx.draw_networkx_edges(Graph,pos)
    #plt.title("Knn Graph")
    # Plotting the Graph 
    nx.draw(Graph,pos, node_size= 10)
    #plt.show()
    # Gettting the Normalised Lapalisian matrix from the grpah
    Graph_Lp = nx.normalized_laplacian_matrix(Graph,pos)
    #Graph_Lp = nx.laplacian_matrix(Graph)
    #Adj = nx.adjacency_matrix(Graph,pos)
    #print(Adj.todense())
    #np.savetxt('D:/MSCS/GraphMining/GM_VE/assig_5/test2edges.out', Adj.todense(), delimiter=',')
    #exit()

    ### generate Eigen values
    values, Vec = LA.eig(Graph_Lp.todense())
    print(Vec.shape)
    # Getting indexes after sorting
    indis = np.argsort(values)
    k = 4
    i = np.where(values < 0.5)[0]
    # print(i.shape)
    U = np.array(Vec[:, i[1:4]])
    
    
    #Fit_plot_clusters(data,U,"clustering")
    eigen_matrix = np.zeros((Graph_Lp.todense().shape[0],k))
    for i in range(0,k):
        eigen_matrix[:,i] = Vec[:,indis[i]].transpose()
    #print(eigen_matrix)
    Figure_plot_clusters(data,eigen_matrix,"clustering")

    