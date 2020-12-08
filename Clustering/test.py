import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from scipy.linalg import eigvals as eigs
from numpy import linalg as LA
from sklearn.cluster import KMeans
import sys

def create_graph(G,k):
    #used sklearn to create the kneighbor graph an in build function.
    graph_knn = kneighbors_graph(G, n_neighbors=k, mode='distance')
    #Converting the data in form of array into matrix
    G_matrix = graph_knn.todense()
    G_matrix=G_matrix+G_matrix.transpose()
    vfunc = np.vectorize(lambda l: int(l>1))
    #Creating the Adjacency Matrix from the data generated above
    Adj = vfunc(G_matrix)
    # Diagonal matrix of Adjacency matrix into zeros
    np.fill_diagonal(Adj, 0)
    # Creating Graph from the Adjacency matrix using the networkx library
    Graph = nx.from_numpy_matrix(Adj)
    #print(Adj.shape)
    return Graph

def plot_knn_graph(G, data):
    color = ['r','b']
    node_color = []
    # create positions for the nodes from the data
    for i in range(0,data.shape[0]):
        G.add_node(i, pos = (data[i][0], data[i][1]))
    pos=nx.get_node_attributes(Graph,'pos')
    for i in range(0,373):
        if i < 97:
            node_color.append(color[0])
        else:
            node_color.append(color[1])
    fig = plt.figure()
    # use positions and node colors for ploting the graph
    nx.draw(G, pos,with_labels=True,node_color=node_color)

    plt.title("Knn Graph")
    plt.show()

def read_data(path):
    x_tp = []
    y_tp = ()
    label = []
    # reading each line and spliting to get the x and y co-ordinate
    with open(path) as f:
        for line in f:
            x, y, z = line.split()
            x = float(x)
            y = float(y)
            x_tp.append( (x,y) )
    # All the x and y co-ordinates are added to a list in the form of tuples
    # Conver the list of points into a numpy array.
    data_knn = np.array(x_tp)
    return data_knn

def Figure_plot_clusters(data, G, name):
    # clustering the graph data
    pred = KMeans(n_clusters=2).fit_predict(G)
    print(pred)
    # predicitng from the data 
    colors = {pred[0]:"red",  pred[372]:"blue"}
    print(colors)
    #
    Markers = {pred[0]:"o", pred[372]:"^"}
    c=[colors[i] for i in pred] 
    m=[Markers[i] for i in pred]
    # ploting the clusters using markers and color
    for i in range(0,373):
        plt.plot(data[i,0],data[i,1], alpha=0.8, c=c[i], marker=m[i])
        plt.annotate(str(i), (data[i,0],data[i,1]))
    plt.title('Points after clustering using {}'.format(name))
    #plt.legend(handles=groups, loc=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return None



if __name__ == "__main__":
    
    # Reading the data from text file
    data = read_data(sys.argv[1])
   
    # checking the shape of returned numpy array
    print(data.shape)
    # Creating the K-nn graph with k = 8 as the input of the the given dataset/data
    Graph = create_graph(data, 8)
    # Ploting K-NN graph
    plot_knn_graph(Graph, data)
    #exit()
    # Getting Noramlised Laplasian Matrix from the graph
    Graph_Lp = nx.normalized_laplacian_matrix(Graph)
    # Generating eigen values and Vvectors from the Laplacian matrix
    values, Vec = LA.eig(Graph_Lp.todense())

    print(values.shape)
    # getting the indices of least eigen values
    indis = np.argsort(values)
    k = 3
    # Generate Matric of eigen vectors for each point based on eigen value
    eigen_matrix = np.zeros((Graph_Lp.todense().shape[0],k))

    for i in range(0,k-1):
        eigen_matrix[:,i] = Vec[:,indis[i]].transpose()
    #make the data into clusters and ploting those clusters
    Figure_plot_clusters(data,eigen_matrix,"splectrual clustering")

