import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from scipy.linalg import eigvals as eigs
from numpy import linalg as LA
from sklearn.cluster import KMeans
from karateclub import SymmNMF
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




def Fit_plot_clusters(data, G, name):
    # km = KMeans(n_clusters=4,init='k-means++',random_state=0)
    # km.fit_predict(G)
    # plt.scatter(data[:,0], data[:,1], c=km.labels_, cmap='rainbow', alpha=1, edgecolors='b',s = 100)
    # print(km.labels_)

    model = SymmNMF(dimensions=4)
    W = model.fit(G)
    x = model.get_memberships() 
    H = model.get_embedding()

    c1 = []
    for i in range(0,75):
        c1.append(x[i])

    plt.scatter(data[:,0],data[:,1], alpha=0.8, c=c1,cmap='rainbow')


    
    # colors = {0:"red",  1:"blue", 2:"black", 3:"orange"}
    # print(colors)
    # #groups = {"o":"Cluster {}".format(pred[0]),"^":"Cluster {}".format(pred[372])}
    # Markers = {0:"o", 1:"^",2:"*",3:"v"}
    # c=[colors[i] for i in pred] 
    # m=[Markers[i] for i in pred]
    # for i in range(0,75):
    #     plt.plot(data[i,0],data[i,1], alpha=2, c=c[i], marker=m[i])
    #     plt.annotate(str(i), (data[i,0],data[i,1]))
    # plt.title('Points after clustering using {}'.format(name))
    # #plt.legend(handles=groups, loc=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return None


if __name__ == "__main__":
    data, pos = read_data('D:/MSCS/GraphMining/GM_VE/assig_5/Dataset2Nodes_Layout.txt')
    Graph = nx.Graph()
    for i in range(0,data.shape[0]):
        Graph.add_node(i, pos = (data[i][0], data[i][1]))
    edges = read_edges('D:/MSCS/GraphMining/GM_VE/assig_5/Dataset2Edges.txt')
    Graph.add_edges_from(edges)    
    pos=nx.get_node_attributes(Graph,'pos')
    #Graph = nx.read_edgelist('D:/MSCS/GraphMining/GM_VE/assig_5/Dataset2Edges.txt')
    fig = plt.figure()
    #nx.draw_networkx_nodes(Graph,pos)
    #nx.draw_networkx_edges(Graph,pos)
    #plt.title("Knn Graph")
    nx.draw(Graph,pos, node_size= 10)
    #plt.show()
    
    Graph_Lp = nx.normalized_laplacian_matrix(Graph,pos)
    #Graph_Lp = nx.laplacian_matrix(Graph)
    Adj = nx.adjacency_matrix(Graph,pos)
    #print(Adj.todense())
    #np.savetxt('D:/MSCS/GraphMining/GM_VE/assig_5/test2edges.out', Adj.todense(), delimiter=',')
    #exit()
    values, Vec = LA.eig(Graph_Lp.todense())
    print(Vec.shape)
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
    #Fit_plot_clusters(data,eigen_matrix,"clustering")
    Fit_plot_clusters(data,Graph,"clustering")
