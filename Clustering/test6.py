import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import NMF
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from scipy.linalg import eigvals as eigs
from numpy import linalg as LA
from sklearn.cluster import KMeans
from karateclub import SymmNMF

def create_graph(G,k,path):
    graph_knn = kneighbors_graph(G, n_neighbors=k, mode='distance')
    G_matrix = graph_knn.todense()
    print(G_matrix.shape)
    G_matrix=G_matrix+G_matrix.transpose()
    vfunc = np.vectorize(lambda l: int(l>1))
    Adj = vfunc(G_matrix)
    
    np.fill_diagonal(Adj, 0)
    Graph = nx.from_numpy_matrix(Adj)
    np.savetxt(path, Adj, delimiter=',')
    #print(Adj.shape)
    return Graph, Adj


def plot_knn_graph(G):
    color = ['r','b']
    node_color = []
    for i in range(0,373):
        if i < 97:
            node_color.append(color[0])
        else:
            node_color.append(color[1])
    fig = plt.figure()
    nx.draw(G, with_labels=True,node_color=node_color)

    plt.title("Knn Graph")
    plt.show()



def read_data(path):
    x_tp = []
    y_tp = ()
    label = []
    with open(path) as f:
        for line in f:
            x, y, z = line.split()
            x = float(x)
            y = float(y)
            x_tp.append( (x,y) )
    data_knn = np.array(x_tp)
    return data_knn

def Fit_plot_clusters(data, G, name):
    #model = NMF(n_components=2, init='random', random_state=0)
    #W = model.fit_transform(G)
    #H = model.components_
    #print(W)
    model = SymmNMF(dimensions=2)
    W = model.fit(G)
    x = model.get_memberships() 
    H = model.get_embedding()
    #print(H.shape)
    #exit()
    #pred = KMeans(n_clusters=2).fit_predict(H)
    #print(pred)
    #colors = {pred[0]:"red",  pred[372]:"blue"}
    #print(colors)
   # groups = {"o":"Cluster {}".format(pred[0]),"^":"Cluster {}".format(pred[372])}
    #Markers = {pred[0]:"o", pred[372]:"^"}
    c1 = []
    for i in range(0,373):
        c1.append(x[i])

    #c1=[x[i] for i in sorted(x.keys(), reverse=True)] 
    #m=[Markers[i] for i in pred]
    #for i in range(0,373):
    plt.scatter(data[:,0],data[:,1], alpha=0.8, c=c1,cmap='rainbow')
        #plt.annotate(str(i), (data[i,0],data[i,1]))
    plt.title('Points after clustering using {}'.format(name))
    #plt.legend(handles=groups, loc=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return None

if __name__ == "__main__":
    # sigma = [[1, 0], [0, 1]] 
    # G2 = np.random.multivariate_normal([2,2], sigma, 20)
    # data = np.
    # print(G2.shape)
    data = read_data('D:/MSCS/GraphMining/GM_VE/assig_5/Dataset1.txt')
    print(data.shape)
    
    Graph, Adj = create_graph(data, 8,'D:/MSCS/GraphMining/GM_VE/assig_5/test.out')
    A = nx.adjacency_matrix(Graph)
    #plot_knn_graph(Graph)
    # Graph_Lp = nx.laplacian_matrix(Graph)
    # I = np.identity(373)
    # #print(I - Graph_Lp)
    # #exit()
    # values, Vec = LA.eig(Graph_Lp.todense())
    # print(values.shape)
    # indis = np.argsort(values)
    # k = 3
    # eigen_matrix = np.zeros((Graph_Lp.todense().shape[0],k))
    # for i in range(0,k-1):
    #     eigen_matrix[:,i] = Vec[:,indis[i]].transpose()



    # Fit_plot_clusters(data,Graph,"clustering")