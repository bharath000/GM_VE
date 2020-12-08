import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from scipy.linalg import eigvals as eigs
from numpy import linalg as LA
from sklearn.cluster import KMeans
import sys
from scipy.sparse.linalg import eigsh
from scipy.cluster.vq import kmeans2

def read_data_nodes(path,pos):
    x_tp = []
    y_tp = []
    label_pos_dict = {}
    with open(path) as f:
        for line in f:
            x, y = line.split(",")
            x = float(x)
            y = float(y)
            
            x_tp.append((pos[x][0],pos[x][1]))
            y_tp.append(y)

            # if y in label_pos_dict.keys():
            #     label_pos_dict[y] = np.append(label_pos_dict[y],np.array(x_tp[-1]))
            # else:
            #     label_pos_dict[y] = np.array(x_tp[-1])

                           
    data_knn = np.array(x_tp)
    return data_knn, y_tp
def read_dict_node_label(path, pos):
    x_tp = []
    y_tp = []
    nodes_labels = {}
    with open(path) as f:
        for line in f:
            x, y = line.split(",")
            x = float(x)
            y = float(y)
            
            #x_tp.append((pos[x][0],pos[x][1]))
            #y_tp.append(y)

            if x in nodes_labels.keys():
                nodes_labels[x].append(y)
                #print(nodes_labels[x])
            else:
                nodes_labels[x] = [y]

    print(len(nodes_labels.keys()))

    return nodes_labels


def read_edges(path):
    x_tp = []
    
    with open(path) as f:
        for line in f:
            x, y = line.split(",")
            x = float(x)
            y = float(y)
            x_tp.append( (x,y) )
            #pos[str(k)] = (x,y)
            #k = k+1
    #data_knn = np.array(x_tp)
    return x_tp
def Accuracy(target, predicted):
    for i in range(1,len(target.keys())+1):
        cnt = 0
        if predicted[i-1] in target[i]:
            cnt += 1
    print("Accuracy: ",cnt/len(target.keys()))


def Figure_plot_clusters_all(data, labels, name):
    #km = KMeans(n_clusters=4,init='k-means++',random_state=0)
    #km.fit_predict(G)
    plt.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow', alpha=1, edgecolors='b')
    #print(km.labels_)
    
    plt.colorbar()

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return None

def Figure_plot_clusters(data, G, name):
    km = KMeans(n_clusters=39,init='k-means++',random_state=0)
    km.fit_predict(G)
    #plt.scatter(data[:,0], data[:,1], c=km.labels_, cmap='rainbow', alpha=1, edgecolors='b',s = 400)
    print(km.labels_)
    Accuracy(data, km.labels_)
   
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return None

    return data_knn, pos
if __name__ == "__main__":
    #creating graph object

    Graph = nx.Graph()
    
    edges = read_edges(sys.argv[1])
    print(len(edges))
    # Adding edges to the grpah
    Graph.add_edges_from(edges)
    pos = nx.spring_layout(Graph, seed = 100) 
    #print(pos)
    node_label_dict = read_dict_node_label("D:\MSCS\GraphMining\GM_VE\gm_project\data\soc-BlogCatalog-ASU.node_labels",pos)
    #exit()
    #node_pos, labels = read_data_nodes("D:\MSCS\GraphMining\GM_VE\gm_project\data\soc-BlogCatalog-ASU.node_labels",pos)

    #fig = plt.figure()
    #Figure_plot_clusters_all(node_pos,labels,"clusters_plot")
    #nx.draw(Graph,pos, node_size= 10)
    #plt.show()
    print("comptung LP")
    Graph_Lp = nx.normalized_laplacian_matrix(Graph,pos)
    print("computing values")

    eigenvalues, eigenvectors = eigsh(Graph_Lp.todense(), k=130)
    print(eigenvalues)
    print(eigenvectors)

    k = 2
    eigen_matrix = np.zeros((Graph_Lp.todense().shape[0],k))
    for i in range(0,k):
        eigen_matrix[:,i] = eigenvectors[:,i].transpose()

    Figure_plot_clusters(node_label_dict,eigen_matrix,"clustering")





    # #values, Vec = LA.eig(Graph_Lp.todense())
    # print("writing file")
    # #np.savetxt("D:\MSCS\GraphMining\GM_VE\gm_project\node_embeddings\eigan_values.csv", Vec, delimiter=',')
    # np.savetxt("D:/MSCS/GraphMining/GM_VE/gm_project/node_embeddings/eigan_values.csv", Vec, delimiter=',')

    # indis = np.argsort(values)
    # k = 39
    # eigen_matrix = np.zeros((Graph_Lp.todense().shape[0],k))
    # print("knn")
    # for i in range(0,k):
    #     eigen_matrix[:,i] = Vec[:,indis[i]].transpose()
    # #print(eigen_matrix)
    # Figure_plot_clusters(node_pos,eigen_matrix,"clustering")





    
    




 