#Imports
import os

import pandas as pd
import pickle

import stellargraph as sg
from stellargraph import StellarGraph

def createGraphLabeled(path):

    adj_pkl = pickle.load(open(path + 'adj.pkl', 'rb'))
    #adj_matrix = np.load('adj.npy')
    node_df = pickle.load(open(path + 'x.pkl','rb'))
    node_features = node_df.values

    label = int(open(path + 'label.txt', "r").read())
    # Print the contents

    # convert the numpy array of edges to a list of tuples
    edge_list = [(src, dst,wgt) for src, dst,wgt in adj_pkl.values]

    # create a pandas DataFrame from the list of tuples
    edge_df = pd.DataFrame(edge_list, columns=["source", "target","weight"])

    #create a StellarGraph object from the edge DataFrame
    #G = StellarGraph(edges=edge_df)
    G = StellarGraph(node_features, edges = edge_df)

    return G,label

def createGraphUnlabeled(path):

    adj_pkl = pickle.load(open(path + 'adj.pkl', 'rb'))
    #adj_matrix = np.load('adj.npy')
    node_df = pickle.load(open(path + 'x.pkl','rb'))
    node_features = node_df.values

    # convert the numpy array of edges to a list of tuples
    edge_list = [(src, dst,wgt) for src, dst,wgt in adj_pkl.values]

    # create a pandas DataFrame from the list of tuples
    edge_df = pd.DataFrame(edge_list, columns=["source", "target","weight"])

    #create a StellarGraph object from the edge DataFrame
    #G = StellarGraph(edges=edge_df)
    G = StellarGraph(node_features, edges = edge_df)

    return G

def readLabeledFolder(folder_name):

    graphs = []
    labels = []

    no_of_graphs = 4

    for i in range(1,no_of_graphs+1):
        path = os.getcwd() + '/' + folder_name + str(i) + '/'
        G,label = createGraphLabeled(path)
        graphs.append(G)
        labels.append(label)
    
    # Convert the list to a Categorical object
    cat_label = pd.Categorical(labels, categories=[0,1])

    # Create a Pandas Series object from the Categorical object
    labels = pd.Series(cat_label)

    return graphs, labels

def readUnlabaledFolder(folder_name):

    graphs = []
    no_of_graphs = len([f for f in os.listdir('graphsmallds/') if not f.startswith('.')])

    for i in range(1,no_of_graphs+1):
        path = os.getcwd() + '/' + folder_name + str(i) + '/'
        G = createGraphUnlabeled(path)
        graphs.append(G)

    return graphs
