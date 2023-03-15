#Imports

import pandas as pd
import pickle

import stellargraph as sg
from stellargraph import StellarGraph

def createGraph(path):

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