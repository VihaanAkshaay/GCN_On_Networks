import pandas as pd
import numpy as np

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
import tensorflow as tf


def generateNewData(model,graphs,LOW_THRESHOLD = 0.2,HIGH_THRESHOLD = 0.8):
    '''
    Creates labels for unlabeled graphs and returns new training data
    
    Inputs:
    -------
    
    model : Trained tf model that is used to predict/generate labels
    graphs: Graphs for which we generate new labels
    
    Outputs:
    --------
    new_graphs: List of graphs that the model can classify with confidence
    graph_labels: Labels for the above graphs
    '''
    
    new_graphs = []
    new_labels = []
    
    new_gen = PaddedGraphGenerator(graphs)
    
    create_gen = new_gen.flow(graphs,
    batch_size=1,
    symmetric_normalization=False,)
    
    pred_labels = model.predict(create_gen)
    
    for i in range(len(graphs)):
        pred_val = pred_labels[i]
    
        if pred_val < LOW_THRESHOLD:
            new_graphs.append(graphs[i])
            new_labels.append(0)
        
        elif pred_val > HIGH_THRESHOLD:
            new_graphs.append(graphs[i])
            new_labels.append(1)
        
    # Convert the list to a Categorical object
    cat_label = pd.Categorical(new_labels, categories=[0,1])

    # Create a Pandas Series object from the Categorical object
    graph_labels = pd.Series(cat_label)
    
    
    return new_graphs,graph_labels


def appendTwoDatasets(graphs_old,graph_labels_old,graphs_new,graph_labels_new):
    '''
    Takes in old graphs, labels and the new ones and appends them to return one dataset
    
    
    Inputs:
    -------
    graphs_old: List of old graph objects
    graph_labels_old: Pandas categorical series object of old labels
    new ones are new of corresponding data
    
    Outputs:
    --------
    
    '''
    final_graphs = graphs_old + graphs_new
    final_labels = pd.concat([graph_labels_old, graph_labels_new], ignore_index=True)
    
    return final_graphs,final_labels