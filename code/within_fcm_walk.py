import random
import numpy as np
# random.seed(0)
# np.random.seed(0)
import networkx as nx
import os
import individual_node2vec
from gensim.models import Word2Vec
from pprint import pprint
import json

graph_filename = 'edgenode_graph.txt'


def learn_embeddings(walks, dimensions, window_size, workers, iter, outfile, min_count=0):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    # walks = [map (str, walk) for walk in walks]
    model = Word2Vec (walks, size=dimensions, window=window_size,
                      min_count=min_count, sg=1, workers=workers, iter=iter)
    model.wv.save_word2vec_format (outfile)
    return


is_directed = True
graph = nx.read_edgelist ('../result/' + graph_filename, nodetype=str,
                          create_using=nx.DiGraph,
                          data=(("weight", float),), delimiter=',')
flag = 'directed'

def random_walk(number_of_samples, number_of_dimensions, window_size, p, q, sample_step):
    node2vecspace = individual_node2vec.Graph(graph, is_directed, p=p, q=q)
    node2vecspace.preprocess_transition_probs()
    walks = node2vecspace.simulate_walks(num_walks=number_of_samples, walk_length=15)
    # pprint(walks)
    # with open('../saved_models/'+flag+'_walks.json', 'w') as f:
    #     json.dump(walks, f, indent=4)
    learn_embeddings(walks, dimensions=number_of_dimensions, window_size=window_size,
                     min_count=5, workers=10, iter=10,
                     outfile='../saved_models/'+str(int(number_of_samples/sample_step))+'_edgenode_uniform.embedding')


# random_walk(50000, 64, 10, 0.5, 0.5)
