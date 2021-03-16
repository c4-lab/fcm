import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation,\
    FeatureAgglomeration, OPTICS, MeanShift, SpectralClustering, SpectralBiclustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from pprint import pprint
import json
import pandas as pd

from math import inf

# import random
# random.seed(0)
# np.random.seed(0)

n_representatives = 10
# print(random.getstate())

labeled_edges = []
with open ('../result/added_nodes_grandgraph.txt') as f:
    for line in f.readlines ():
        edge = line.strip ().split (',')
        edge[2] = int (edge[2])
        labeled_edges.append (edge)

sorted_labeled_edges_temp = sorted (labeled_edges, key=lambda x: x[2], reverse=True)
sorted_labeled_edges_temp = [[x[0] + ' >>>>> ' + x[1], x[2]] for x in sorted_labeled_edges_temp]
# pprint(sorted_labeled_edges)

# for x in sorted_labeled_edges_temp:
#     edgenodes_freq_dict[x[0]] = x[1]
# pprint(edgenodes_freq_dict)

def clustering(threshold, number_of_dimensions):
    sorted_labeled_edges = []
    edgenodes_freq_dict = {}
    for x in sorted_labeled_edges_temp:
        if x[1] >= threshold:  # 5
            sorted_labeled_edges.append (x)
            edgenodes_freq_dict[x[0]] = x[1]
    print (len (edgenodes_freq_dict.keys ()))

    all_labels = []

    for cluster_number in range (3, 12):
        flag = 'directed'
        dimensions = number_of_dimensions
        embeddings = []
        nodes = []

        embedding_fname = flag + '_edgenode_uniform.embedding'

        with open ('../saved_models/' + embedding_fname, 'r') as f:
            for idx, line in enumerate (f.readlines ()):
                if idx == 0:
                    continue
                line = line.strip ().split (' ')
                # print(line)
                node = ' '.join (line[:-dimensions])
                # print(node)
                embedding = [float (x) for x in line[-1 * dimensions:]]
                if node in edgenodes_freq_dict.keys ():
                    nodes.append (node)
                    embeddings.append (embedding)
        print (len (nodes), len (embeddings))

        print ('cluster count: ', cluster_number)


        def draw_scree_plot(A):
            A = np.asmatrix (A.T) * np.asmatrix (A)
            U, S, V = np.linalg.svd (A)
            eigvals = S ** 2 / np.sum (S ** 2)

            fig = plt.figure (figsize=(8, 5))
            sing_vals = np.arange (dimensions) + 1
            plt.plot (sing_vals, eigvals, 'ro-', linewidth=2)
            plt.title ('Scree Plot')
            plt.xlabel ('Principal Component')
            plt.ylabel ('Eigenvalue')
            # I don't like the default legend so I typically make mine like below, e.g.
            # with smaller fonts and a bit transparent so I do not cover up data, and make
            # it moveable by the viewer in case upper-right is a bad place for it
            leg = plt.legend (['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                              shadow=False, prop=matplotlib.font_manager.FontProperties (size='small'),
                              markerscale=0.4)
            leg.get_frame ().set_alpha (0.4)
            # leg.draggable (state=True)
            # plt.show()
            plt.savefig ('../plots/' + ''.join (embedding_fname.split ('.')[:-1]) + '_screeplot.png')
            plt.clf ()
            return


        def find_n_representatives_freq(edges_clusters):
            # pprint(sorted_labeled_edges)
            edges_representatives = {}
            for key in edges_clusters.keys ():
                edges = edges_clusters[key]
                representatives = []
                representative_i = 0
                # print('*')
                for edge_weight in sorted_labeled_edges:
                    # print(edge_weight)
                    edge_string = edge_weight[0]
                    # print(edge_string)
                    if edge_string in edges:
                        representatives.append ((edge_string, edge_weight[1]))
                        # print(edge_string, edge_weight[2])
                        representative_i += 1
                        # if representative_i == n_representatives:
                        #     break
                edges_representatives[key] = representatives
            return edges_representatives


        def find_n_representatives_dist(cluster_centers, labels):
            edge_representatives = {}
            for cluster in labels:
                distdf = pd.DataFrame.from_records (X)
                distdf['cluster_label'] = labels
                distdf = distdf[distdf['cluster_label'] == cluster].drop ('cluster_label', 1)
                X_ = distdf.to_numpy ()
                indices = distdf.index
                center_, dist_ = -1, inf
                for c in cluster_centers:
                    for sample in X_:
                        dist = np.linalg.norm (c - sample)
                        if dist < dist_:
                            dist_ = dist
                            center_ = c
                dists = []
                for i in range (X_.shape[0]):
                    d = np.linalg.norm (X_[i,] - center_)
                    dists.append ((i, d))
                # pprint(dists)
                # edge_representatives_indices_ = [x[0] for x in sorted (dists, key=lambda tup: tup[1])][:n_representatives]
                edge_representatives_indices_ = [(x[0], x[1]) for x in sorted (dists, key=lambda tup: tup[1])][:]
                # # print(n_top_rep)
                # return n_top_rep
                # edge_representatives_indices_ = find_n_representatives (center_, X_)
                edge_representatives_indices = [(indices[x[0]], x[1]) for x in edge_representatives_indices_]
                # print(edge_representatives_indices)

                representatives = []
                for c in edge_representatives_indices:
                    en = nodes[c[0]]
                    for x in sorted_labeled_edges:
                        if en == x[0]:
                            f = x[1]
                            # print(en, f)
                    representatives.append ((en, c[1], f))
                edge_representatives[str (cluster)] = representatives
            # print(center)
            # pprint(X_.shape)
            # print(len(edge_representatives.keys()))
            return edge_representatives


        def decomposition_and_clustering(nodes, X):
            # svd = TruncatedSVD(n_components=12, n_iter=12)
            # X = svd.fit_transform(X)

            # model = KMeans (init='k-means++')
            # visualizer = KElbowVisualizer (model, k=(2, 35))
            #
            # visualizer.fit (X)
            # visualizer.show (outpath='../plots/'+''.join(embedding_fname.split('.')[:-1])+'_optimum-k.png')
            # print('optimum k value:', visualizer.elbow_value_)

            # pprint(X)
            # clustering_algorithm = KMeans(n_clusters=visualizer.elbow_value_, init='k-means++')
            # clustering_algorithm = KMeans (n_clusters=cluster_number, init='k-means++', )
            # clustering_algorithm = DBSCAN(eps=0.01)
            # clustering_algorithm = AgglomerativeClustering(n_clusters=cluster_number)
            # clustering_algorithm = FeatureAgglomeration(n_clusters=cluster_number)
            clustering_algorithm = AffinityPropagation(damping=0.5, max_iter=300)
            # clustering_algorithm = OPTICS(metric='correlation')
            # clustering_algorithm = MeanShift()
            # clustering_algorithm = SpectralBiclustering(n_clusters=cluster_number)
            # OPTICS()
            clustering_algorithm.fit (X)
            labels = clustering_algorithm.labels_
            all_labels.append(labels)

            # silhouette = silhouette_score (X, labels, metric='euclidean', sample_size=len (X))
            # print ('silhouette: ', silhouette, 'inertia: ', clustering_algorithm.inertia_)
            clusters = {}
            for c in range (len (labels)):
                label = str (labels[c])
                edgenode = nodes[c]
                if label not in clusters.keys ():
                    clusters[label] = [edgenode]
                else:
                    clusters[label].append (edgenode)
            # pprint(clusters)
            with open ('../result/' + 'edgesclusters_' + str (cluster_number) + '.json', 'w') as f:
                json.dump (clusters, f, indent=4)

            # edge_representatives = find_n_representatives_dist (clustering_algorithm.cluster_centers_, labels)
            # with open ('../result/representatives/' + 'edgesclusters_dist_' + str (cluster_number) + '.json', 'w') as f:
            #     json.dump (edge_representatives, f, indent=4)
            edge_representatives = find_n_representatives_freq (clusters)
            with open ('../result/representatives/' + 'edgesclusters_freq_' + str (cluster_number) + '.json', 'w') as f:
                json.dump (edge_representatives, f, indent=4)
            return


        X = np.array ([np.array (a) for a in embeddings])
        # pprint(X)
        # draw_scree_plot(X)
        decomposition_and_clustering (nodes, X)
    return all_labels

# clustering(5, 64)
