import numpy as np
# np.random.seed(0)
import os
import json
import pandas as pd
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn_extra.cluster import KMedoids
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from shutil import copyfile

scaler = MinMaxScaler ()

id_labels = {}
with open ('../data/Nodes/Nodes.csv', 'r') as f:
    for line in f.readlines ():
        line = line.strip ().split (',')
        id_labels[line[0]] = line[1]
# pprint(id_labels)


flag = 'directed'
edge_clustering_file = 'edgesclusters_11'
cluster_search = {}
with open ('../result/' + edge_clustering_file + '.json', 'r') as f:
    clusters = json.load (f)
    for key in clusters.keys ():
        for edge in clusters[key]:
            cluster_search[edge] = key

pprint (cluster_search)
print (len (cluster_search.keys ()))


def jaccard_similarity_calc(labeled_edge_list):
    set1 = set (labeled_edge_list)
    cluster_jaccardscore = {}
    for key in clusters.keys ():
        set2 = set (clusters[key])
        jaccardscore = len (set1.intersection (set2)) * 1.0 / len (set1.union (set2))
        cluster_jaccardscore[key] = jaccardscore
    return cluster_jaccardscore

def coverage_calc(labeled_edge_list):
    set1 = set (labeled_edge_list)
    cluster_coveragescore = {}
    for key in clusters.keys ():
        set2 = set (clusters[key])
        jaccardscore = len (set1.intersection (set2)) * 1.0 / len (set1)
        cluster_coveragescore[key] = jaccardscore
    return cluster_coveragescore


def edge_clusters(labeled_edge_list):
    edges_cluster_labels = []
    for edge in labeled_edge_list:
        try:
            cluster = cluster_search[edge]
        except Exception as e:
            cluster = ''
        # print(edge, '\t', cluster)
        edges_cluster_labels.append (cluster)
    edges_cluster_labels[:] = [x for x in edges_cluster_labels if x != '']
    edges_cluster_labels = '-'.join (edges_cluster_labels)
    # print (edges_cluster_labels)
    return edges_cluster_labels


# fout1 = open ('../result/fcm_coveragfe_' + edge_clustering_file + '.csv', 'w')
# fout1.write ('fcm\tfcmcluster\n')
fcm_jaccards = []
for fname in os.listdir ('../data/Labeled_Edges/'):
    print (fname)
    labeled_edge_list = []
    with open ('../data/Labeled_Edges/' + fname, 'r') as f:
        for idx, line in enumerate (f.readlines ()):
            if idx == 0:
                continue
            line = line.strip ().split (',')
            print (line)
            edge = ' >>>>> '.join (line)
            labeled_edge_list.append (edge)
    coverages = coverage_calc (labeled_edge_list)
    coverages['fcm'] = fname[:-4]
    print (coverages)
    fcm_jaccards.append (coverages)
coverages_df = pd.DataFrame (fcm_jaccards)
coverages_df.to_csv ('../result/' + 'raw_coverage.csv', index=False)
jaccard_labels = [x for x in coverages_df.columns if x != 'fcm']
print (jaccard_labels)
coverages_df[jaccard_labels] = scaler.fit_transform (coverages_df[jaccard_labels])
pprint (coverages_df)
coverages_df.to_csv ('../result/' + 'scaled_coverage.csv', index=False)


'''
def draw_scree_plot(A):
    dimensions = coverages_df.shape[1] - 1
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
    # plt.savefig ('../plots/'+edge_clustering_file+'_jaccard_screeplot.png')
    plt.clf ()
    return


def decomposition_and_clustering():
    # reduced_dimensions = 2
    # svd = TruncatedSVD (n_components=reduced_dimensions, n_iter=10)
    # decomposed_X = svd.fit_transform (jaccards_df[jaccard_labels])
    # print (type (decomposed_X))
    #
    # decomposed_X_df = pd.DataFrame (data=decomposed_X,
    #                                 columns=['dim_' + str (x + 1) for x in range (reduced_dimensions)])
    # pprint (decomposed_X_df)

    decomposed_X = coverages_df[jaccard_labels]

    model = KMedoids (init='k-medoids++')
    visualizer = KElbowVisualizer (model, k=(2, 30))
    visualizer.fit (decomposed_X)
    # visualizer.show (outpath='../plots/' + edge_clustering_file + '_jaccard_optimum-k.png')
    clustering_algorithm = KMedoids (n_clusters=visualizer.elbow_value_, init='k-medoids++')
    labels = clustering_algorithm.fit (decomposed_X).labels_
    fcm_names = list (coverages_df['fcm'])
    print (len (set (labels)))
    # for i in range (len (fcm_names)):
    #     fout1.write (fcm_names[i] + '\t' + str (labels[i]) + '\n')

    fout2 = open ('../result/representatives/' + 'fcm_kmedoids.txt', 'w')
    for i in list (clustering_algorithm.medoid_indices_):
        fout2.write (coverages_df['fcm'].tolist ()[int (i)] + '\n')
    # fout1.close ()
    fout2.close ()


def representatives():
    with open ('../result/representatives/' + 'fcm_kmedoids.txt') as f:
        for line in f.readlines ():
            fname = line.strip ()
            copyfile ('../data/Labeled_Edges/' + fname+'.csv' , '../result/representatives/' + fname+'.csv')


def representative_jaccards():
    # jaccards = pd.read_csv('../result/scaled_jaccards.csv')
    # pprint(jaccards)
    representatives = []
    with open ('../result/representatives/' + 'fcm_kmedoids.txt') as f:
        for fcm in f.readlines ():
            representatives.append (fcm.strip ())
    pprint (representatives)
    coverages_df[coverages_df['fcm'].isin (representatives)].to_csv (
        '../result/representatives/' + 'scaled_coverages.csv', index=False)


draw_scree_plot (coverages_df[jaccard_labels])
# decomposition_and_clustering ()
# representatives ()
# representative_jaccards ()
# my_model = PCA (n_components=0.80)
# my_model.fit_transform (jaccards_df[jaccard_labels])
# print (my_model.explained_variance_ratio_.cumsum ())
'''
