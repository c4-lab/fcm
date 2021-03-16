import within_fcm_walk
import cluster
import json_to_csv
import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist

from math import log
from sklearn.metrics import adjusted_rand_score

from pprint import pprint

def variation_of_information(X, Y):
    n = float(sum([len(x) for x in X]))
    sigma = 0.0
    for x in X:
        p = len(x) / n
        for y in Y:
            q = len(y) / n
            r = len(set(x) & set(y)) / n
            if r > 0.0:
                sigma += r * (log(r / p, 2) + log(r / q, 2))
    return abs(sigma)

# X1 = [ ['1','2','3','4','5'], ['6','7','8','9','10'] ]
# Y1 = [ ['1','2','3','4','5'], ['6','7','8','9','10'] ]
# print(variation_of_information(X1, Y1))
n_iters = 2
sample_step = 5000000
all_cluster_labels = []

for number_of_samples in range(1, n_iters+1):
# for number_of_samples in [5, 6]:
    number_of_samples = number_of_samples*sample_step
    for number_of_dimensions in [16]:
        print('\n******************\nnumber of dimensions: ', number_of_dimensions)
        within_fcm_walk.random_walk(number_of_samples= number_of_samples,
                                    number_of_dimensions= number_of_dimensions, window_size= 10,
                                    p= 0.5, q= 0.5, sample_step=sample_step)
    #     labels = cluster.clustering(threshold= 0, number_of_dimensions= number_of_dimensions)
    #     with open('../saved_models/directed_edgenode_uniform.embedding', 'r') as f:
    #         edgenodes = [' '.join(line.strip().split(' ')[:-number_of_dimensions]) for i, line in enumerate(f.readlines()) if i>0]
    #     edgenode_labels = []
    #     for cn in labels:
    #         temp_dict = {}
    #         for i in range(len(edgenodes)):
    #             temp_dict[edgenodes[i]] = cn[i]
    #         edgenode_labels.append(temp_dict)
    #     # pprint(edgenode_labels)
    #     # print(len(edgenodes), len(labels[0]))
    #     json_to_csv.two_dimensional_table()
    #     json_to_csv.one_dimensional_table()
    #     json_to_csv.clean()
    #     all_cluster_labels.append(edgenode_labels)
    # # calculate the adjusted rand index and variation of information here
    # # for cluster_inside_dimension in all_cluster_labels:
    # #     for x in cluster_inside_dimension:
    # #         print(len(x), end=',')
    # #     print()


number_of_dimensions = 16
for i in range(1, n_iters):
    print(i, i+1)
    with open('../saved_models/'+str(i)+'_edgenode_uniform.embedding', 'r') as f:
        u = []
        for idx, line in enumerate(f.readlines()):
            if idx!=0:
                line = line.strip().split(' ')[-number_of_dimensions:]
                line = [float(x) for x in line]
                u.append(line)
    # pprint(u)
    with open('../saved_models/'+str(i+1)+'_edgenode_uniform.embedding', 'r') as f:
        v = []
        for idx, line in enumerate(f.readlines()):
            if idx!=0:
                line = line.strip().split(' ')[-number_of_dimensions:]
                line = [float(x) for x in line]
                v.append(line)
    # print(len(u[0]), len(v[0]))
    u = np.array(u)
    v = np.array(v)
    mtx1, mtx2, disparity = procrustes(u, v)
    print('procrustes disparity: ', disparity)
    print('directed_hausdorff: ', max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0]))
    print('orthogonal procrustes: ', orthogonal_procrustes(u, v)[1])
    # print('cosine distance: ', cdist(u, v, metric='cosine'))
    # break


# for i_iter in range(0, n_iters-1):
#     for cn, cluster_number in enumerate(range(3, 12)):
#         print('cluster number: ', cluster_number)
#         X_dict, Y_dict = all_cluster_labels[i_iter][cn], all_cluster_labels[i_iter+1][cn]
#         # print(X_dict)
#         common_edgenodes = set(X_dict.keys()).intersection(set(Y_dict.keys()))
#         # print(common_edgenodes)
#         X_, Y_ = [], []
#         for key in common_edgenodes:
#             X_.append(X_dict[key])
#             Y_.append(Y_dict[key])
#         X, Y = [], []
#         for i in range(cluster_number):
#             clus = []
#             for x in range(len(X_)):
#                 if i == X_[x]:
#                     clus.append(x)
#             X.append(clus)
#             clus = []
#             for y in range(len(Y_)):
#                 if i == Y_[y]:
#                     clus.append(y)
#             Y.append(clus)
#         # pprint(X)
#         # pprint(Y)
#         print('VI: ', variation_of_information(X, Y))
#         print('Adj Rand Idx: ', adjusted_rand_score(X_, Y_))
