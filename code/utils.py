import csv
import os
import networkx as nx
import numpy as np
from pprint import pprint
import shutil

id_labels = {}
with open ('../data/Nodes/Nodes.csv', 'r') as f:
    for line in f.readlines ():
        line = line.strip ().split (',')
        id_labels[line[0]] = line[1]
    # print(id_labels)


def add_label():
    for fname in ['posgraph', 'neggraph']:
        labled_edge_lists = []
        with open ('../result/' + fname + '.csv', 'r') as f:
            for line in f.readlines ():
                line = line.strip ().split (',')
                line[0] = id_labels[line[0]]
                line[1] = id_labels[line[1]]
                labled_edge_lists.append (line)

        with open ('../result/' + fname + '.txt', 'w', newline='') as f:
            writer = csv.writer (f)
            writer.writerows (labled_edge_lists)


def normalized_grand_graph():
    labeled_edge_list = []
    edge_list = []
    for fname in ['posgraph', 'neggraph']:
        with open ('../result/' + fname + '.csv', 'r') as f:
            for line in f.readlines ():
                line = line.strip ().split (' ')
                if fname == 'neggraph':
                    line[2] = -1 * int (line[2])
                else:
                    line[2] = int (line[2])
                edge_list.append (line)
    min_weight = -1 * min (edge_list, key=lambda t: t[2])[2]
    for e in range (len (edge_list)):
        edge_list[e][2] = edge_list[e][2] + (min_weight + 1)
        labeled_e = edge_list[e][:]
        labeled_e[0] = id_labels[labeled_e[0]]
        labeled_e[1] = id_labels[labeled_e[1]]
        labeled_edge_list.append (labeled_e)

    with open ('../result/grandgraph.txt', 'w', newline='') as f:
        writer = csv.writer (f)
        writer.writerows (labeled_edge_list)
    with open ('../result/grandgraph.csv', 'w', newline='') as f:
        writer = csv.writer (f)
        writer.writerows (edge_list)


def added_node_graph():
    labeled_edge_list = []
    edge_list = []
    for fname in os.listdir ('../data/Edges/'):
        with open ('../data/Edges/' + fname, 'r') as f:
            for idx, line in enumerate (f.readlines ()):
                if idx == 0:
                    continue
                line = line.strip ().split (',')
                if float (line[2]) > 0:
                    edge_list.append (line)
                    labeled_edge_list.append ([id_labels[line[0]], id_labels[line[1]], line[2]])
                else:
                    edge_list.append ([line[0], '-' + (line[1]), line[2][1:]])
                    labeled_edge_list.append ([id_labels[line[0]], 'NOT '+ id_labels[line[1]], line[2][1:]])

    combinations = {}
    labeled_combinations = {}
    for line in edge_list:
        edge = ','.join(line[:2])
        if edge not in combinations.keys():
            combinations[edge] = 1
        else:
            combinations[edge] += 1
    for line in labeled_edge_list:
        edge = ','.join(line[:2])
        if edge not in labeled_combinations.keys():
            labeled_combinations[edge] = 1
        else:
            labeled_combinations[edge] += 1

    edge_list = []
    for k in combinations.keys():
        edge = k.split(',')
        edge.append(combinations[k])
        edge_list.append(edge)

    labeled_edge_list = []
    for k in labeled_combinations.keys ():
        edge = k.split (',')
        edge.append (labeled_combinations[k])
        labeled_edge_list.append (edge)

    with open ('../result/added_nodes_grandgraph.txt', 'w', newline='') as f:
        writer = csv.writer (f)
        writer.writerows (labeled_edge_list)
    # with open ('../result/added_nodes_grandgraph.csv', 'w', newline='') as f:
    #     writer = csv.writer (f)
    #     writer.writerows (edge_list)

    return


def added_node_fcm_file_generate():
    for fname in os.listdir ('../data/Edges/'):
        labeled_edge_list = []
        edge_list = []
        with open ('../data/Edges/' + fname, 'r') as f:
            for idx, line in enumerate (f.readlines ()):
                if idx == 0:
                    continue
                line = line.strip ().split (',')
                # print(line)
                if float (line[2]) > 0:
                    edge_list.append (line)
                    labeled_edge_list.append ([id_labels[line[0]], id_labels[line[1]]])
                else:
                    edge_list.append ([line[0], '-' + (line[1]), line[2][1:]])
                    labeled_edge_list.append ([id_labels[line[0]], 'NOT ' + id_labels[line[1]]])
        # pprint(labeled_edge_list)
        with open('../data/Labeled_Edges/'+fname, 'w') as fout:
            fout.write('Source,Target\n')
            for edge in labeled_edge_list:
                fout.write(','.join(edge)+'\n')

    return

def edges_convert_to_nodes_grandgraph():
    all_edge_graphs_weights = {}
    for fname in os.listdir('../data/Labeled_Edges/'):
        org_graph = nx.read_edgelist ('../data/Labeled_Edges/'+fname, nodetype=str,
                             create_using=nx.DiGraph,
                             data=(("weight", float),), delimiter=',')
        edge_graph = nx.line_graph(org_graph)
        for edgenode in edge_graph.edges:
            if edgenode not in all_edge_graphs_weights.keys():
                all_edge_graphs_weights[edgenode] = 1
            else:
                all_edge_graphs_weights[edgenode] += 1
    # pprint(all_edge_graphs_weights)
    all_edge_graphs_weights_edgelist = []
    for key in all_edge_graphs_weights.keys():
        # print(key[0], '->', key[1], ':', all_edge_graphs_weights[key])
        all_edge_graphs_weights_edgelist.append([str(key[0][0]+' >>>>> '+key[0][1]), str(key[1][0]+' >>>>> '+key[1][1]),
                                                 str(all_edge_graphs_weights[key])])
    # pprint(all_edge_graphs_weights_edgelist)
    with open('../result/edgenode_graph.txt', 'w') as fout:
        for edgenode in all_edge_graphs_weights_edgelist:
            fout.write(','.join(edgenode)+'\n')
    # converted_graph = nx.line_graph(org_graph)
    # o_adj = nx.adjacency_matrix(org_graph).todense()
    # np.savetxt('test1.out', o_adj, fmt='%d')
    # c_adj = nx.adjacency_matrix(converted_graph).todense()
    # np.savetxt ('testc.out', c_adj, fmt='%d')
    return

def separated_fcm_edgenode():
    all_edge_graphs_weights = {}
    edgenodes = set()
    for idx, fname in enumerate(os.listdir ('../data/Labeled_Edges/')):
        list_of_edges = []
        with open('../data/Labeled_Edges/' + fname, 'r') as curr_fcm:
            for x in curr_fcm.readlines ():
                if x=='Source,Target\n':
                    continue
                nodes = [str(idx)+'_'+n for n in x.strip().split(',')]
                list_of_edges.append(nodes)
            # pprint(list_of_edges)
        with open('temp_fcm_edgelist', 'w', newline='') as temp_fcm:
            writer = csv.writer (temp_fcm)
            writer.writerows (list_of_edges)
        # shutil.copyfile('../data/Labeled_Edges/' + fname, 'temp_fcm_edgelist')
        org_graph = nx.read_edgelist ('./temp_fcm_edgelist', nodetype=str,
                                      create_using=nx.DiGraph,
                                      data=(("weight", float),), delimiter=',')
        os.remove('./temp_fcm_edgelist')
        edge_graph = nx.line_graph (org_graph)
        for edgenode in edge_graph.edges:
            # print(edgenode)
            # edgenodes.add(edgenode)
            if edgenode not in all_edge_graphs_weights.keys ():
                all_edge_graphs_weights[edgenode] = 1
            else:
                all_edge_graphs_weights[edgenode] += 1
    # print(len(list_of_edges), len(edgenodes))
    pprint(all_edge_graphs_weights)
    all_edge_graphs_weights_edgelist = []
    for key in all_edge_graphs_weights.keys ():
        # print(key[0], '->', key[1], ':', all_edge_graphs_weights[key])
        all_edge_graphs_weights_edgelist.append (
            [str (key[0][0] + ' >>>>> ' + key[0][1]), str (key[1][0] + ' >>>>> ' + key[1][1]),
             str (all_edge_graphs_weights[key])])
    # pprint(all_edge_graphs_weights_edgelist)
    with open ('../result/edgenode_graph.txt', 'w') as fout:
        for edgenode in all_edge_graphs_weights_edgelist:
            fout.write (','.join (edgenode) + '\n')
    return

# normalized_grand_graph()
added_node_graph ()
# added_node_fcm_file_generate()
# edges_convert_to_nodes_grandgraph()
separated_fcm_edgenode()
