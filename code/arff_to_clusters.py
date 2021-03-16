import pandas as pd
from pprint import pprint
filename = 'em'
with open('../weka_result/'+filename+'.arff', 'r') as fin:
    lines = [line.strip() for line in fin.readlines()]
    start = lines.index('@data')+1
    print(start)
    lines = lines[start:]

# pprint(lines)

clusters = {}
for line in lines:
    line = line.split(',')
    # print(line)
    edgenode = line[1].replace('\'', '')
    edgenode.replace('dont', 'don\'t').replace('Individuals', 'Individual\'s')\
        .replace('US governments late response to virus control','US government\'s late response to virus control')
    cluster = line[-1]
    # print(edgenode, cluster)
    if cluster not in clusters.keys():
        clusters[cluster] = [edgenode]
    else:
        clusters[cluster].append(edgenode)

# pprint(clusters)
maxlen = -1
for k in clusters.keys():
    if len(clusters[k])>maxlen:
        maxlen = len(clusters[k])
for k in clusters.keys():
    if len(clusters[k])!=maxlen:
        for i in range(maxlen-len(clusters[k])):
            clusters[k].append('')

df = pd.DataFrame.from_dict(clusters)
pprint(df.head())

df.to_excel('../weka_result/'+filename+'.xlsx', index=False)


with open('../result/added_nodes_grandgraph.txt', 'r') as f:
    edgenodes = [[' >>>>> '.join(line.strip().split(',')[:2]), line.strip().split(',')[2]] for line in f.readlines()]

edgenodes_freq = {}
for edgenode in edgenodes:
    if edgenode[0] not in edgenodes_freq.keys():
        edgenodes_freq[edgenode[0]] = int(edgenode[1])
    else:
        pass

# pprint(edgenodes_freq)


clusters_freq = {}
for k in clusters.keys():
    cluster = clusters[k]
    clusters_freq[k] = []
    for edgenode in cluster:
        if edgenode=='':
            continue
        edgenode = edgenode.replace('dont', 'don\'t').replace('Individuals', 'Individual\'s') \
            .replace('US governments late response to virus control','US government\'s late response to virus control')
        clusters_freq[k].append([edgenode, edgenodes_freq[edgenode]])

for k in clusters_freq.keys():
    clusters_freq[k].sort(key=lambda x:x[1], reverse=True)
    clusters_freq[k] = [edgenode[0] for edgenode in clusters_freq[k][:5]]
pprint(clusters_freq)

df = pd.DataFrame.from_dict(clusters_freq)
df.to_excel('../weka_result/'+filename+'_representatives.xlsx', index=False)
