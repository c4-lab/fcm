import matplotlib
import matplotlib.pyplot as plt

from pprint import pprint

cluster_numbers = []
ari = []

filenames = [('console_out_iteration.txt', 67), ('console_out.txt', 249)]
x = 0
with open('../result/'+filenames[x][0]) as f:
    for i, line in enumerate(f.readlines()):
        if i < filenames[x][1]:
            continue
        print(line.strip())
        if 'cluster number:' in line:
            cluster_numbers.append(int(line.split(' ')[-1]))
        elif 'Adj Rand Idx:' in line:
            ari.append(float(line.split(' ')[-1]))

pprint(cluster_numbers)
pprint(ari)

plt.plot(cluster_numbers, ari, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('cluster numbers')
plt.ylabel('ARI')

plt.show()
