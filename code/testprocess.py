from pprint import pprint
with open('../saved_models/directed_edgenode_uniform_processed.embedding', 'r') as f:
    lines = [' '.join(line.strip().split(' ')[:-16]).replace('\'', '')+','+','.join(line.strip().split(' ')[-16:]) for line in f.readlines()]
    pprint(lines)

with open('../saved_models/directed_edgenode_uniform_processed.csv', 'w') as f:
    for idx, line in enumerate(lines):
        if idx==0:
            colnames = ['edgenode']
            for i in range(16):
                colnames.append('dim_'+str(i+1))
            f.write(','.join(colnames)+'\n')
            continue
        f.write(line+'\n')
