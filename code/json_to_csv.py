import json
import pandas as pd
import os
import xlsxwriter
from pprint import pprint

def two_dimensional_table():
    flag = 'directed'
    for app in ['freq', ]:
        # 'dist'
        list_of_df = []
        cluster_numbers = [cluster_number for cluster_number in range(3, 12)]
        filename = '../result/representatives/'+'edgesclusters_'+app
        writer = pd.ExcelWriter (filename+'.xlsx', engine='xlsxwriter')
        for cluster_number in cluster_numbers:
            filename = 'representatives/'+'edgesclusters_'+app+'_'+str(cluster_number)
            with open('../result/'+filename+'.json', 'r') as f:
                myjson = json.load(f)
                # pprint(myjson)
                for key in myjson:
                    edgenodeslist = []
                    for edgenode in myjson[key]:
                        edgenodeslist.append(edgenode[0])
                    myjson[key] = edgenodeslist
                maxlen = -1
                for k in myjson.keys():
                    if len(myjson[k])>maxlen:
                        maxlen = len(myjson[k])
                # print(maxlen)
                for k in myjson.keys():
                    klen = len(myjson[k])
                    for i in range(klen, maxlen):
                        myjson[k].append('')
                # pprint(myjson)
                df = pd.DataFrame(myjson)
                list_of_df.append(df)
        for i, df in enumerate(list_of_df):
            df.to_excel(writer, sheet_name='%s' %cluster_numbers[i], index=False)
            # pprint(df)
            # df.to_excel('../result/'+filename+'.xlsx', index=False)
        writer.save ()


def one_dimensional_table():
    flag = 'directed'
    for app in ['freq', ]:
        # 'dist'
        list_of_df = []
        cluster_numbers = [cluster_number for cluster_number in range (3, 12)]
        filename = '../result/representatives/' + 'edgesclusters_' + app
        writer = pd.ExcelWriter (filename + '_one_dimension.xlsx', engine='xlsxwriter')
        for cluster_number in cluster_numbers:
            edgenodes_cluster_freq = []
            filename = 'representatives/' + 'edgesclusters_' + app+'_'+str(cluster_number)
            with open('../result/'+filename+'.json', 'r') as f:
                myjson = json.load(f)
                # pprint(myjson)
                for cluster in myjson.keys():
                    for edgenode in myjson[cluster]:
                        if app == 'dist':
                            edgenodes_cluster_freq.append([edgenode[0], int(cluster), float(edgenode[1]), int(edgenode[2])])
                        elif app == 'freq':
                            edgenodes_cluster_freq.append ([edgenode[0], int (cluster), int (edgenode[1])])
            df = pd.DataFrame(edgenodes_cluster_freq)
            if app == 'dist':
                df.columns = ['edgenode', 'cluster', 'dist', 'freq']
                df = df.sort_values (['cluster', 'dist', 'freq'], ascending=[True, True, False])
            else:
                df.columns = ['edgenode', 'cluster', 'freq']
                df = df.sort_values(['cluster', 'freq'], ascending=[True, False])
            list_of_df.append(df)
        # filename += ''
        # df.to_excel ('../result/' + filename + '.xlsx', index=False)
        for i, df in enumerate(list_of_df):
            df.to_excel(writer, sheet_name='%s' %cluster_numbers[i], index=False)
        writer.save()

# two_dimensional_table()
# print('***********')
# one_dimensional_table()

def clean():
    for filename in os.listdir('../result/representatives/'):
        if filename.endswith('.json'):
            os.remove('../result/representatives/'+filename)
