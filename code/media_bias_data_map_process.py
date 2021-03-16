import pandas as pd
import numpy as np
from pprint import pprint
import math

df = pd.read_excel('../result/qualtrix_survey_cleaned_appended.xlsx')
media_use = [x.split('_')[0].split('-')[1].strip() for x in df.iloc[0].to_list() if '-' in x]
media_use = set(media_use)
print(media_use)

placeholders = ['nan', 'social media', 'Public Broadcasting / International', 'cable and network news ', 'Other',
                'Government Reporting', 'Digital Native News', 'Add your own', 'Alternative News', 'Print Media']
df = pd.read_excel('../data/media/Media Bias Chart.xlsx')
media_all = set([x for x in df.iloc[:, 0].to_list() if str(x) not in placeholders])
print(media_all)


print(media_all.union(media_use).difference(media_all.intersection(media_use)))

df = df[df['Unnamed: 0'].isin(media_all)]
df.columns = ['media', 'adfonte_reliability', 'adfonte_bias', 'mediabiasfactcheck.com_reliability', 'mediabiasfactcheck.com_bias']

for col in ['adfonte_reliability', 'adfonte_bias']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(math.inf)

df.fillna('n/a', inplace=True)
print(df.head())

adfonte_bias_conditions = [
    (df['adfonte_bias']>=-6.0) & (df['adfonte_bias']<=6.0),
    (df['adfonte_bias']>=-18.0) & (df['adfonte_bias']<=-6.0),
    (df['adfonte_bias']>=-30.0) & (df['adfonte_bias']<=-18.0),
    (df['adfonte_bias']>=-42.0) & (df['adfonte_bias']<=-30.0),
    (df['adfonte_bias']>=6.0) & (df['adfonte_bias']<=18.0),
    (df['adfonte_bias']>=18.0) & (df['adfonte_bias']<=30.0),
    (df['adfonte_bias']>=30.0) & (df['adfonte_bias']<=42.0)
]
adfonte_bias_choices = ['neutral', 'skews-left', 'hyperpartisan-left', 'mostextreme-left',
                               'skews-right', 'hyperpartisan-right', 'mostextreme-right']
adfonte_bias_choices = [0, -1, -2, -3, 1, 2, 3]

adfonte_reliability_conditions = [
    (df['adfonte_reliability']/ 8.0<=1.0),
    (df['adfonte_reliability'] / 8.0 <= 2.0),
    (df['adfonte_reliability'] / 8.0 <= 3.0),
    (df['adfonte_reliability'] / 8.0 <= 4.0),
    (df['adfonte_reliability'] / 8.0 <= 5.0),
    (df['adfonte_reliability'] / 8.0 <= 6.0),
    (df['adfonte_reliability'] / 8.0 <= 7.0),
    (df['adfonte_reliability'] / 8.0 <= 8.0),
]
adfonte_reliability_choices = ['inaccurate/fabricated', 'misleading', 'selective/incomplete/unfair persuasion/propaganda',
                               'opnion', 'analysis', 'mix of fact reporting and analysis', 'fact reporting', 'original fact reporting']
adfonte_reliability_choices = [x for x in range(len(adfonte_reliability_choices))]

df['adfonte_bias_discrete'] = np.select(adfonte_bias_conditions, adfonte_bias_choices, default='n/a')
df['adfonte_reliability_discrete'] = np.select(adfonte_reliability_conditions, adfonte_reliability_choices, default='n/a')
# print(df['adfonte_reliability_discrete'].to_list())
df = df.reindex(sorted(df.columns), axis=1)
print(df.columns)
df.to_excel('../result/media_bias_chart_cleaned.xlsx', index=False)
