import pandas as pd
from pprint import pprint
import os
import re

mycolumns = 'ResponseId,Q1_1,Q1_2,Q1_3,Q1_4,Q1_5,Q1_6,Q1_7,Q1_8,Q1_9,Q1_10,Q1_11,Q1_12,Q1_13,Q1_14,Q1_15,Q2_1,Q2_2,Q2_3,Q2_4,Q2_5,Q2_6,Q2_7,Q3_1,Q3_2,Q3_3,Q3_4,Q3_5,Q3_6,Q4_1,Q4_2,Q4_3,Q4_4,Q4_5,Q4_6,Q4_7,Q4_8,Q4_9,Q4_10,Q4_11,Q4_12,Q5_1,Q5_2,Q5_3,Q5_4,Q5_5,Q5_6,Q5_7,Q5_8,Q5_9,Q5_10,Q5_11,Q5_12,Q5_13,Q5_14,Q5_15,Q5_16,Q5_17,Q5_18,Q5_19,Q5_20,Q5_21,Q5_22,Q5_23,Q5_24,Q5_25,Q5_26,Q5_27,Q5_28,Q5_29,Q5_30,Q5_31,Q5_32,Q5_33,Q5_34,Q5_35,Q5_36,Q5_37,Q5_38,Q5_39,Q6_1,Q6_2,Q6_3,Q6_4,Q6_5,Q6_6,Q6_7,Q6_8,Q6_9,Q7_1,Q7_2,Q7_3,Q7_4,Q8_1,Q8_2,Q8_3'.split(',')
# print(mycolumns)
df = pd.read_excel('../data/media/qualtrix_survey_1.xlsx')
df = df[mycolumns]
df = df.dropna(how='all')
df = df.fillna('')
for col in df.columns:
    df[col] = df[col].str.replace('\n', ' ')
    df[col] = df[col].str.replace('\r', ' ')

column_texts = df.iloc[0].to_list()
clean_column_names = []
xstr = 'In the following, please indicate  which of the following media channels you use for any reason.  It is only necessary to mark the boxes that  apply; sources which do not have any marks will be interpreted as not being  used.'
xstr_ = 'During the last 7 days, how many days did you use the following media  sources? Also, please check box if you used this media source specifically for information related to Covid-19.'
for c in column_texts:
    c = c.replace(xstr, '').replace(xstr_, '').strip()
    c = re.sub(' +', ' ', c)
    # name = c.split(' ')
    # print(name)
    # print(c)
    clean_column_names.append(c)
# pprint(df.head())
df.iloc[0] = clean_column_names

# pprint(df.head())
df_records = {'id': df['ResponseId'].to_list()}

boolean_ques = 'Used for information related to COVID-19?'
for col in df.columns:
    if '_' in col:
        df_records[col+'_days'] = None
        df_records[col+'_'+boolean_ques] = None

# print(df_records)

for col in df.columns:
    if '_' in col:
        numeric = []
        boolean = []
        for idx, row in df.iterrows():
            if idx==1:
                numeric.append(row[col]+'_numeric')
                boolean.append(row[col]+'_boolean')
                continue
            responses = row[col]
            if boolean_ques in responses:
                boolean.append(True)
            else:
                boolean.append(False)
            responses = responses.replace(boolean_ques, '')
            responses = responses.split(' ')[0]
            responses = int(responses) if responses!='' else 0
            numeric.append(responses)
        df_records[col+'_days'] = numeric
        df_records[col+'_'+boolean_ques] = boolean

# pprint(df_records)

df = pd.DataFrame.from_dict(df_records)
# df.columns = df.loc[0, :].values.tolist()
# df = df.iloc[1:]
# pprint(df.head())
df.to_excel('../result/qualtrix_survey_cleaned_appended.xlsx', index=False)
