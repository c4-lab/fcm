import pandas as pd
from scipy.stats import entropy
import numpy as np
from pprint import pprint

media_bias_df = pd.read_excel('../result/media_bias_chart_cleaned.xlsx')
pprint(media_bias_df.columns)
media_bias_df = media_bias_df[['media', 'adfonte_bias_discrete', 'adfonte_reliability_discrete']]
media_bias_df = media_bias_df.dropna()
media_dict = {}
for idx, row in media_bias_df.iterrows():
    if row['media'] not in media_dict.keys():
        media_dict[row['media'].lower()] = {'bias': row['adfonte_bias_discrete'], 'reliability': row['adfonte_reliability_discrete']}
    else:
        print('xxxx')

# pprint(media_dict)

survey_df = pd.read_excel('../result/qualtrix_survey_cleaned_appended.xlsx')
survey_df.columns = [x.lower() for x in survey_df.loc[0, :].values.tolist()]
survey_df = survey_df.iloc[1:]
# pprint(survey_df.head())

survey_responses_scores = {'response_id': [], 'bias_score': [], 'reliability_volume': [], 'diversity_score': [],
                           'frequency_score': [], 'iqr_score': []}
for idx, row in survey_df.iterrows():
    response_id = row['response id']
    bias_score = 0.0
    reliability_volume = 0.0
    frequency_score = 0
    media_count_list = []
    bias_range_list = []
    for col in survey_df.columns:
        media_col = col[:-len ('_boolean')] + '_numeric'
        freq = 0
        for media in media_dict.keys():
            if '_boolean' in col and media in col:
                if row[col]==True:
                    freq = row[media_col]
                    # print(media, row[media_col], media_dict[media])
                    bias_score += freq*media_dict[media]['bias']
                    for i in range(freq):
                        bias_range_list.append(media_dict[media]['bias'])
                    reliability_volume += freq*media_dict[media]['reliability']
                    frequency_score += freq
                media_count_list.append (freq)
    diversity_score = entropy(media_count_list) if len(media_count_list)!=0 else 0.0
    iqr_score = np.percentile(np.array(bias_range_list), 75) - np.percentile(np.array(bias_range_list), 25) if len(bias_range_list)!=0 else 0.0
    print(response_id, bias_score, reliability_volume, len(media_count_list), diversity_score, frequency_score, iqr_score)
    survey_responses_scores['response_id'].append(response_id)
    survey_responses_scores['bias_score'].append(bias_score)
    survey_responses_scores['reliability_volume'].append(reliability_volume)
    survey_responses_scores['diversity_score'].append(diversity_score)
    survey_responses_scores['frequency_score'].append(frequency_score)
    survey_responses_scores['iqr_score'].append(iqr_score)

for k in survey_responses_scores.keys():
    print(k, len(survey_responses_scores[k]))
df = pd.DataFrame.from_dict(survey_responses_scores)
df = df.fillna(-999)
pprint(df.head())
df.to_excel('../result/survey_response_merged_media_metric.xlsx', index=False)
