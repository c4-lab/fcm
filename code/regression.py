import pandas as pd
import statsmodels.api as sm
from pprint import pprint

mediadf = pd.read_excel('../result/survey_response_merged_media_metric.xlsx')
pprint(mediadf.columns)

surveydf = pd.read_excel('../data/survey_map_metadata.xlsx')
surveydf = surveydf[['FCM ID', 'Survey Response ID Part 1', 'Survey Response ID Part 2']]
surveydf.columns = ['fcm', 'response_id', 'response_id_2']
pprint(surveydf.columns)

# fcmdf = pd.read_csv('../result/directed_scaled_jaccards.csv')
# pprint(fcmdf.columns)
fcmdf = pd.read_csv('../result/scaled_coverage.csv')
pprint(fcmdf.columns)

# mediadf = mediadf.merge(surveydf, on='response_id')
# print(mediadf.head())

fcmdf = fcmdf.merge(surveydf, on='fcm')
pprint(fcmdf.head())

mediadf = mediadf.merge(fcmdf, on='response_id')
print(mediadf.head())

mediadf.to_excel('../result/regression.xlsx', index=False)

mediadf = pd.read_excel('../result/regression.xlsx')
print(mediadf.columns)

for cluster in range(0, 11):
    cluster = str(cluster)
    X = mediadf[['bias_score', 'frequency_score', 'iqr_score']]
    y = mediadf[cluster]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    print(cluster, str(model.summary()))

    with open('../result/regression/'+str(cluster)+'.txt', 'w') as f:
        f.write(str(model.summary()))
