import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

def get_dataframe(csvFile="companyinfo.csv"):
    return pd.read_csv(csvFile)

def getWords(words):
    if isinstance(words, str):
        res = words.split()
        res = [s.strip(',') for s in res]
        return res
    else:
        return []

df = get_dataframe()[['Company ID', 'Companies', 'Primary Industry Code', 'Keywords']]
#print(info)
array = getWords(df['Keywords'].iloc[0])
numEntries = len(df['Companies'])
keywordAry = df['Keywords'].to_numpy()

separateKeywordAry = [getWords(string) for string in keywordAry]
df['Individual Words'] = separateKeywordAry
#print(df)
df.drop('Keywords', axis=1, inplace=True)
df.drop('Primary Industry Code', axis=1, inplace=True)
df.rename(columns = {'Individual Words': 'Keywords'}, inplace = True)
#print(df)
df['keywords_str'] = df['Keywords'].apply(lambda x: ' '.join(x))
#print(df)

vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df['keywords_str'])

num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

cluster_labels = kmeans.labels_
df['cluster'] = cluster_labels
print(df)
