
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('dataset/data.csv')

df.head()

df.describe()



columns = df.columns.to_list()



columns.remove('label')

columns.remove('filename')


X = df[columns]

Y = df['label']

X.head()

Y.head()

minmax= MinMaxScaler()

scaledX = pd.DataFrame(minmax.fit_transform(X), columns=X.columns.to_list())

scaledX.head()



len(Y.unique())

kmeans=KMeans(n_clusters=len(Y.unique()), max_iter=3000, verbose=True)

kmeans.fit(scaledX, Y)

clusters = kmeans.labels_


def purity_score(y_true, y_pred):

    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

metrics.cluster.contingency_matrix(Y, clusters)

purity = purity_score(Y, clusters)
print(purity)


