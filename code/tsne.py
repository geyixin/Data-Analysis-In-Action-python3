#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

'''
  TSNE
降维、高维图以低维图显示
'''
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inputPath = '../Data/consumption_data.xls'

data = pd.read_excel(inputPath, index_col='Id')
data_standard = (data - data.mean())/data.std()

k = 3
interation = 500

# print(data_standard)

model = KMeans(n_clusters=k, n_jobs=1, max_iter=interation)
model.fit(data_standard)
r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
r.columns = list(data.columns) + ['聚类类别']

tsne = TSNE()
tsne.fit_transform(data_standard)
# print(data_standard.head(3))
tsne = pd.DataFrame(tsne.embedding_, index=data_standard.index)
# print(tsne.head(3))

d = tsne[r['聚类类别'] == 0]
# print(d)
plt.plot(d[0], d[1], 'r.')

d = tsne[r['聚类类别'] == 1]
plt.plot(d[0], d[1], 'go')

d = tsne[r['聚类类别'] == 2]
plt.plot(d[0], d[1], 'b*')
# print(r)
print(type(r))   # <class 'pandas.core.frame.DataFrame'>
plt.show()
plt.savefig('../Result/tene.png')




