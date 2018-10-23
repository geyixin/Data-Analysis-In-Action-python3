#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
from sklearn.cluster import KMeans

inputPath = '../Data/consumption_data.xls'
ouputPath = '../DataOut/consumption_data_fixed.xls'

k = 3
interation = 500

data = pd.read_excel(inputPath, index_col='Id')
data_standard = (data - data.mean())/data.std()

# print(data_standard)

model = KMeans(n_clusters=k, n_jobs=1, max_iter=interation)
model.fit(data_standard)


'''
简单打印
'''
'''
r1 = pd.Series(model.labels_).value_counts()    # 统计各个类别数目
r2 = pd.DataFrame(model.cluster_centers_)   # 聚类中心
# print(r1,r2)
r = pd.concat([r2,r1], axis=1)
# print(list(data.columns))
r.columns = list(data.columns) + ['类别数目']
# print(r)
'''

'''
详细输出
'''
# print(model.labels_)
# print(data.index)
r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
r.columns = list(data.columns) + ['聚类类别']
# print(r)
# r.to_excel(ouputPath)
pd.DataFrame(r).to_excel(ouputPath)     # 两种输出方式

print(len(data.iloc[0]))