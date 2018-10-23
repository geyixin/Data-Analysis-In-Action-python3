#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

'''
利用k-means实现离群点检测
'''

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inputpath = '../Data/consumption_data.xls'
data = pd.read_excel(inputpath, index_col='Id')
# print(data.head(3))
data_standard = (data - data.mean())/data.std()  # 归一化
# print(data_standard.head(3))

k = 3   # 簇的个数
threshold = 2   # 离群点阈值
iteration = 500  # 聚类最大循环次数

model = KMeans(n_clusters=k, n_jobs=1, max_iter=iteration)
model.fit(data_standard)

'''
数据标准化：数据连接+列名重置
'''
r = pd.concat([data_standard, pd.Series(model.labels_, index=data.index)], axis=1)  # 连接
'''
data_standard：
           R         F         M
Id                              
1   0.764186 -0.493579 -1.158711
2  -1.024757 -0.630079  0.622527
3  -0.950217  0.871423 -0.341103

model.labels_：
1
1
0

r:
           R         F         M  0
Id                                 
1   0.764186 -0.493579 -1.158711  1
2  -1.024757 -0.630079  0.622527  1
3  -0.950217  0.871423 -0.341103  0

'''
r.columns = list(data.columns) + ['label']   # 重命名列名
'''
r:
           R         F         M  聚类类别
Id                                    
1   0.764186 -0.493579 -1.158711     1
2  -1.024757 -0.630079  0.622527     1
3  -0.950217  0.871423 -0.341103     0
'''
# print(r.head(3))

norm = []
for i in range(k):
    # 下面这行作用：找到列名为label，且值为 i 的所有行，然后把每一行的中列名为 R F M三列取出来，放进norm_temp中。
    # print(r.head(3))
    norm_temp = r[['R','F','M']][r['label'] == i] - model.cluster_centers_[i]
    # print(norm_temp)
    norm_temp = norm_temp.apply(np.linalg.norm, axis=1)   # axis=1:按行求范数; =0，按列；=None，矩阵范数
    # print(norm_temp.head(3))
    norm.append(norm_temp/norm_temp.median())
    # print(norm[0][:3])

norm = pd.concat(norm)
# print(len(norm))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

norm[norm <= threshold].plot(style='go')

discrete_point = norm[norm > threshold]
discrete_point.plot(style='ro')

'''
标记离群点
'''
# print(discrete_point)
for i in range(len(discrete_point)):
    index = discrete_point.index[i]   # 取index
    distance = discrete_point.iloc[i]   # 取值
    plt.annotate('(%s, %.2f)' % (index, distance),
                 xy=(index, distance), xytext=(index, distance))
    # plt.annotate('(%s, %.2f)'%(index, distance),
    #              xy=(index, distance), xytext=(index/2, distance/2),
    #              arrowprops=dict(facecolor="y", headlength=3,
    #                              headwidth=3,width=2))
# 第一个参数是注释内容；xy:箭头尖的位置；xytext：注释内容起始位置
plt.xlabel('编号')
plt.ylabel('相对距离')
plt.grid()
plt.show()






