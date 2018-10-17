#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

'''
连续属性离散化

三种常用方法：
1. 等宽法
2. 等频法
3. 聚类
'''
import pandas as pd
from sklearn.cluster import KMeans

input = '../Data/discretization_data.xls'
data = pd.read_excel(input)
data = data['肝气郁结证型系数'].copy()
# print(data)
k = 4
label = range(k)

d1 = pd.cut(data, k, labels=label)  # 等宽离散
# print(d1)

w = [1.0*i/k for i in range(k+1)]
w = data.describe(percentiles=w)[4:4+k+1]
w[0] = w[0]*(1-1e-10)
# print(w)
d2 = pd.cut(data, w, labels=label)  # 等频离散

kmodel = KMeans(n_clusters=k, n_jobs=1)  # 建立模型，n_jpbs：并行数，一般等于CPU数量
kmodel.fit(data.values.reshape((len(data), 1)))    # 训练模型
# print(kmodel.cluster_centers_)
c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)   # 聚类中心,排序
# print(type(c))
# print(c.sort_values(0),'\n','---------')
# print(pd.DataFrame.rolling(c,2))
w = pd.DataFrame.rolling(c,2).mean().iloc[1:]   # 将相邻两项中点作为边界点
# print(w)
w = [0] + list(w[0]) + [data.max()]
# print(w)
d3 = pd.cut(data, w, labels=label)

# print(d1,d2,d3)

'''
绘图显示
'''

def cluster_plot(d, k):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SemHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8,3))
    for j in range(0,k):
        plt.plot(data[d==j], [j for i in d[d==j]], 'o')  # 纵坐标，横坐标，显示格式

    plt.ylim(-0.5, k-0.5)   # y坐标刻度范围
    return plt

num = 1
for d in [d1,d2,d3]:
    outpath = '../Result/' + 'd' + str(num) + '.png'
    num += 1
    # print(outpath)
    cluster_plot(d, k).savefig(outpath)
# cluster_plot(d1, k).show()