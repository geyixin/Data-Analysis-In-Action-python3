#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
from sklearn.decomposition import PCA

input = '../Data/principal_component.xls'
ouput = '../DataOut/component_pca.xls'

data = pd.read_excel(input, header=None)

'''
主成分 初探
'''

'''
pca = PCA()
pca.fit(data)
print(pca.components_)  # 模型的各个特征向量
print('--------------------')
print(pca.explained_variance_ratio_)    # 递减显示各个特征向量的各自的方差百分比（又称贡献率）

# 依次计算前几个贡献率之和，发现前三个之和已经不错了
for i in range(1,5):
    print(sum(pca.explained_variance_ratio_[:i]))

'''

'''
主成分 确定 重建模型
'''

pca = PCA(3)
pca.fit(data)
low_d = pca.transform(data)   # 降维
print(low_d)
pd.DataFrame(low_d).to_excel(ouput)    # 保存为excel
# init_data = pca.inverse_transform(low_d)    # 复原数据（尽量，无法完全复原）
# print(init_data)