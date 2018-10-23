#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

input = '../Data/bankloan.xls'
data = pd.read_excel(input)
# print(data.head(3))
x = data.iloc[:,:8].as_matrix()     # iloc:切片，as_matrix()：转换成数组形式
y = data.iloc[:,8].as_matrix()

rlr = RLR()
rlr.fit(x, y)
res = rlr.get_support()   # 获取特征筛选结果,保留的为True，舍弃的为False
# rlr.scores_()   # 也可通过此方式获取各个特征分数
# print(res)
# print('有效特征：%s' % ','.join(np.array(data.iloc[:,:8].columns)[res]))
# print(np.array(data.iloc[:,:8].columns)[res])

x = data[np.array(data.iloc[:,:8].columns)[res]].as_matrix()    # 重新确定 x
# print(x[:3,:])

lr = LR()
lr.fit(x, y)
print('模型正确率：%s' % lr.score(x, y))