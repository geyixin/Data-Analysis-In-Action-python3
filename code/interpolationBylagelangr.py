#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

'''
使用拉格朗日法对缺失数据插补
'''

import pandas as pd
from scipy.interpolate import lagrange

input = '../data/catering_sale.xls'
output = '../DataOut/sales.xls'

data = pd.read_excel(input)
data['销量'][(data['销量'] < 400) | (data['销量'] > 5000)] = None
# print(data)

'''
s: 列向量
n: 需要插值的位置
k: 取得前后的数据个数，默认为5
'''
def insertdata(s, n, k=5):
    y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))]
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)

LL = len(data)

for i in data.columns:
    for j in range(LL):
        if (data[i].isnull())[j]:
            data[i][j] = insertdata(data[i],j)

# print(data)

data.to_excel(output)













