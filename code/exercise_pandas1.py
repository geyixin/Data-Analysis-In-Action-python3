#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd

data = pd.read_csv('../data/air_data_small.csv', encoding='utf-8')

# data = data[data['A'].notnull()*data['B'].notnull()]
index1 = data['A'] != 0     # 找到A列不为0
index2 = data['B'] != 0     # 找到B列不为0
index3 = (data['C'] != 0) & (data['D'] != 0)    # 找到C列和D列同时不为0

data = data[index1 & index2 & index3]

print(data.head(12))
