#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier as DTC

input = '../Data/sales_data.xls'
# da = pd.read_excel(input)
# print(da.head(3))
data = pd.read_excel(input, index_col='序号')
# print(data.head(3))
data[data == '好'] = 1
data[data == '是'] = 1
data[data == '高'] = 1
data[data != 1] = -1

x = data.iloc[:,:3].as_matrix().astype(int)
y = data.iloc[:,3].as_matrix().astype(int)

dtc = DTC()
dtc.fit(x, y)

# print(x[:3])
x = pd.DataFrame(x)
# print(x.head(3))
# print(x.columns)

with open('../Result/tree.dot', 'w') as f:
    f = export_graphviz(dtc, feature_names=x.columns, out_file=f)