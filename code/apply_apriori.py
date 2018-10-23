#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
from apriori import *

inputpath = '../Data/menu_orders.xls'
ouputpath = '../DataOut/menu_order_fixed.xls'

data = pd.read_excel(inputpath, header=None)

# print(data)

ct = lambda x : pd.Series(1, index=x[pd.notnull(x)])

b = map(ct, data.as_matrix())

data = pd.DataFrame(list(b)).fillna(0)

# print(data)

del b

support = 0.2
confidence = 0.5
ms = '------>'

find_rule(data, support, confidence, ms).to_excel(ouputpath)