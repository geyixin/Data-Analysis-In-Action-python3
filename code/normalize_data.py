#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

"""
数据规范

三种常用方法：
1）最小-最大规范化 ：映射到[0,1]
        x-min
      ---------
       max-min
    x：原始数据，min：原始数据最小值，max：原始数据最大值
2）零均值规范化 : 处理后数据均值为0，标准差为1
        x-X
      --------
         B
    x：原始数据，X：原始数据均值，B：原始数据标准差
3）小数定标规范化 : 映射到[-1,1]
        x
     -------
      10**k   
    k就是小数点移动的位数，由数据中绝对值的最大值决定
"""

import pandas as pd
import numpy as np

input = '../Data/normalization_data.xls'
data = pd.read_excel(input, header=None)    # 不加header=None，会默认将第一行作为列的标题

print(data)

max_min = (data - data.min())/(data.max() - data.min())

zero_average = (data - data.mean())/data.std()

small_data = data/10**np.ceil(np.log10(data.abs().max()))   # ceil:向上取整

'''
注意：上面的min()、max()、data.abs().max())  等都是对每一列操作的
'''