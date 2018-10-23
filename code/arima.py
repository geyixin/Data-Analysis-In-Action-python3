#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox as al
from statsmodels.tsa.arima_model import ARIMA

inputpath = '../Data/arima_data.xls'
data = pd.read_excel(inputpath, index_col='日期')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

'''
时序图
'''
# data.plot()
# plt.show()

'''
自相关图
'''
# plot_acf(data)
# plt.show()

'''
平稳性检测
ADF(data)后依次返回：
adf, pvalue, usedlag, nobs, critical values, icbest, regresults, restore
'''
print('原始序列的ADF值为：', ADF(data['销量']))

'''
差分
'''
D_data = data.diff().dropna()   # dropna()：滤除缺失值
# print(D_data.head(3))
D_data.columns = ['销量差分']   # 换列名
# print(D_data.head(3))

'''
# 差分后的时序图
'''
# D_data.plot()
# plt.title(u"差分后的时序图")
# plt.show()

'''
差分后的自相关图
'''
# plot_acf(D_data)
# plt.show()
'''
差分后的偏自相关图
'''
# plot_pacf(D_data)
# plt.show()
'''
差分后的平稳新检验
'''
print('差分后的ADF：', ADF(D_data['销量差分']))

'''
差分后的白噪声检验
'''
print('差分后的白噪声检验：', al(D_data, lags=1))


'''
发现一阶差分后的数据D_data的
时序图在一定值附近浮动，不再像Data那样递增；
自相关有很强的短期相关性；
PDF后的pvalue（就是p值）为0.022左右，小于0.05；
因此，一阶差分之后的序列是平稳序列。
不再需要继续差分了，下面定阶即可。
'''

# print(data.head(3))
data['销量'] = data['销量'].astype(float)
# print(data.head(3))

p_max = int(len(D_data)/10)   # 一般不超过len/10
q_max = int(len(D_data)/10)   # 一般不超过len/10
# print(p_max)
bic_matrix = []

for p in range(p_max + 1):
    temp = []
    for q in range(q_max + 1):
        try:
            temp.append(ARIMA(data, (p,1,q)).fit().bic)
        except:
            temp.append(None)
    bic_matrix.append(temp)

# print(bic_matrix)
bic_df = pd.DataFrame(bic_matrix)
# print(bic_df)

p, q = bic_df.stack().idxmin()  # 先用stack展平，再用idxmin找出最小值位置
print("BIC中p和q分别为: {p}、{q}".format(p=p, q=q))

model = ARIMA(data, (p,1,q)).fit()    # 建立模型

print('输出模型报告：','\n',model.summary2())
print('输出预测5的结果：','\n',model.forecast(5))   # 预测值、标准误差、置信区间
# print(model.summary.tables[1])






















































