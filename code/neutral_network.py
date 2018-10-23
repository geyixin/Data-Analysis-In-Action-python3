#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


input = '../Data/sales_data.xls'
data = pd.read_excel(input, index_col='序号')

data[data == '好'] = 1
data[data == '是'] = 1
data[data == '高'] = 1
data[data != 1] = 0

x = data.iloc[:,:3].as_matrix().astype(int)
y = data.iloc[:,3].as_matrix().astype(int)

model = Sequential()
model.add(Dense(input_dim=3, units=10))
model.add(Activation('relu'))
model.add(Dense(input_dim=10, units=1))
model.add(Activation('sigmoid'))

# model.compile(optimizer='adam', loss='binary_crossentropy', class_metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=10)
yp = model.predict_classes(x).reshape(len(y))
# print(yp)
# print(y)

def cm_plot(y, yp):
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Greens)
    # plt.matshow(cm)

    plt.colorbar()

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    return plt

cm_plot(y, yp).savefig('../Result/neutral.png')