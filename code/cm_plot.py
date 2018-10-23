#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def cm_plot(y, yp):
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    return plt