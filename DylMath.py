import numpy as np
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from random import random
from math import tanh
from tqdm import trange, tqdm
from multiprocessing import Pool

from DylRand import nearlySorted
from DylUtils import *


def calcPC(arr: list) -> float:
    # calc % correct
    # add up all the times a number is on the correct side
    # divide by total to get the average
    return sum((arr[i] < len(arr)/2) == (i < len(arr)/2) for i in range(len(arr))) / (len(arr) - 1)

def auc(results: tuple) -> float:
    total: float = 0.0
    for x,y in results:
        total += y * (1 / len(results))
    return total

def genROC(predicted: tuple) -> tuple: 
    def genFPFTPF(threshold: int, predicted: tuple, D0: tuple, D1: tuple):
        FPcount: int = 0
        TPcount: int = 0
        length: int = len(predicted)//2
        for i in range(length):
            if i >= threshold: # positive and should be negative
                if predicted[i] in D0: # false positive
                    FPcount += 1
                if predicted[i] in D1: # true positive
                    TPcount += 1
            if i + length >= threshold: # positive and should be positive
                if predicted[i + length] in D0: # false positive
                    FPcount += 1
                if predicted[i + length] in D1: # true positive
                    TPcount += 1
        TPF: float = TPcount / length
        FPF: float = FPcount / length
        #print(threshold, FPcount, TPcount, sep='\t')
        return FPF, TPF
    length: int = len(predicted)
    actual: tuple = tuple(int(i > length/2 - 1) for i in range(length))
    D0 = tuple((i for i in range(length//2)))
    D1 =  tuple((i for i in range(length//2, length)))
    points = []
    for i in range(length):#//2 + 2):
        points.append(genFPFTPF(i, predicted, D0, D1))
    points.append((0,0))
    return points


def graphROC(predicted: tuple, ax=None):
    if ax == None:
        ax = plt
    ax.plot(*zip(*genROC(predicted)))
    ax.plot((0,1),(0,1),c="r")
    if ax == plt:
        ax.ylim(top=1.1,bottom=-0.1)
        ax.xlim(left=-0.1,right=1.1)
        ax = plt.gca()
        ax.set(xlabel="False Positive Fraction", ylabel="True Positive Fraction") 
        plt.show()

def graphROCs(arrays: list):
    rows = 1
    if not (len(arrays) % 2):
        rows = 2
    if not (len(arrays) % 3):
        rows = 3
    if not (len(arrays) % 4):
        rows = 4
    cols = len(arrays) // rows
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, num="plots")
    fig.suptitle("ROC curves")
    
    print("pooling")
    with Pool(5) as p:
        print("pooled")
        results = p.map(genROC,arrays)
        print("resulted")
    for i,ax in enumerate(axes.flat):
        ax.set(xlabel="False Positive Fraction", ylabel="True Positive Fraction")
        ax.label_outer()
        ax.plot(*zip(*results[i]), c='blue')
        ax.plot((0,1),(0,1),c='red')
        ax.set_title(f"Iteration #{i} PC: {int(calcPC(arrays[i]) * 100)}% AUC: {auc(results[i])}")
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    plt.show()

if __name__ == "__main__":
    print(calcPC(predicted))
    length: int = 100
    data: list = list(range(100))
    predicted: tuple = [data.pop(0) if (random() >  1+tanh(1.5*(i/length - 1))) else data.pop(len(data) - 1) for i in range(length)]
    #print(predicted)
    #graphROC(predicted)