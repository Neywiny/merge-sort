import numpy as np
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from random import random
from math import tanh, sqrt
from tqdm import trange, tqdm
from multiprocessing import Pool
from p_tqdm import p_map

from DylRand import nearlySorted
from DylUtils import *

def var(arr: list) -> float:
    """var(arr) -> binomial variance of the array"""
    return pc*(1-pc)/len(arr)

def stdev(inp: list) -> float:
    """std(inp) -> standard deviation of the input
    inp can be a list or the variance of that list"""
    return sqrt(var(inp)) if type(inp) == list else sqrt(inp)

def se(inp: list, n=None) -> float:
    """se(inp) -> standard error of the input
    inp can be a list or the stdev of the list, in which case
    n needs to be provided"""
    return stdev(inp) / sqrt(len(n) if n != None else len(inp))

def pc(arr: list) -> float:
    # calc % correct
    # add up all the times a number is on the correct side
    # divide by total to get the average
    return sum((arr[i] < len(arr)/2) == (i < len(arr)/2) for i in range(len(arr))) / (len(arr) - 1)

def auc(results: tuple) -> float:
    if type(results[0]) != tuple and type(results[0]) != list:
        results = genROC(results)
    total: float = 0.0
    for i,(x,y) in enumerate(results):
        try:
            total += y * (x - results[i + 1][0])
        except IndexError:
            pass
    return total

def genROC(predicted: tuple, D0: tuple=None, D1: tuple=None) -> tuple: 
    def genFPFTPF(threshold: int, predicted: tuple, D0: tuple, D1: tuple):
        FPcount: int = 0
        TPcount: int = 0
        for i in range(len(predicted)):
            if i >= threshold: # positive and should be negative
                if predicted[i] in D0: # false positive
                    FPcount += 1
                if predicted[i] in D1: # true positive
                    TPcount += 1
        TPF: float = TPcount / len(D1)
        FPF: float = FPcount / len(D0)
        #print(threshold, FPcount, TPcount, sep='\t')
        return FPF, TPF
    length: int = len(predicted)
    actual: tuple = tuple(int(i > length/2 - 1) for i in range(length))
    if D0 == None:
        D0 = tuple((i for i in range(length//2)))
        D1 =  tuple((i for i in range(length//2, length)))
    points = []
    for i in range(length):
        points.append(genFPFTPF(i, predicted, D0, D1))
    points.append((0,0))
    return points


def graphROC(predicted: tuple, D0=None, D1=None):
    plt.plot(*zip(*genROC(predicted, D0, D1)))
    plt.plot((0,1),(0,1),c="r", linestyle="--")
    plt.ylim(top=1.1,bottom=-0.1)
    plt.xlim(left=-0.1,right=1.1)
    plt.title(f"PC: {int(pc(predicted) * 100)}% AUC: {auc(predicted):.2f}")
    plt.gca().set(xlabel="False Positive Fraction", ylabel="True Positive Fraction") 
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
    
    #with Pool(processes=8) as p:
    #    results = list(tqdm(p.imap(genROC,arrays), total=len(arrays)))
    results = p_map(genROC, arrays)

    for i,ax in enumerate(axes.flat):
        ax.set(xlabel="False Positive Fraction", ylabel="True Positive Fraction")
        ax.label_outer()
        ax.plot(*zip(*results[i]), c='blue')
        ax.plot((0,1),(0,1),c='red', linestyle=":")
        ax.set_title(f"Iteration #{i} PC: {int(pc(arrays[i]) * 100)}% AUC: {auc(results[i]):.2f}")
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    plt.show()

if __name__ == "__main__":
    data = []
    D0 = []
    D1 = []
    with open("sampledata.csv") as f:
        for i,line in enumerate(f):
            if len(line) > 10:
                line = line.strip().split(" ")
                point = float(line[2])
                data.append(point)
                if line[1] == "1": # case should be positive
                    D1.append(i)
                else: # case should be negative
                    D0.append(i)
    newData = [-1 for i in range(len(data))]
    #print(data)
    for i, d in enumerate(sorted(data)):
        newData[i] = data.index(d)
    D0.sort()
    D1.sort()
    #print(D0, D1)
    print(auc(genROC(newData)))
    graphROC(newData, D0, D1)