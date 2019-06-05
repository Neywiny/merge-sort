import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from random import random
from math import tanh

from nearlySorted import nearlySorted

def bLen(*args):
    """bLen(*args) -> len(args)
    a better version of len, which takes args because in python3 everything is a generator"""
    return len(args)

def genROC(threshold:int, predicted:tuple):
    """genROC(threshold:int, predicted:tuple) -> FPF, TPF
    generates FPF and TPF numbers for a given threshold"""
    #predicted = tuple(map(lambda x: int(x > threshold), predicted))
    actual:tuple = tuple(int(i > threshold) for i in range(length))
    # generate sensitivity and specificity by totalling TP and TN, dividing by actually P and actually N
    
    nTrue = bLen(*filter(lambda x: x == 1, actual))
    nFalse = bLen(*filter(lambda x: x == 0, actual))
    if nTrue == 0:
        return 1,1
    if nFalse == 0:
        return 0,0
    TPF = sensitivity = sum([actual[i] and predicted[i] for i in range(len(actual))]) / nTrue
    TNF = specificity = sum([not (actual[i] or predicted[i]) for i in range(len(actual))]) / nFalse
    # accuracy = [No. correct decisions]/[No. cases]
    accuracy:float = sum([actual[i] == predicted[i] for i in range(len(actual))]) / len(actual)
    # generate the other 2 3 letter numbers
    FPF:float = sum([predicted[i] and not actual[i] for i in range(len(actual))]) / nFalse
    FNF:float = sum([actual[i] and not predicted[i] for i in range(len(actual))]) / nTrue
    # validate that the numbers all add up
    if (TPF + FNF != 1) or (TNF + FPF != 1):
        raise ArithmeticError("TPF, FNF, TNF, or FPF are wrong")
    
    #print(actual, predicted, TPF, FPF)
    return FPF, TPF

if __name__ == "__main__":
    length:int = 100
    predicted:list = [random() <  1+tanh(1.5*(i/length - 1)) for i in range(length)]
    #predicted = [i < 50 for i in range(length)]
    print(predicted)
    line = []
    for threshold in range(-1, length):  
        line.append(genROC(threshold, predicted))
    plt.plot((0,1),(0,1),c="r")
    plt.plot(*zip(*line),'bo:',markersize=2)
    plt.ylim(top=1.1,bottom=-0.1)
    plt.xlim(left=-0.1,right=1.1)
    plt.show()