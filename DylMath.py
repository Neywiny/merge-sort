import ROC1
import numpy as np
np.set_printoptions(threshold=np.inf)
np.seterr(all="ignore")
from random import random
from math import tanh, sqrt
from tqdm import trange, tqdm
from multiprocessing import Pool
#from p_tqdm import p_map
try:
    import alweifubwaef
    import matplotlib
    matplotlib.use('QT4Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
except BaseException as e:
    pass

from DylRand import nearlySorted
from DylUtils import *
from DylData import *

unbiasedMeanMatrixVar = ROC1.unbiasedMeanMatrixVar
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

def var(arr: list, npc=None) -> float:
    """var(arr) -> binomial variance of the array"""
    if npc == None:
        npc = pc(arr)
    return npc*(1-npc)/len(arr)

def auc(results: tuple) -> float:
    if type(results[0]) != tuple and type(results[0]) != list:
        results = genROC(results)
    total: float = 0.0
    for i,(x,y) in enumerate(results[:-1]):
        total += y * (x - results[i + 1][0])
    return total

def aucSM(sm) -> float:
    return np.mean(sm)

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
    if D1 ==  None:
        D1 =  tuple((i for i in range(length//2, length)))
    points: list = [(1,1)]
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

def graphROCs(arrays: list, withPatches=False, withLine=True):
    rows = int(sqrt(len(arrays)))
    cols = len(arrays) // rows
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, num="plots")
    fig.suptitle("ROC curves")
    
    if withLine:
        if len(arrays[0]) < 1024:
            results = list(map(genROC, arrays))
        else:
            with Pool(processes=8) as p:
                results = list(tqdm(p.imap(genROC,arrays), total=len(arrays)))
    if withPatches:
        pbar = tqdm(total=len(arrays)*(len(arrays[0])//2)**2)
    for i,ax in enumerate(axes.flat):
        ax.set(xlabel="False Positive Fraction", ylabel="True Positive Fraction")
        ax.label_outer()
        ax.plot((0,1),(0,1),c='red', linestyle=":")
        if withLine:
            ax.plot(*zip(*results[i]), c='blue')
            ax.set_ylim(top=1.02, bottom=0)
            ax.set_xlim(left=-0.01, right=1)
            if not withPatches:
                ax.set_title(f"Iteration #{i} AUC: {auc(results[i]):.2f}")
        if withPatches:
            sm = successMatrix(arrays[i])
            yes = []
            no = []
            for (y,x), value in np.ndenumerate(sm):
                if value:
                    yes.append(Rectangle((x/(len(arrays[0])//2),y/(len(arrays[0])//2)),1/(len(arrays[0])//2),1/(len(arrays[0])//2)))
                else:
                    no.append(Rectangle((x/(len(arrays[0])//2),y/(len(arrays[0])//2)),1/(len(arrays[0])//2),1/(len(arrays[0])//2)))
                pbar.update(1)
            patches = PatchCollection(no, facecolor = 'r', alpha=0.75, edgecolor='None')
            ax.add_collection(patches)
            patches = PatchCollection(yes, facecolor = 'g', alpha=0.75, edgecolor='None')
            ax.add_collection(patches)
            area: float = 0.0
            for box in yes:
                area += box.get_height()*box.get_width()
            ax.set_ylim(top=1, bottom=0)
            ax.set_xlim(left=0, right=1)
            #ax.set_title(f"Iteration #{i} PC: {int(len(yes)/(len(yes)+len(no)) * 100)}% AUC: {area:.2f}")
            ax.set_title(f"Iteration #{i} PC: {int(pc(arrays[i])*100)}% AUC: {area:.2f}")
    if withPatches:
        pbar.close()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    plt.show()

def successMatrix(predicted: list, D0: list=None, D1: list=None):
    if D0 == None:
        D0 = [i for i in range(len(predicted) // 2)]
    if D1 == None:
        D1 = [i for i in range(len(predicted) // 2, len(predicted))]

    arr = np.full((len(D1), len(D0)), -1)
    for col, x in enumerate(reversed(D0)):
        for row, y in enumerate(D1):
            arr[row, col] = int(predicted.index(y) > predicted.index(x))
    return arr

if __name__ == "__main__":
    from DylSort import mergeSort
    test = 3
    if test == 1:
        #print(D0, D1) 
        newData, D0, D1 = continuousScale("sampledata.csv")
        print(auc(genROC(newData)))
        arrays = [newData[:]]
        for _ in mergeSort(newData):
            arrays.append(newData[:])
        print(arrays)
        graphROCs(arrays)
    elif test == 2:
        predicted = [0, 1, 5, 2, 3, 6, 4, 7, 8, 9]
        mat = successMatrix(predicted)
        print(mat)
        graphROC(predicted)
    elif test == 3:
        predicted = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        print(aucSM(successMatrix(predicted, [*range(10)], [*range(10,20)])))