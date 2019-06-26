import ROC1
import numpy as np
np.set_printoptions(threshold=np.inf)
np.seterr(all="ignore")
from random import random
import math
from tqdm import trange, tqdm
from multiprocessing import Pool
#from p_tqdm import p_map
try:
    import matplotlib
    matplotlib.use('QT4Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
except BaseException as e:
    pass

from DylRand import nearlySorted
from DylData import *

unbiasedMeanMatrixVar = ROC1.unbiasedMeanMatrixVar

def paramToParams(predicted, D0=None, D1=None):
    if isinstance(predicted[0], (list, tuple)):
        return predicted[0], predicted[1], predicted[2]
    else:
        return predicted, D0, D1

def stdev(inp: list) -> float:
    """std(inp) -> standard deviation of the input
    inp can be a list or the variance of that list"""
    return math.sqrt(var(inp)) if isinstance(inp, (list, tuple)) else math.sqrt(inp)

def se(inp: list, n=None) -> float:
    """se(inp) -> standard error of the input
    inp can be a list or the stdev of the list, in which case
    n needs to be provided"""
    return stdev(inp) / math.sqrt(len(n) if n != None else len(inp))

def pc(arr: list, D0: list, D1: list) -> float:
    # calc % correct
    # add up all the times a number is on the correct side
    # divide by total to get the average
    pc: float = 0.0
    for i,val in enumerate(arr):
        if val in D0:
            if i < len(D0):
                pc += 1
        elif val in D1:
            if i > len(D0):
                pc += 1
    return pc / len(arr)

def var(arr: list, npc=None) -> float:
    """var(arr) -> binomial variance of the array"""
    if npc == None:
        npc = pc(arr)
    return npc*(1-npc)/len(arr)

def auc(results: tuple) -> float:
    if not isinstance(results[0], (list, tuple)):
        results = genROC(results)
    total: float = 0.0
    for i,(x,y) in enumerate(results[:-1], start=1):
        total += y * (x - results[i][0])
    return total

def hanleyMcNeil(auc, n0, n1):
    # The very good power-law variance estimate from Hanley/McNeil
    auc2=auc*auc
    q1=auc/(2.-auc)
    q2=2.*auc2/(1.+auc)
    return( (auc-auc2+(n1-1.)*(q1-auc2)+(n0-1.)*(q2-auc2))/n0/n1 )

def aucSM(sm) -> float:
    return np.mean(sm)

def genROC(predicted: tuple, D0: tuple=None, D1: tuple=None) -> tuple: 
    predicted, D0, D1 = paramToParams(predicted, D0, D1)

    length: int = len(predicted)
    actual: tuple = tuple(int(i > length/2 - 1) for i in range(length))
    points: list = [(1,1)]
    FPcount: int = len(D0)
    TPcount: int = len(D1)
    for threshold in reversed(range(length)):
        if predicted[threshold] in D0: # false positive
            FPcount -= 1
        if predicted[threshold] in D1: # true positive
            TPcount -= 1
        TPF: float = TPcount / len(D1)
        FPF: float = FPcount / len(D0)
        points.append((TPF, FPF))
    points.append((0,0))
    return points


def graphROC(predicted: tuple, D0=None, D1=None):
    predicted, D0, D1 = paramToParams(predicted, D0, D1)
    plt.plot(*zip(*genROC(predicted, D0, D1)))
    plt.plot((0,1),(0,1),c="r", linestyle="--")
    plt.ylim(top=1.1,bottom=-0.1)
    plt.xlim(left=-0.1,right=1.1)
    plt.title(f"PC: {int(pc(predicted) * 100)}% AUC: {auc(predicted):.2f}")
    plt.gca().set(xlabel="False Positive Fraction", ylabel="True Positive Fraction") 
    plt.show()

def graphROCs(arrays: list, withPatches=False, withLine=True, D0=None, D1=None):
    rows = int(math.ceil(math.sqrt(len(arrays))))
    cols = int(math.ceil(len(arrays) / rows))
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, num="plots")
    fig.suptitle("ROC curves")
    
    if withLine:
        params = [(array, D0, D1) for array in arrays]
        if len(arrays[0]) < 1024:
            results = list(map(genROC, params))
        else:
            with Pool() as p:
                results = list(p.imap(genROC,params))
    if withPatches:
        pbar = tqdm(total=len(arrays)*(len(arrays[0])//2)**2)
    for i,ax in enumerate(axes.flat):
        if i >= len(arrays):
            continue
        ax.set(xlabel="False Positive Fraction", ylabel="True Positive Fraction")
        ax.label_outer()
        ax.plot((0,1),(0,1),c='red', linestyle=":")
        if withLine:
            ax.plot(*zip(*results[i]), c='blue')
            ax.set_ylim(top=1.02, bottom=0)
            ax.set_xlim(left=-0.01, right=1)
            if not withPatches:
                ax.set_title(f"Iteration #{i} AUC: {auc(results[i]):.5f}")
        if withPatches:
            sm = successMatrix(arrays[i], D0, D1)
            yes = []
            no = []
            length = len(arrays[0])//2
            yLen = len(D1)
            xLen = len(D0)
            for (y,x), value in np.ndenumerate(sm):
                if value:
                    yes.append(Rectangle((x/xLen,y/yLen),1/xLen,1/yLen))
                else:
                    no.append(Rectangle((x/xLen,y/yLen),1/xLen,1/yLen))
                pbar.update(1)
            patches = PatchCollection(no, facecolor = 'r', alpha=0.75, edgecolor='None')
            ax.add_collection(patches)
            patches = PatchCollection(yes, facecolor = 'g', alpha=0.75, edgecolor='None')
            ax.add_collection(patches)
            area = len(yes) / (len(yes) + len(no))
            ax.set_ylim(top=1, bottom=0)
            ax.set_xlim(left=0, right=1)
            ax.set_title(f"Iteration #{i} PC: {int(pc(arrays[i], D0, D1)*100)}% AUC: {area:.5f}")
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
        for row, y in enumerate(reversed(D1)):
            yInd, xInd = predicted.index(y), predicted.index(x)
            arr[row, col] = int(xInd < yInd)
    if -1 in arr:
        raise EnvironmentError("failed to create success matrix")
    return arr

if __name__ == "__main__":
    from DylSort import mergeSort
    test = 4
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
    elif test == 4:
        arrays = [[0, 1, 4, 2, 5, 3, 6], 
                  [0, 1, 2, 4, 3, 5, 6],
                  [0, 1, 2, 4, 3, 5, 6],
                  [0, 1, 2, 3, 4, 5, 6]]
        graphROCs(arrays, D0=[0, 1, 2, 3], D1=[4, 5, 6])
