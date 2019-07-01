import ROC1
import sys
import numpy as np
import math
np.set_printoptions(threshold=np.inf)
np.seterr(all="ignore")
from random import random
from tqdm import trange, tqdm
from multiprocessing import Pool
from scipy.interpolate import interp1d
#from p_tqdm import p_map
try:
    import matplotlib
    matplotlib.use('QT4Agg')
    import matplotlib.pyplot as plt
    font = {'size' : 56}
    #matplotlib.rc('font', **font)
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

def auc(results: tuple, D0=None, D1=None) -> float:
    if not isinstance(results[0], (list, tuple)):
        results = genROC(results, D0, D1)
    total: float = 0.0
    for i,(x,y) in enumerate(results[:-1], start=1):
        total += 0.5*(y + results[i][1]) * (x - results[i][0])
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
    if D0 == None:
        D0 = tuple((i for i in range(length//2)))
    if D1 == None:
        D1 =  tuple((i for i in range(length//2, length + (length % 2))))
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
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.plot(*zip(*genROC(predicted, D0, D1)))
    ax.plot((0,1),(0,1),c="r", linestyle="--")
    ax.set_ylim(top=1.1,bottom=-0.1)
    ax.set_xlim(left=-0.1,right=1.1)
    ax.set_title(f"AUC: {auc(predicted, D0, D1):.5f}")
    ax.set_xlabel("FPF")
    ax.set_ylabel("TPF") 
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

def avROC(rocs):
    #hard coded SeSp
    #e = 9*sys.float_info.epsilon

    # convert [(x1, y1), (x2, y2) ...] into np array for better arithmatic
    rocs = [np.array(roc) for roc in rocs]


    rotrocs = [{'u': tuple((roc[:,0] + roc[:,1])/2), 'v': tuple((roc[:,1]-roc[:,0])/2)} for roc in rocs]
    
    stdA = np.array(sorted(set((roc['u'] for roc in rotrocs))))

    aprotrocs = list()
    for roc in rotrocs:
        inter = interp1d(roc['u'], roc['v'])
        for x in stdA:
            aprotrocs.append(inter(x))

    ymean = np.zeros((1, len(stdA[0])))
    for apro in aprotrocs:
        ymean += apro
    ymean /= len(aprotrocs)

    fpout = stdA - ymean
    tpout = stdA + ymean

    ret = tuple(zip(fpout[0].tolist(), tpout[0].tolist()))
    return ret

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
    test = 6
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
    elif test == 5:
        graphROC([4, 1, 2, 3], [1, 2], [3, 4])
    elif test == 6:
        from DylSort import mergeSort
        from DylComp import Comparator
        from DylData import continuousScale
        
        data, D0, D1 = continuousScale(128, 128)
        comp = Comparator(data, rand=True, level=0, seed=15)

        arrays = [data[:]]

        for _ in mergeSort(data, comp=comp):
            arrays.append(data[:])

        graphROC(arrays[-1], D0=D0, D1=D1)

        data = arrays[-2]

        chunk1 = data[:len(data) // 2]
        chunk2 = data[len(data) // 2:]
        D01 = list(filter(lambda img: img in chunk1, D0))
        D11 = list(filter(lambda img: img in chunk1, D1))
        D02 = list(filter(lambda img: img in chunk2, D0))
        D12 = list(filter(lambda img: img in chunk2, D1))
        rocs = [genROC(chunk1, D0=D01, D1=D11), genROC(chunk2, D0=D02, D1=D12)]
        roc1 = genROC(chunk1, D01, D11)
        roc2 = genROC(chunk2, D02, D12)
        roc = avROC(rocs)
        print(auc(roc1, D01, D11))
        print(auc(roc2, D02, D12))
        print(auc(roc, D0, D1))
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.plot(*zip(*roc), 'b', label='avg')
        ax.plot(*zip(*roc1), 'm', label='chunk1')
        ax.plot(*zip(*roc2), 'g', label='chunk2')
        ax.plot((0,1),(0,1),c="r", linestyle="--")
        ax.set_ylim(top=1.1,bottom=-0.1)
        ax.set_xlim(left=-0.1,right=1.1)
        ax.set_xlabel("FPF")
        ax.set_ylabel("TPF") 
        ax.legend()
        plt.show()
