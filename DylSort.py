import numpy as np

from DylMath import *
from DylMerger import *
from tqdm import tqdm

def genD0D1(d0d1: list, arr: list) -> tuple:
    D0, D1 = list(), list()
    
    for item in arr:
        if item in d0d1[0]:
            D0.append(item)
        elif item in d0d1[1]:
            D1.append(item)
    return D0, D1

def runStats(groups, d0d1, n, currLayer, nLayers):
    aucs, varOfSM, hanleyMcNeils, estimates = list(), list(), list(), list()
    start = 0
    for group in groups:
        if d0d1 != None:
            D0, D1 = genD0D1(d0d1, group)
        else:
            D0, D1 = list(sorted(group))[:len(group)//2], list(sorted(group))[len(group)//2:]
        sm = successMatrix(group, D0, D1)
        auc = aucSM(sm)
        aucs.append(auc)
        hanleyMcNeils.append((len(D0), len(D1)))
        smVAR = unbiasedMeanMatrixVar(sm)
        if smVAR == smVAR: # if not NaN
            varOfSM.append(smVAR)
    varOfAverageAUC = np.var(aucs, ddof=1) / len(aucs)
    aucs = np.array(aucs)
    avgAUC = np.mean(aucs)

    estimateNs = [list()]
    for i, ns in enumerate(hanleyMcNeils):
        estimateNs[0].append(ns)

    # while there are groups to 'merge'
    while len(estimateNs[-1]) != 1:
        # get the previous layer and sort by N0 + N1
        oldNs = sorted(estimateNs[-1], key=lambda x: sum(x))
        # roughly the same code as mergers creation
        estimateNs.append(list())
        while oldNs:
            i = 0
            toMerge = list()
            segments = min(n, len(oldNs) - i)
            for iSegment in range(segments):
                toMerge.append(oldNs.pop(0))
            estimateNs[-1].append([sum((x[0] for x in toMerge)), sum((x[1] for x in toMerge))])
        estimateNs[-1].sort(key=lambda x:sum(x))
        estimates.append(hanleyMcNeil(avgAUC, estimateNs[-1][-1][0], estimateNs[-1][-1][1]) / len(estimateNs[-1]))


    for i, (N0, N1) in enumerate(hanleyMcNeils):
        hanleyMcNeils[i] = hanleyMcNeil(avgAUC, N0, N1)
    if len(varOfSM) == 0:
        varEstimate = float(varOfAverageAUC)
    elif currLayer == nLayers:
        varEstimate = (sum(varOfSM) / (len(varOfSM)**2))
    else:
        varEstimate = (currLayer*(sum(varOfSM) / (len(varOfSM)**2)) + (nLayers - currLayer) * float(varOfAverageAUC)) / nLayers

    # bootstimate
    iters = max(n**n, 1000)
    z = np.array([np.mean(aucs[np.random.randint(len(aucs), size=len(aucs))]) for i in range(iters)])
    z.sort()
    lowBoot = z[len(z) // 20]
    highBoot = z[len(z) - (len(z) // 20)]
    # arcsin transform
    lowSine = np.sin(np.sin(np.arcsin(np.sqrt(avgAUC)) - 1))
    highSine = np.sin(np.sin(np.arcsin(np.sqrt(avgAUC)) + 1))

    stats = [avgAUC, varEstimate, sum(hanleyMcNeils) / len(hanleyMcNeils)**2, lowBoot, highBoot, lowSine, highSine, *estimates]
    #stats = [aucs, vars, float(npvar)]
    return stats

def mergeSort(arr: list, comp=None, retStats: bool=False, retMid: bool=False, n: int=2, d0d1 = None) -> list:
    """mergeSort(arr: list, level=3)
    Can either be provided a comparator or will make its own
    merge sorts the list arr with 'level' amount of optimization
    yields the arr after each pass through
    also yields the stats used if retStats"""
    if comp == None:
        from DylComp import Comparator
        comp = Comparator(arr, level=3, rand=False)

    # do this after comp created just in case
    if not arr:
        yield arr, None if retStats else arr
        return

    groups: list = list([arr[i]] for i in range(len(arr)))

    mergers = []

    if retMid:
        mini = min(arr)
        maxi = max(arr)
        split = (maxi - mini) // 2
        medians = []
        percentages = []
    currLayer = -1
    nLayers = calcNLayers(arr) - 1
    # while there are partitions
    while len(groups) != 1:
        currLayer += 1
        i = 0
        # interleave big and small groups for merging
        groups.sort(key=lambda x: len(x))
        #smalls = groups[:len(groups) // 2]
        #bigs = list(reversed(groups[len(smalls):]))
        #groups = list()
        #smallsI = 0
        #bigsI = 0
        #ratio = len(bigs) / len(smalls)
        #while smallsI < len(smalls):
        #    groups.append(smalls[smallsI])
        #    smallsI += 1
        #    percent = smallsI * ratio
        #    while bigsI < percent:
        #        groups.append(bigs[bigsI])
        #        bigsI += 1

        while groups:
            #last group, odd one out
            if i + 1 >= len(groups):
                break
            # get n arrays
            # feed the MultiMergers with them
            segments = min(n, len(groups) - i)
            arrays = list()
            for iSegment in range(segments):
                arrays.append(groups.pop(0))
            mergers.append(MultiMerger(arrays, comp, i, 0))
        i += n
        #while we have active mergers
        while mergers:
            for merger in mergers:
                res = merger.inc()
                if res: #if that merger is done
                    #print(merger.output)
                    if -1 in merger.output:
                        raise FloatingPointError("it didn't actually do it")
                    for i, v in enumerate(merger.output):
                        if merger.output.count(v) > 1:
                            raise EnvironmentError(f"duplicated {v}")
                    comp.learn(merger.output)
                    groups.append(merger.output)
                    mergers.remove(merger)
        arr = []
        for group in groups:
            arr.extend(group)
        # run dem stats
        if retStats:
            stats = runStats(groups, d0d1, n, currLayer, nLayers)
            #stats = [aucs, vars, float(npvar)]
            yield arr, stats
        elif not retMid:
            yield arr
        else:
            yield percentages, medians

def treeMergeSort(arr: list, comp, n: int=2, retStats: bool=False, d0d1 = None):
    if n < 2:
        raise IOError("can't split a tree with n < 2")
    sizess = [[0, len(arr)]]
    while max(sizess[-1]) > n: #needs to be broken down further
        sizess.append([0])
        for i, size in enumerate(sizess[-2]):
            quotient, remainder = divmod(size, n)
            while size > 0:
                if remainder > 0:
                    sizess[-1].append(quotient + 1)
                    remainder -= 1
                    size -= quotient + 1
                else:
                    sizess[-1].append(quotient)
                    size -= quotient
    for sizes in sizess:
        for i, size in enumerate(sizes[1:], start=1):
            sizes[i] += sizes[i - 1]
    # do the first layer, which pulls from the array
    mergerss = [[], []]
    i = 0
    while i < len(sizess[-1]) - 1:
        segments = min(n, len(sizess[-1]) - i)
        groups = list()
        for iSeg in range(segments):
            group = arr[sizess[-1][i + iSeg]:sizess[-1][i + iSeg + 1]]
            if len(group) != 1: # such that we need to add another layer to it
                mergerss[0].append(MultiMerger([[img] for img in group], comp))
                groups.append(mergerss[0][-1].output)
            else:
                groups.append(group)
        mergerss[1].append(MultiMerger(groups, comp))
        i += segments
    # now build up layers of mergerss where the groups are the outputs of the last layer
    while len(mergerss[-1]) > 1: # while not on top level
        mergerss.append(list())
        i = 0
        while (segments == min(n, len(mergerss[-2]) - i)) != 0:
            groups = list()
            for iSeg in range(segments):
                groups.append(mergerss[-2][i + iSeg].output)
            mergerss[-1].append(MultiMerger(groups, comp))
            i += segments
    #print(mergerss)
    #for mergers in mergerss:
    #    print([[len(group) for group in merger.groups] for merger in mergers])
    left = True
    for layer, mergers in enumerate(mergerss, start=1):
        for merger in mergers if left else reversed(mergers):
            while not merger.inc():
                pass
        groups = list((merger.output for merger in mergers))
        left != left
        arr = []
        for group in groups: arr.extend(group)
        yield (arr, runStats(groups, d0d1, n, layer, len(mergerss))) if retStats else arr
    #print(f"n? {n} Did it sort right? {mergerss[-1][-1].output == sorted(mergerss[-1][-1].output)}. How many layers? {layer} How many comparisons? {len(comp)}")
if __name__ == "__main__":
    from DylRand import *

    test = 5
    if test == 1:
        from random import shuffle, seed
        from tqdm import trange
        from DylComp import Comparator
        #maxi = 79
        #if True:
        for maxi in trange(1000):
            seed(maxi)
            arr = list(range(maxi))#79 is prime
            shuffle(arr)
            sArr = sorted(arr)
            comp = Comparator(arr, level=0, rand=True)
            for _ in mergeSort(arr, comp=comp, retStats=False, retMid=True):
                pass
    elif test == 2:
        maxi = 1024
        l: list = randomDisease(maxi)
        for level in range(5):
            for arr, comp in mergeSort(l[:], level = level, retComp=True):
                pass
            print(level, len(comp), comp.dupHistory, min(comp.seps.values()), comp.optHistory)
    elif test == 3:
        m = Merger([0, 1, 2, 4, 5, 6, 7, 8, 9], [3], Comparator([0,1, 2, 3, 4, 5, 6, 7, 8, 9], level=3))
        while not m.inc():
            continue
            print(m.output)
        print(m.output)
        print(m.comp.compHistory)
    elif test == 4:
        data = continuousScale(128)
        arrays = [data[:]]
        print(data)
        for _ in mergeSort(data):
            arrays.append(data[:])
            print(data)
        graphROCs(arrays, True)
    elif test == 5:
        from DylComp import Comparator
        print("treeMergeSort")
        for n in range(2, 18):
            data = [*reversed(range(197))]
            comp = Comparator(data, level=0, rand=False)
            for _ in treeMergeSort(data, comp, n=n):
                pass
            print(n, len(comp))
        print("regular mergeSort")
        for n in range(2, 18):
            data = [*reversed(range(197))]
            comp = Comparator(data, level=0, rand=False)
            for _ in mergeSort(data, comp, n=n):
                pass
            print(n, len(comp))
    elif test == 6:
        from DylComp import Comparator
        from random import shuffle

        data = [i for i in range(256)]
        shuffle(data)
        comp = Comparator(data)
        res = bitonic_sort(True, data, comp)
        if res != sorted(data):
            print("woops")
        print(len(comp))
    elif test == 7:
        from DylData import continuousScale
        from DylComp import Comparator
        from tqdm import trange, tqdm

        avgs = []
        for plot in trange(100):
            data = continuousScale(256)
            comp = Comparator(data, level=0, rand=True)
            for _ in mergeSort(data, comp=comp, retMid=True):
                pass
            avgs.append(len(comp))
        print(sum(avgs) / 100)

        avgs = []
        for plot in trange(100):
            data = continuousScale(256)
            comp = Comparator(data, level=0, rand=True)
            for _ in mergeSort(data, comp=comp, retMid=False):
                pass
            avgs.append(len(comp))
        print(sum(avgs) / 100)
        avgs = []
        for plot in trange(5):
            data = continuousScale(256)
            comp = Comparator(data, level=3, rand=True)
            for _ in mergeSort(data, comp=comp, retMid=True):
                pass
            avgs.append(len(comp))
        print(sum(avgs) / 5)

        avgs = []
        for plot in trange(5):
            data = continuousScale(256)
            comp = Comparator(data, level=3, rand=True)
            for _ in mergeSort(data, comp=comp, retMid=False):
                pass
            avgs.append(len(comp))
        print(sum(avgs) / 5)
    elif test == 8:
        from DylComp import Comparator
        from DylData import continuousScale

        data = continuousScale(16)
        comp = Comparator(data, level=3, rand=True)
        print(data)
        for val in sorted(data):
            print(val, comp.getLatentScore(val))
        
        for l, (arr, stats) in enumerate(mergeSort(data, comp, retStats=True, retMid=False), start=1):
            print(l, list(map(lambda x: int(x >= 8), arr)), stats)
    elif test == 9:
        from DylComp import Comparator
        from DylData import continuousScale
        from tqdm import trange
        comp = Comparator(list(range(256)), level=3)
        for n in trange(2, 257):
            comp.clearHistory()
            arr = continuousScale(256)
            for _ in mergeSort(arr, comp=comp, n=n):
                pass
            assert arr == sorted(arr)
            print(n, len(comp), sep=',')
    elif test == 10:
        from DylComp import Comparator
        from DylData import continuousScale

        data = continuousScale(256)
        comp = Comparator(data, level=0, rand=True)
        print("AUC\tSMV\tNPV\tHmN")
        for arr in mergeSort(data, comp=comp, retStats=True):
            print(*[f"{x:0.5f}" for x in arr[1]], sep='\t')
    elif test == 11:
        from DylComp import Comparator
        from DylData import continuousScale

        data = continuousScale(256)
        arrays = [tuple(data)]
        print(data)
        comp = Comparator(data, level=0, rand=False)
        for _ in mergeSort(data, comp, n=4):
            print(data)
            arrays.append(tuple(data))
        print(data)
    elif test == 12:
        from DylComp import Comparator
        from DylRand import nearlySorted

        """ for insSort in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            data = nearlySorted(256, 10)
            comp = Comparator(data, rand=True, level=0)
            for _ in mergeSort(data, comp):
                print("done a layer")
            print(insSort, len(comp)) """
        
        data = nearlySorted(256, 10)
        for shuffle in (True, False):
            comp = Comparator(data, rand=True, level=0)
            arr = data[:]
            for _ in mergeSort(arr, comp):
                pass
            print(shuffle, len(comp), arr)
    elif test == 13:
        from DylData import continuousScale
        from DylComp import Comparator
        from DylRand import nearlySorted
        from functools import cmp_to_key
        from tqdm import trange

        class Image:
            def __init__(self, id, comp):
                self.id = id
                self.comp = comp
            def __lt__(self, other):
                return self.comp(self.id, other.id)

        shuffle = False

        #if True:
        for level in range(4):
            with open("/dev/urandom", 'rb') as file: 
                rand = [x for x in file.read(10)]
            seed = 1
            for val in rand: seed *= val
            seed %= 2**32
            near = nearlySorted(256, 40)
            cont = continuousScale(256)
            comp = Comparator(cont, level=level, rand=True, seed=seed)
            images = [Image(id, comp) for id in cont]
            images.sort()
            timCont = len(comp)

            comp.clearHistory()
            comp = Comparator(cont, level=level, rand=True, seed=seed)
            for _ in mergeSort(cont,comp):
                pass
            meCont = len(comp)

            comp.clearHistory()
            comp = Comparator(near, level=level, rand=True, seed=seed)

            images = [Image(id, comp) for id in near]
            images.sort()
            timNear = len(comp)

            comp = Comparator(near, level=level, rand=True, seed=seed)
            comp.clearHistory()
            for _ in mergeSort(near,comp):
                pass
            meNear = len(comp)

            comp = Comparator(near, level=level, rand=True, seed=seed)

            images = [Image(id, comp) for id in range(256)]
            images.sort()
            timFull = len(comp)

            data = list(range(256))

            comp = Comparator(data, level=level, rand=True, seed=seed)
            comp.clearHistory()
            for _ in mergeSort(data,comp):
                pass
            meFull = len(comp)

            print(level, timCont, meCont, timNear, meNear, timFull, meFull)
            