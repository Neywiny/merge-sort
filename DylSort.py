import numpy as np

from DylMath import *
from DylMerger import *
from tqdm import tqdm

def swap(arr: list, indexA: int, indexB):
    """swaps in arr either the indecies indexA and indexB"""
    if sizes == None: # swap elements
        temp = arr[indexA]
        arr[indexA] = arr[indexB]
        arr[indexB] = temp

def genD0D1(d0d1: list, arr: list) -> tuple:
    D0, D1 = list(), list()
    
    for item in arr:
        if item in d0d1[0]:
            D0.append(item)
        elif item in d0d1[1]:
            D1.append(item)
    return D0, D1

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
        print(*(len(group) for group in groups))
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
            if currLayer == 0:
                varEstimate = float(varOfAverageAUC)
            elif currLayer == nLayers:
                varEstimate = (sum(varOfSM) / (len(varOfSM)**2))
            else:
                varEstimate = (currLayer*(sum(varOfSM) / (len(varOfSM)**2)) + (nLayers - currLayer) * float(varOfAverageAUC)) / nLayers
            stats = [avgAUC, varEstimate, sum(hanleyMcNeils) / len(hanleyMcNeils)**2, *estimates]
            #stats = [aucs, vars, float(npvar)]
            yield arr, stats
        elif not retMid:
            yield arr
        else:
            yield percentages, medians

#https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
def oddeven_merge_sort(length):
    """ "length" is the length of the list to be sorted.
    Returns a list of pairs of indices starting with 0 """
    def oddeven_merge_sort_range(lo, hi):
        """ sort the part of x with indices between lo and hi.

        Note: endpoints (lo and hi) are included.
        """
        def oddeven_merge(lo, hi, r):
            step = r * 2
            if step < hi - lo:
                yield from oddeven_merge(lo, hi, step)
                yield from oddeven_merge(lo + r, hi, step)
                yield from [(i, i + r) for i in range(lo + r, hi - r, step)]
            else:
                yield (lo, lo + r)

        if (hi - lo) >= 1:
            # if there is more than one element, split the input
            # down the middle and first sort the first and second
            # half, followed by merging them.
            mid = lo + ((hi - lo) // 2)
            yield from oddeven_merge_sort_range(lo, mid)
            yield from oddeven_merge_sort_range(mid + 1, hi)
            yield from oddeven_merge(lo, hi, 1)
    yield from oddeven_merge_sort_range(0, length - 1)

def compare_and_swap(comp, x, a, b):
    if comp(x[b], x[a]):
        x[a], x[b] = x[b], x[a]

def bitonic_sort(up, x, comp):
    def bitonic_merge(up, x, comp): 
        def bitonic_compare(up, x, comp):
            dist = len(x) // 2
            for i in range(dist):  
                if comp(x[i + dist], x[i]) == up:
                    x[i], x[i + dist] = x[i + dist], x[i] #swap
        # assume input x is bitonic, and sorted list is returned 
        if len(x) == 1:
            return x
        else:
            bitonic_compare(up, x, comp)
            first = bitonic_merge(up, x[:len(x) // 2], comp)
            second = bitonic_merge(up, x[len(x) // 2:], comp)
            if up:
                comp.learn(first + second)
            else:
                comp.learn(list(reversed(first + second)))
            return first + second
    if len(x) <= 1:
        return x
    else: 
        first = bitonic_sort(True, x[:len(x) // 2], comp)
        second = bitonic_sort(False, x[len(x) // 2:], comp)
        return bitonic_merge(up, first + second, comp)

if __name__ == "__main__":
    from DylRand import *

    test = 10
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
        from random import shuffle
        data = [i for i in range(256)]
        pairs_to_compare = list(oddeven_merge_sort(len(data)))
        for i in range(10):
            shuffle(data)
            comp = Comparator(data, level=3, rand=False)
            for i in pairs_to_compare: compare_and_swap(comp, data, *i)
            if data != sorted(data):
                print("woops", data, sorted(data))
            print(len(comp))
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
            