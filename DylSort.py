import numpy as np

from DylMath import *
from DylMerger import *
from tqdm import tqdm

def swap(arr: list, indexA: int, indexB: int, sizes: list=None):
    """swaps in arr either the indecies indexA and indexB, or the slices of the array as defined by 'sizes'"""
    if sizes == None: # swap elements
        temp = arr[indexA]
        arr[indexA] = arr[indexB]
        arr[indexB] = temp
    else: # swap groups
        start = sum(sizes[:indexB])
        toSwap = arr[start:start + sizes[indexB]]
        start = sum(sizes[:indexA])
        temp = arr[start:start + sizes[indexA]]
        start = sum(sizes[:indexA])
        for index in range(start, sizes[indexB]):
            arr[start + index] = toSwap[index]
        start = sum(sizes[:indexB])
        for index in range(start, sizes[indexA]):
            arr[start + index] = temp[index]

def genD0D1(d0d1: list, arr: list) -> tuple:
    D0, D1 = list(), list()
    
    for item in arr:
        if item in d0d1[0]:
            D0.append(item)
        elif item in d0d1[1]:
            D1.append(item)
    return D0, D1

def mergeSort(arr: list, comp=None, shuffle: bool=False, retStats: bool=False, retMid: bool=False, n: int=2, insSort: int=1, d0d1 = None) -> list:
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
    sizes: list = [insSort for i in range(len(arr) // insSort)]
    if insSort > 1:
        start = 0
        sorters = []
        while start < len(arr):
            sorters.append(insertion_sort(arr[start:start + insSort], comp, start))
            start += insSort

        while sorters:
            for sorter in sorters:
                res = next(sorter)
                if isinstance(res, (list, tuple)):
                    start, l = res
                    for i, val in enumerate(l, start=start):
                        arr[i] = val
                    sorters.remove(sorter)
    mergers = []

    if retMid:
        mini = min(arr)
        maxi = max(arr)
        split = (maxi - mini) // 2
        medians = []
        percentages = []

    # while there are partitions
    while sizes[0] < len(arr):
        # i is the partition number
        i: int = 0
        # start is the index the partition starts at
        start: int = 0
        # for each of the partitions
        s = 0
        groups = []
        for ds in sizes:
            groups.append(arr[s:s+ds])
            s += ds
        for i, size in enumerate(sizes):
            #last group, odd one out
            if i + 1 >= len(sizes):
                break
            if shuffle and sizes[i] > 4 and comp(groups[i][-1],groups[i + 1][0]):
                #if the next group comes before the current group:
                comp.learn([groups[i + 1] + groups[i]])
                swap(arr, i, i+1, sizes)
                sizes[i] += sizes[i + 1]
                sizes.pop(i + 1)
            elif shuffle and sizes[i] > 4 and i > 0 and comp(groups[i + 1][-1], groups[i][0]):
                #if the next group comes after this group
                comp.learn([groups[i] + groups[i + 1]])
                sizes[i] += sizes[i + 1]
                sizes.pop(i + 1)
            else:
                # get n arrays
                # feed the MultiMergers with them
                pos = start
                segments = min(n, len(sizes) - i)
                arrays = [0 for _ in range(segments)]
                for arrNumber in range(segments):
                    arrays[arrNumber] = arr[start:start + sizes[i + arrNumber]]
                    start += sizes[i + arrNumber]
                mergers.append(MultiMerger(arrays, comp, pos, 0))
                for _ in range(segments - 1):
                    # merge the sizes
                    sizes[i] += sizes[i + 1]
                    sizes.pop(i + 1)

        
        #while we have active mergers
        while mergers:
            for merger in mergers:
                res = merger.inc()
                if res: #if that merger is done
                    #print(merger.output)
                    start = merger.start
                    if -1 in merger.output:
                        raise FloatingPointError("it didn't actually do it")
                    comp.learn(merger.output)
                    for i, v in enumerate(merger.output):
                        if merger.output.count(v) > 1:
                            raise EnvironmentError(f"duplicated {v}")
                        arr[start + i] = v
                    mergers.remove(merger)

        # run dem stats
        if retStats:
            aucs, vars, hanleyMcNeils = list(), list(), list()
            start = 0
            for size in sizes:
                curr = arr[start:size + start]
                if d0d1 != None:
                    D0, D1 = genD0D1(d0d1, curr)
                else:
                    D0, D1 = list(sorted(curr))[:len(curr)//2], list(sorted(curr))[len(curr)//2:]
                sm = successMatrix(curr, D0, D1)
                auc = aucSM(sm)
                aucs.append(auc)
                hanleyMcNeils.append(hanleyMcNeil(auc, len(D0), len(D1)))
                vars.append(unbiasedMeanMatrixVar(sm))
                start += size
            npvar = np.var(aucs, ddof=1) / len(aucs)
            stats = [sum(aucs) / len(sizes), sum(vars) / (len(vars)**2), float(npvar), sum(hanleyMcNeils) / (len(hanleyMcNeils)**2)]
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

#https://rosettacode.org/wiki/Sorting_algorithms/Insertion_sort#Python
def insertion_sort(l, comp, start):
    for i in range(1, len(l)):
        j = i-1 
        key = l[i]
        while comp(key,l[j]) and (j >= 0):
           l[j+1] = l[j]
           j -= 1
           yield False
        yield False
        l[j+1] = key
    yield (start, l)

if __name__ == "__main__":
    from DylRand import *

    test = 11
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
            print(level, len(comp), comp.dupHistory, min(comp.minSeps.values()), comp.optHistory)
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

        data = continuousScale(16)
        comp = Comparator(data, level=0, rand=False)

        for arr in mergeSort(data, comp=comp, insSort=4):
            print(arr)
        print(comp.compHistory)
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
            for _ in mergeSort(data, comp, insSort=insSort, shuffle=True):
                print("done a layer")
            print(insSort, len(comp)) """
        
        data = nearlySorted(256, 10)
        for shuffle in (True, False):
            comp = Comparator(data, rand=True, level=0)
            arr = data[:]
            for _ in mergeSort(arr, comp, shuffle=shuffle):
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
            for _ in mergeSort(cont,comp, shuffle=shuffle):
                pass
            meCont = len(comp)

            comp.clearHistory()
            comp = Comparator(near, level=level, rand=True, seed=seed)

            images = [Image(id, comp) for id in near]
            images.sort()
            timNear = len(comp)

            comp = Comparator(near, level=level, rand=True, seed=seed)
            comp.clearHistory()
            for _ in mergeSort(near,comp, shuffle=shuffle):
                pass
            meNear = len(comp)

            comp = Comparator(near, level=level, rand=True, seed=seed)

            images = [Image(id, comp) for id in range(256)]
            images.sort()
            timFull = len(comp)

            data = list(range(256))

            comp = Comparator(data, level=level, rand=True, seed=seed)
            comp.clearHistory()
            for _ in mergeSort(data,comp, shuffle=shuffle):
                pass
            meFull = len(comp)

            print(level, timCont, meCont, timNear, meNear, timFull, meFull)
            