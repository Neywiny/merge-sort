import numpy as np

from DylMath import *

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

class Merger:
    """A class for merging 2 arrays into 1 array
    Must feed it a comparator object. It can also hold onto start/stop indecies if you ask
    'Toggle' parameter defines if you want to do a cocktail merge, as in switch which end you're merging from each call of inc()
    """
    def __init__(self, groupA: list, groupB: list, comp, start=0, stop=0, LL=None, LR=None, RL=None, RR=None, toggle:bool=True):
        self.groupA: list = groupA
        self.groupB: list = groupB
        comp.learn(groupA)
        comp.learn(groupB)
        self.output: list = [-1 for i in [*groupA, *groupB]]
        self.indexA: int = 0
        self.indexB: int = 0
        self.outIndex: int = 0
        self.indexARight: int = len(groupA) - 1
        self.indexBRight: int = len(groupB) - 1
        self.indexORight: int = len(self.output) - 1
        self.comp = comp
        self.left: bool = True
        self.toggle: bool = toggle

        #hold onto these for the parent object, they're a secret tool that will help us later
        self.start = start
        self.stop = stop

        self.passedA = groupA[:]
        self.passedB = groupB[:]

        if LL == None:
            return
        if LL > 0 and LR > 0 and RL > 0 and RR > 0:
            self.vals = {'RR':RR, 'LR':LR, 'LL': LL, 'RL': RL}

            if LL + LR == len(groupA):
                #put the groupB in the middle of groupA
                for i, val in enumerate(groupA):
                    if i < LL:
                        self.output[i] = val
                        self.outIndex += 1
                    if i + LR >= len(groupA):
                        self.output[i + len(groupB)] = val
                for i, val in enumerate(groupB):
                    self.output[i + self.outIndex] = val
                self.left = -3
                return

            if RL + RR == len(groupB):
                #put the groupA in the middle of groupB
                for i, val in enumerate(groupB):
                    if i < RL:
                        self.output[i] = val
                        self.outIndex += 1
                    if i + RR >= len(groupB):
                        self.output[i + len(groupA)] = val
                for i, val in enumerate(groupA):
                    self.output[i + self.outIndex] = val
                self.left = -3
                return
            return
                
            for self.indexA, l in enumerate(groupA[:LL]):
                self.output[self.indexA] = l

            for self.indexB, r in enumerate(groupB[:RL]):
                self.output[LL + self.indexB] = r

            for i, l in enumerate(groupA[-LR:]):
                self.output[len(self.output) - LR + i] = l

            for i, r in enumerate(groupB[-RR:]):
                self.output[len(self.output) - LR - RR + i] = r

            self.groupA = groupA[LL:len(self.groupA) - LR]
            self.groupB = groupB[RL:len(self.groupB) - RR]

            if len(self.groupA) != 0 or len(self.groupB) != 0:
                print(self.groupA, self.groupB)

            self.indexARight = len(self.groupA) - 1
            self.indexBRight = len(self.groupB) - 1

            self.outIndex = LL + RL
            self.indexORight -= RR + LR

            if self.indexA == self.indexARight and self.indexB == self.indexBRight:
                self.left = -3 # in inc you're done
            return
    
    def inc(self) -> bool:
        """do a merge, and if configured, switch which end the next merge will be done from
        returns True if the merge is completed"""
        if self.indexB >= len(self.groupB) and self.left != -3:
            while self.indexA <= self.indexARight:
                self.output[self.outIndex] = self.groupA[self.indexA]
                self.outIndex += 1
                self.indexA += 1
            return True
        if self.indexA >= len(self.groupA) and self.left != -3:
            while self.indexB <= self.indexBRight:
                self.output[self.outIndex] = self.groupB[self.indexB]
                self.outIndex += 1
                self.indexB += 1
            return True
        
        if self.left == True:
            iA = self.indexA
            iB = self.indexB
            iO = self.outIndex
            # if the element from A is less than the element from B
            if self.comp(self.groupA[iA], self.groupB[iB]):
                self.output[iO] = self.groupA[iA]
                self.indexA += 1
            else:
                self.output[iO] = self.groupB[iB]
                self.indexB += 1
            self.outIndex += 1
        elif self.left == False:
            iA = self.indexARight
            iB = self.indexBRight
            iO = self.indexORight

            if iB < 0: # done with group B
                while self.indexARight >= self.indexA:
                    self.output[self.outIndex] = self.groupA[self.indexA]
                    self.indexA += 1
                    self.outIndex += 1
                return True

            # if the element from A is less than the element from B
            if self.comp(self.groupA[iA], self.groupB[iB]):
                self.output[iO] = self.groupB[iB]
                self.indexBRight -= 1
            else:
                self.output[iO] = self.groupA[iA]
                self.indexARight -= 1
            self.indexORight -= 1
        else: # this signals no need to compare
            return True

        # go from other side
        self.left = self.toggle ^ self.left
        return (self.indexA > self.indexARight) and (self.indexB > self.indexBRight)
        #return (self.outIndex == len(self.output)) or (self.outIndex == self.indexORight)

def mergeSort(arr: list, comp=None, shuffle: bool=False, retStats: bool=False, retMid: bool=False) -> list:
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
    sizes: list = [1 for i in range(len(arr))]
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
        # for each ot the partitions
        for i, size in enumerate(sizes):
            #last group, odd one out
            if i + 1 >= len(sizes):
                break
            if shuffle:
                #if the next group comes before the current group:
                s = 0
                groups = []
                for ds in sizes:
                    groups.append(arr[s:s+ds])
                    s += ds

                for n, group in enumerate(groups[:-1]):
                    if comp(groups[n + 1][-1],group[0]):
                        swap(arr, n, n+1, sizes)
            #if comp(arr[start], arr[start + size + sizes[i + 1] - 1]):
            #    print("switch places", arr, sizes, arr[start], arr[start + size + sizes[i + 1] - 1])
            
                             # if the current groups comes before the next group
            elif (not shuffle) or comp(arr[start + size], arr[start + size - 1]): # if out of order
                # if there was a merge (will be false on the last of odd # partitions)
                mid = start + size
                stop = start + size + sizes[i + 1]
                L = arr[start:mid]
                if stop > len(arr):
                    stop = len(arr)
                R = arr[mid:stop]

                if retMid:
                    Lscores = list(map(lambda x: comp.getLatentScore(x), L))
                    Rscores = list(map(lambda x: comp.getLatentScore(x), R))

                    medians.append(np.abs(np.median(Lscores) - np.median(Rscores)))
                    for leftI,left in enumerate(L):
                        if left > split:
                            LL = (leftI)
                            break
                    else:
                        LL = 0

                    for leftI,left in enumerate(R):
                        if left > split:
                            RL = (leftI)
                            break
                    else:
                        RL = 0
                    
                    for rightI,right in enumerate(reversed(L)):
                        if right <= split:
                            LR = (rightI)
                            break
                    else:
                        LR = 0

                    for rightI,right in enumerate(reversed(R)):
                        if right <= split:
                            RR = (rightI)
                            break
                    else:
                        RR = 0

                    if LL < 0:
                        LL = 0
                    if LR < 0:
                        LR = 0
                    if RL < 0:
                        RL = 0
                    if RR < 0:
                        RR = 0

                    if len(L) > 1:
                        if LL + LR == len(L):
                            if RL + RR == len(R):
                                pass
                                #print("no")

                    percentages.append((LL/len(L) + RL/len(R) + LR/len(L) + RR/len(R)) / 4)

                    mergers.append(Merger(L, R, comp, start, stop, LL, LR, RL, RR))
                else:
                    mergers.append(Merger(L, R, comp, start, stop))

                #merge(comp, arr, start, start + size, start + size + sizes[i + 1])
                
                # merge the sizes
                sizes[i] += sizes[i + 1]
                sizes.pop(i + 1)
                

            else: 
                sizes[i] += sizes[i + 1]
                sizes.pop(i + 1)
            # keeps from going out of bounds, shouldn't be needed
            if i + 1 < len(sizes):
                # shimmy the start index
                start += size + sizes[i + 1]
        
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
                        arr[start + i] = v
                    mergers.remove(merger)

        # run dem stats
        if retStats:
            aucs, vars = list(), list()
            start = 0
            for size in sizes:
                curr = arr[start:size + start]
                sm = successMatrix(curr, list(sorted(curr))[:len(curr)//2], list(sorted(curr))[len(curr)//2:])
                aucs.append(aucSM(sm))
                vars.append(unbiasedMeanMatrixVar(sm))
                start += size
            npvar = np.var(aucs, ddof=1) / len(aucs)
            #stats = [sum(aucs) / len(sizes), sum(vars) / len(vars), float(npvar)]
            stats = [aucs, vars, float(npvar)]
            yield arr, stats
        elif not retMid:
            yield arr
        else:
            yield percentages, medians

def merge(comp, arr: list, start: int, mid: int, stop: int):
    """merges 2 slices of the array in place, as defined by arr[start] -> arr[mid], arr[mid] -> arr[stop]
    no return value"""
    i = j = 0
    k = start
    L = arr[start:mid]
    if stop > len(arr):
        stop = len(arr)
    R = arr[mid:stop]
    # print(start, mid, stop, L, R)
    # Copy data to temp arrays L[] and R[] 
    while i < len(L) and j < len(R): 
        #  if   L[i]< R[j]:
        if comp(L[i], R[j]):
            arr[k] = L[i] 
            i+=1
        else: 
            arr[k] = R[j] 
            j+=1
        k+=1
        
    # Checking if any element was left 
    while i < len(L): 
        arr[k] = L[i] 
        i+=1
        k+=1
        
    while j < len(R): 
        arr[k] = R[j] 
        j+=1
        k+=1

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

    test = 8
    if test == 1:
        from random import shuffle, seed
        from tqdm import trange

        for maxi in trange(10000):
            seed(maxi)
            arr = list(range(maxi))#79 is prime
            shuffle(arr)
            sArr = sorted(arr)
            for _ in mergeSort(arr, level=0, retStats=False):
                pass
            if arr != sArr:
                print(arr)
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
            comp = Comparator(data, level=3, rand=False, withComp=True)
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
            comp = Comparator(data, level=0, rand=True, withComp=True)
            for _ in mergeSort(data, comp=comp, retMid=True):
                pass
            avgs.append(len(comp))
        print(sum(avgs) / 100)

        avgs = []
        for plot in trange(100):
            data = continuousScale(256)
            comp = Comparator(data, level=0, rand=True, withComp=True)
            for _ in mergeSort(data, comp=comp, retMid=False):
                pass
            avgs.append(len(comp))
        print(sum(avgs) / 100)
        avgs = []
        for plot in trange(5):
            data = continuousScale(256)
            comp = Comparator(data, level=3, rand=True, withComp=True)
            for _ in mergeSort(data, comp=comp, retMid=True):
                pass
            avgs.append(len(comp))
        print(sum(avgs) / 5)

        avgs = []
        for plot in trange(5):
            data = continuousScale(256)
            comp = Comparator(data, level=3, rand=True, withComp=True)
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
        
        for l, (arr, stats) in enumerate(mergeSort(data, comp, retStats=True, retMid=False)):
            print(l + 1, list(map(lambda x: int(x >= 8), arr)), stats)