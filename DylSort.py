import numpy as np

from DylComp import Comparator
from DylRand import *
from DylMath import *
from DylUtils import *

def swap(arr: list, indexA: int, indexB: int, sizes: list=None):
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
    def __init__(self, groupA: list, groupB: list, comp: Comparator, start=0, stop=0, toggle:bool=True):
        self.groupA: list = groupA
        self.groupB: list = groupB
        comp.learn(groupA)
        comp.learn(groupB)
        self.output: list = [0 for i in [*groupA, *groupB]]
        self.indexA: int = 0
        self.indexB: int = 0
        self.outIndex: int = 0
        self.indexARight: int = len(groupA) - 1
        self.indexBRight: int = len(groupB) - 1
        self.indexORight: int = len(self.output) - 1
        self.comp: Comparator = comp
        self.left: bool = True
        self.toggle: bool = True
    
        #hold onto these for the parent object, they're a secret tool that will help us later
        self.start = start
        self.stop = stop
    def inc(self):
        if self.indexB == len(self.groupB):
            while self.indexA <= self.indexARight:
                self.output[self.outIndex] = self.groupA[self.indexA]
                self.outIndex += 1
                self.indexA += 1
            return True
        if self.indexA == len(self.groupA):
            while self.indexB <= self.indexBRight:
                self.output[self.outIndex] = self.groupB[self.indexB]
                self.outIndex += 1
                self.indexB += 1
            return True
        if self.left:
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
        else:
            iA = self.indexARight
            iB = self.indexBRight
            iO = self.indexORight
            # if the element from A is less than the element from B
            if self.comp(self.groupA[iA], self.groupB[iB]):
                self.output[iO] = self.groupB[iB]
                self.indexBRight -= 1
            else:
                self.output[iO] = self.groupA[iA]
                self.indexARight -= 1
            self.indexORight -= 1

        # go from other side
        self.left = self.toggle ^ self.left
        return (self.indexA > self.indexARight) and (self.indexB > self.indexBRight)
        #return (self.outIndex == len(self.output)) or (self.outIndex == self.indexORight)

def mergeSort(arr: list, comp: Comparator=None, shuffle: bool=False, retStats: bool=False) -> list:
    """mergeSort(arr: list, level=3)
    merge sorts the list arr with 'level' amount of optimization
    yields the arr after each pass through
    also yields the stats used if retStats"""
    if comp == None:
        comp = Comparator(arr,level, rand)

    # do this after comp created just in case
    if not arr:
        yield arr, None if retStats else arr
        return
    sizes: list = [1 for i in range(len(arr))]
    mergers = []
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
                    comp.learn(merger.output)
                    for i, v in enumerate(merger.output):
                        arr[start + i] = v
                    mergers.remove(merger)

        # run dem stats
        aucs, vars = list(), list()
        start = 0
        for size in sizes:
            curr = arr[start:size + start]
            sm = successMatrix(curr, list(sorted(curr))[:len(curr)//2], list(sorted(curr))[len(curr)//2:])
            aucs.append(aucSM(sm))
            vars.append(unbiasedMeanMatrixVar(sm))
            start += size
        npvar = np.var(aucs, ddof=1) / len(aucs)
        stats = [sum(aucs) / len(sizes), sum(vars) / len(vars), float(npvar)]
        yield arr, stats if retStats else arr

def merge(comp, arr: list, start: int, mid: int, stop: int) -> bool:
    i = j = 0
    k = start
    L = arr[start:mid]
    if stop > len(arr):
        stop = len(arr)
    R = arr[mid:stop]
    if len(L) != len(R):
        print("unequal lengths", L, R)
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

def combsort(arr: list, level: int=3, retComp: bool=False, rand: bool=False) -> list:
    comp = Comparator(arr, level)
    gap = len(arr)
    shrink = 1.3
    sorted = False

    while sorted == False:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted = True

        i = 0
        while i + gap < len(arr):
            if comp(arr[i + gap], arr[i]):
                swap(arr, i, i + gap)
                sorted = False
            i = i + 1
        yield arr, comp if retComp else arr

if __name__ == "__main__":
    test = 4
    if test == 1:
        maxi = 512
        l: list = randomDisease(maxi)
        for arr, comp in mergeSort(l, level = 4, retComp=True):
            print(len(comp), comp.dupHistory, min(comp.minSeps.values()), comp.optHistory)

        l: list = randomDisease(maxi)
        for arr, comp in combsort(l, level = 4, retComp=True):
            print(len(comp), comp.dupHistory, min(comp.minSeps.values()), comp.optHistory)
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
        data = continuousScale(256)
        arrays = [data[:]]
        print(data)
        for _ in mergeSort(data, rand=True):
            arrays.append(data[:])
            print(data)
        graphROCs(arrays, True)