#!/usr/bin/python3.6

import math
import pickle
import sys
from multiprocessing import Pool
from warnings import filterwarnings

from DylSort import mergeSort, combsort
from DylMath import *
from DylComp import Comparator

def sort(tid, i=0):
    results = list()
    data = continuousScale(256)
    sm = successMatrix(data)
    comp = Comparator(data, level=0, rand=True, withComp=withComp)
    for l, (arr, stats) in enumerate(mergeSort(data, comp, retStats=True)):
        stats.extend([len(comp), list(comp.minSeps.items())])
        results.append(stats)
    results.append(comp.compHistory)
    return results


if __name__ == "__main__":
    filterwarnings('ignore')
    test = 3
    if test == 1:
        lMax: int = 2048
        iters: int = 1
        levelMax: int = 3
        data, D0, D1 = continuousScale("sampledata.csv")
        print(data)
        lMax = len(data)
        #exit()
        with open("sample_results.csv", "w") as f:
            level: int = levelMax
            sorter = mergeSort
            if True:
            #for sorter in tqdm((mergeSort, combsort)):
                #for n in range(levelMax + 1):
                #    counts[n] = list((0, 0, 0))
                if True:
                #for level in trange(levelMax + 1):
                    for i in trange(iters):
                        known:dict = dict()
                        actual = [*range(lMax)]
                        arrs = [data[:]]
                        for arr,comp in sorter(data, level, True):#, total=math.cail(math.log(len(lCopy), 2)):
                            f.write(writer(comp.minSeps.values()) + ', ' + str(unbiasedMeanMatrixVar(successMatrix(data))) + '\n')
                            arrs.append(data[:])
                    for key, value in comp.counts.items():
                        f.write(str(key))
                        f.write(",")
                        f.write(str(value))
                        f.write(",")
                        f.write(str(comp.minSeps[key]))
                        f.write("\n")

        graphROCs(arrs)
    elif test == 2:
        nums = [i for i in range(2048)]
        data = [nums.pop((len(nums)) // 2) if i % 2 else nums.pop(0) for i in range(len(nums))]
        power = 8
        data = randomDisease(2**power)
        arrays = [data[:]]
        #graphROC(data)
        #print(successMatrix(data))
        for arr,_ in tqdm(mergeSort(data, level=0, rand=False), total=power):
            arrays.append(data[:])
            #print(successMatrix(data))
        graphROCs(arrays, withLine=True,withPatches=True)
    elif test == 3:
        results = list()
        if len(sys.argv) > 1:
            args = list(map(lambda x: eval(x), sys.argv[2:]))
            iters = args[0]
            ids = [*range(iters)]
            withComp = args[1]
            with Pool() as p:
                i = 1
                for result in p.imap_unordered(sort, ids):
                    print(f'{i} / {iters}', end='\r', flush=True)
                    i += 1
                    results.append(result)
            print('\n')
        else:
            withComp = False
            iters = 1
            results = [sort(0, i) for i in range(iters)]
        #change output file if requested to do so
        with open('results/results'+str(sys.argv[1] if len(sys.argv) > 1 else ''), 'wb') as f:
            pickle.dump(results, f)
