from tqdm import tqdm, trange
from math import log, ceil

from DylSort import mergeSort, combsort
from DylRand import *
from DylMath import *
from DylUtils import *

if __name__ == "__main__":
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
                        for arr,comp in sorter(data, level, True):#, total=ceil(log(len(lCopy), 2)):
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
        iters = 1000
        with open("variances.csv", "w") as vars:
            with open("aucs.csv", "w") as aucs:
                for i in trange(iters):
                    data = continuousScale(256)
                    sm = successMatrix(data)
                    aucs.write(str(auc(data)) + ',')
                    vars.write(str(unbiasedMeanMatrixVar(sm)) + ',')
                    arrays = [data[:]]
                    for arr, stats in mergeSort(data, level=0, retStats=True, rand=True):
                        arrays.append(data[:])
                        aucs.write(str(stats[0]) + ',')
                        vars.write(str(stats[1]) + ',')
                    aucs.write('\n')
                    vars.write('\n')
        #graphROCs(arrays, withPatches=False, withLine=True)
