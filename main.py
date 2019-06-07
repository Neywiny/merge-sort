from tqdm import tqdm, trange
from math import log, ceil

from DylSort import mergeSort, combsort
from DylRand import nearlySorted, randomDisease
from DylMath import graphROCs
from DylUtils import *

if __name__ == "__main__":
    lMax: int = 2048
    iters: int = 1
    levelMax: int = 4
    data = []
    with open("sampledata.txt") as f:
        for line in f:
            if len(line) > 10:
                line = line.split(" ")
                data.append(float(line[1]))
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
                    #arrs = [tuple(data)]
                    for arr,comp in sorter(data, level, True):#, total=ceil(log(len(lCopy), 2)):
                        f.write(str(list(comp.minSeps.values())) + '\n')
                        #arrs.append(tuple(arr))
                    #graphROCs(arrs)
                for key, value in comp.counts.items():
                    f.write(str(key))
                    f.write(",")
                    f.write(str(value))
                    f.write(",")
                    f.write(str(comp.minSeps[key]))
                    f.write("\n")
                