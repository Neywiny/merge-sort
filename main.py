#!/usr/bin/python3.6

import math
import pickle
import os.path
import sys
from multiprocessing import Pool
from warnings import filterwarnings

from DylComp import Comparator
from DylMath import *
from DylSort import mergeSort

def sort(tid, i=0):
    results = list()
    data = continuousScale(256)
    sm = successMatrix(data)
    comp = Comparator(data, level=0, rand=True)
    comp.genRand(len(data)//2, len(data)//2, 7.72, 'exponential')
    for l, (arr, stats) in enumerate(mergeSort(data, comp, retStats=True, retMid=retMid, n=2)):
        stats.extend([len(comp), comp.genSeps()])
        results.append(stats)
    if arr != sorted(arr, key=lambda x: comp.getLatentScore(x)[0]):
        print(arr)
        print(sorted(arr, key=lambda x: comp.getLatentScore(x)[0]))
        raise EOFError("did not sort")
    return results


if __name__ == "__main__":
    filterwarnings('ignore')
    test = 3
    if test == 1:
        lMax: int = 2**8
        iters: int = 1
        levelMax: int = 0
        #data, D0, D1 = continuousScale("sampledata.csv")
        data = continuousScale(lMax)
        #print(data)
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
                        arrs = [data[:]]
                        comp = Comparator(data, level=0, rand=True)
                        for arr in tqdm(sorter(data, comp=comp), total=math.ceil(math.log(lMax, 2))):
                            #f.write(str(list(comp.minSeps.values())) + ', ' + str(unbiasedMeanMatrixVar(successMatrix(data))) + '\n')
                            arrs.append(data[:])
                    """for key, value in comp.counts.items():
                        f.write(str(key))
                        f.write(",")
                        f.write(str(value))
                        f.write(",")
                        f.write(str(comp.minSeps[key]))
                        f.write("\n")"""
        #print(arrs)
        graphROCs(arrs)
    elif test == 2:
        from DylData import continuousScale
        power = 8
        data = continuousScale(2**power)
        arrays = [data[:]]
        #graphROC(data)
        #print(successMatrix(data))
        comp = Comparator(data, level=0, rand=True)
        comp.genRand(2**(power - 1), 2**(power - 1), 1.7, 'normal')
        #comp.bRecord = False
        for _ in tqdm(mergeSort(data, comp=comp), total=power):
            arrays.append(data[:])
            #print(successMatrix(data))
        graphROCs(arrays, withLine=True,withPatches=False, D0=list(range(2**(power - 1))), D1=range(2**(power - 1), 2**(power)))
    elif test == 3:
        from tqdm import tqdm
        results = list()
        if len(sys.argv) > 1:
            args = list(map(lambda x: eval(x), sys.argv[2:]))
            iters = args[0]
            ids = [*range(iters)]
            retMid = args[1]
            topBar = tqdm(total=iters, smoothing=0, bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            botBar = tqdm(total=iters, smoothing=0, bar_format="{bar}")
            with Pool() as p:
                for result in p.imap_unordered(sort, ids):
                    topBar.update()
                    botBar.update()
                    results.append(result)
            botBar.close()
            topBar.close()
            print('\n')
        else:
            retMid = False
            iters = 1
            results = [sort(0, i) for i in range(iters)]
        #change output file if requested to do so
        with open('results/results'+str(sys.argv[1] if len(sys.argv) > 1 else ''), 'wb') as f:
            pickle.dump(results, f)
    elif test == 4:
        from random import shuffle
        import numpy as np
        import matplotlib.pyplot as plt

        power = 10

        img = np.zeros((power + 1, 2**power))

        data = list(range(2**power))
        comp = Comparator(data, level=0)

        shuffle(data)
        img[0] = data[:]
        for i,_ in enumerate(mergeSort(data, comp=comp), start=1):
            img[i] = data[:]

        plt.imshow(img, cmap='Greys', extent=[0, 2**power, 0, 2**power], aspect=1)

        plt.show()
