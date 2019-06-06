from random import choice, randint
from tqdm import tqdm, trange
from sklearn import metrics

from DylSort import mergeSort
from DylRand import nearlySorted, randomDisease
from DylMath import graphROCs
from DylUtils import *

if __name__ == "__main__":
    lMax: int = 256
    iters: int = 1
    levelMax: int = 1
    compLens = [0] * (levelMax + 1)
    with open("results.csv", "w") as f:
        level: int = 1
        if True:
        #for level in trange(levelMax + 1):
            counts:dict = dict()
            for n in range(lMax):
                counts[n] = 0
            for i in range(iters):
                known:dict = dict()
                #lCopy: list = nearlySorted(lMax, 0)
                lCopy = randomDisease(lMax)
                actual = [*range(lMax)]
                arrs = [tuple(lCopy)]
                print("sorting")
                for arr,comp in tqdm(mergeSort(lCopy, level, True)):
                    arrs.append(tuple(arr))
                print("generating ROC's")
                graphROCs(arrs)
                compLens[level] += comp.compHistory
            compLens[level] /= iters
        print(compLens)