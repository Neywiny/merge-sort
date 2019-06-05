from random import choice, randint
from tqdm import tqdm, trange
from sklearn import metrics

from better import mergeSort
from nearlySorted import nearlySorted
from roc import genROC

def calcPC(arr:list) -> float:
    #calc % correct
    pc = 0
    for i in range(len(arr) - 1):
        pc += 1 if arr[i] < arr[i + 1] else 0
    pc /= (i + 1)
    return pc

if __name__ == "__main__":
    lMax:int = 100
    iters:int = 1000
    levelMax:int = 3
    compLens = [0] * (levelMax + 1)
    with open("results.csv", "w") as f:
        #level:int = 3
        #if True:
        for level in trange(levelMax + 1):
            counts:dict = dict()
            for n in range(lMax):
                counts[n] = 0
            for i in trange(iters):
                known:dict = dict()
                lCopy:list = nearlySorted(lMax, 0)
                actual = [*range(lMax)]
                for arr,comp in mergeSort(lCopy, level, True):
                    continue
                    print("PC:",calcPC(arr))
                    predictions = [arr.index(i) for i in range(lMax)]
                    confusion_matrix = metrics.confusion_matrix(actual, predictions)
                    print("Jaccard:",metrics.jaccard_score(actual, predictions, average='weighted'))
                compLens[level] += comp.compHistory
            compLens[level] /= iters
        print(compLens)