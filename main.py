from random import choice
from tqdm import tqdm
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
    iters:int = 1
    levelMax:int = 0
    with open("results.csv", "w") as f:
        for i in range(lMax):
            f.write(str(i) + ',')
        f.write('\n')
        level:int = 0
        if True:
        #for level in tqdm(range(levelMax + 1)):
            counts:dict = dict()
            for n in range(lMax):
                counts[n] = 0
            for i in range(iters):#tqdm(range(iters)):
                known:dict = dict()
                lCopy:list = nearlySorted(lMax, 0)
                actual = [*range(lMax)]
                for arr in mergeSort(lCopy, level):
                    print("PC:",calcPC(arr))
                    predictions = [arr.index(i) for i in range(lMax)]
                    confusion_matrix = metrics.confusion_matrix(actual, predictions)
                    #print(confusion_matrix)
                    print("Jaccard:",metrics.jaccard_score(actual, predictions, average='weighted'))
                    #print(genROC(arr, [*range(lMax//2)], [*range(lMax//2, lMax)]))
            