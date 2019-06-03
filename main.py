from random import randint
from tqdm import tqdm
from better import mergeSort

""" def mergeSort(arr: list) -> dict():
    comparisons: dict = dict()
    if len(arr) > 1: 
        mid:int = len(arr)//2 #Finding the mid of the array 
        L:list = arr[:mid] # Dividing the array elements  
        R:list = arr[mid:] # into 2 halves 
        lComp:dict = mergeSort(L) # Sorting the first half 
        rComp:dict = mergeSort(R) # Sorting the second half 
        for arrIndex,v in lComp.items():
            comparisons[arrIndex]:int = v
        for arrIndex,v in rComp.items():
            if arrIndex in lComp:
                comparisons[arrIndex] += v
            else:
                comparisons[arrIndex]:int = v
        leftIndex = rightIndex = arrIndex = 0
        
        # Copy data to temp arrays L[] and R[] 
        while leftIndex < len(L) and rightIndex < len(R): 
            comp = -1
            if L[leftIndex] in known:
                if R[rightIndex] in known[L[leftIndex]]:
                    comp = known[L[leftIndex]][R[rightIndex]]
            if R[leftIndex] in known:
                if L[leftIndex] in known[R[rightIndex]]:
                    comp = not known[R[rightIndex]][L[leftIndex]]
            if comp == -1:
                if L[leftIndex] in comparisons:
                    comparisons[L[leftIndex]] += 1
                else:
                    comparisons[L[leftIndex]]:int = 1
                if R[rightIndex] in comparisons:
                    comparisons[R[rightIndex]] += 1
                else:
                    comparisons[R[rightIndex]]:int = 1

                if not (R[rightIndex] in known):
                    known[R[rightIndex]] = dict()
                if not (L[leftIndex] in known):
                    known[L[leftIndex]] = dict()
                comp = L[leftIndex] < R[rightIndex]
                print(L[leftIndex] , R[rightIndex])
                global count
                for c in known[R[rightIndex]].keys():
                    # for all c s.t. we know the relationship b/t b and c
                    if (known[R[rightIndex]][c] == comp) and (c not in known[L[leftIndex]]): #a < b < c or a > b > c
                        known[L[leftIndex]][c] = comp   #a < c or a > c 
                        count += 1


                known[L[leftIndex]][R[rightIndex]]:bool = comp
                known[R[rightIndex]][L[leftIndex]]:bool = not comp

            if comp: 
                arr[arrIndex] = L[leftIndex] 
                leftIndex += 1
            else: 
                arr[arrIndex] = R[rightIndex] 
                rightIndex += 1
            arrIndex += 1
          
        # Checking if any element was left 
        while leftIndex < len(L): 
            arr[arrIndex] = L[leftIndex] 
            leftIndex += 1
            arrIndex += 1
          
        while rightIndex < len(R): 
            arr[arrIndex] = R[rightIndex] 
            rightIndex += 1
            arrIndex += 1
    return comparisons """

if __name__ == "__main__":
    lMax:int = 100
    with open("results.csv", "w") as f:
        for i in range(lMax):
            f.write(str(i) + ',')
        f.write('\n')
        for i in tqdm(range(1000)):
            known:dict = dict()
            l:list = [randint(0, lMax - 1) for i in range(lMax)]
            sL = sorted(l)
            global count
            comp = mergeSort(l)
            counts = dict()
            for n in range(lMax):
                counts[n] = 0
            for comparison in comp.compHistory:
                counts[comparison[0]] += 1
                counts[comparison[1]] += 1
            for n in range(lMax):
                f.write(str(counts[n]) + ',')
            f.write('\n')
            if sL != l:
                print("woops")
                print(l, sL)