from comparator import Comparator

    
def mergeSort(arr: list, level=3, retComp = False) -> list:
    """mergeSort(arr: list, level=3)
    merge sorts the list arr with 'level' amount of optimization
    returns the Comparator object used"""
    comp = Comparator(arr,level)
    if len(arr) != 0:
        sizes: list = [1 for i in range(len(arr))]
    #merge partitions
    while sizes[0] < len(arr):
        i:int = 0
        start:int = 0
        while i < len(sizes) - 1:
            size:int = sizes[i]
            if merge(comp, arr, start, start + size, start + size + sizes[i + 1]):
                if i + 1 < len(sizes):
                    sizes[i] += sizes[i + 1]
                    sizes.pop(i + 1)
            if i + 1 < len(sizes):
                start += size + sizes[i + 1]
            i += 1
            #print(arr, sizes)
        yield arr, comp if retComp else arr


def merge(comp, arr: list, start: int, mid: int, stop: int) -> bool:
    i = j = 0
    k = start
    L = arr[start:mid]
    if stop > len(arr):
        stop = len(arr)
    R = arr[mid:stop]
    #print(start, mid, stop, L, R)
    if R: #check to see if there actually is a division
        # Copy data to temp arrays L[] and R[] 
        while i < len(L) and j < len(R): 
            #if L[i] < R[j]:
            if comp.compare(L[i], R[j]):
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
    return True if R else False

if __name__ == "__main__":
    l:list = [2,6,8,9,6,4,2,4,7]
    print(l)
    mergeSort(l)
    print(l)