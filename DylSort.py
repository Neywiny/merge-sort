from DylComp import Comparator
from DylUtils import *

def swap(arr: list, indexA: int, indexB: int):
    temp = arr[indexA]
    arr[indexA] = arr[indexB]
    arr[indexB] = temp

def mergeSort(arr: list, level=3, retComp = False) -> list:
    """mergeSort(arr: list, level=3)
    merge sorts the list arr with 'level' amount of optimization
    yields the arr after each pass through
    also yields the Comparator object used if retComp"""
    comp = Comparator(arr,level)

    # do this after comp created just in case
    if not arr:
        yield arr, comp if retComp else arr
        return
    sizes: list = [1 for i in range(len(arr))]
    # while there are partitions
    while sizes[0] < len(arr):
        # i is the partition number
        i: int = 0
        # start is the index the partition starts at
        start: int = 0
        # for each ot the partitions
        while i < len(sizes) - 1:
            size: int = sizes[i]
            # if there was a merge (will be false on the last of odd # partitions)
            #if comp(arr[start], arr[start + size + sizes[i + 1] - 1]):
                #print("switch places")
            if level != 4 or comp(arr[start + size], arr[start + size - 1]): # if out of order
                if merge(comp, arr, start, start + size, start + size + sizes[i + 1]):
                    # keeps from going out of bounds, shouldn't be needed
                    if i + 1 < len(sizes):
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
            # move to the next partition
            i += 1
            # print(arr, sizes)
        yield arr, comp if retComp else arr


def merge(comp, arr: list, start: int, mid: int, stop: int) -> bool:
    i = j = 0
    k = start
    L = arr[start:mid]
    if stop > len(arr):
        stop = len(arr)
    R = arr[mid:stop]
    # print(start, mid, stop, L, R)
    if R: # check to see if there actually is a division
        # Copy data to temp arrays L[] and R[] 
        while i < len(L) and j < len(R): 
            # if L[i] < R[j]:
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
    return True if R else False

def combsort(arr: list, level: int=3, retComp: bool=False) -> list:
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
    l: list = [2,6,8,9,6,4,2,4,7]
    print(l)
    for arr, comp in mergeSort(l, retComp=True):
        print(comp.compHistory, comp.dupHistory)
    print(l)

    l: list = [8,7,6,5,4,3,2,1,0]
    print(l)
    for arr, comp in combsort(l, retComp=True):
        print(comp.compHistory, comp.dupHistory)
    print(l)