l = [8, 7, 6, 5, 4, 3, 2, 1, 0]

def mergeSort(arr: list) -> list:
    if len(arr) != 0:
        sizes: list = [1 for i in range(len(arr))]
    #merge partitions
    while sizes[0] < len(arr):
        i = 0
        while i < len(sizes):
            size = sizes[i]
            if merge(arr,i * size, ((i + 1) * size), ((i + 2) * (size))):
                if i + 1 < len(sizes):
                    sizes[i] += sizes[i + 1]
                    sizes.pop(i + 1)
                    #print(i,sizes)
            i += 2
            print(arr)


def merge(arr: list, l1: int, split: int, r2: int):
    i = j = 0
    k = l1
    L = arr[l1:split]
    if r2 > len(arr):
        r2 = len(arr)
    R = arr[split:r2]
    print(l1, split, r2, L, R)
    if R: #check to see if there actually is a division
        # Copy data to temp arrays L[] and R[] 
        while i < len(L) and j < len(R): 
            if L[i] < R[j]: 
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
    mergeSort(l)
    print(l)