import numpy as np

def genROC(arr:list, sensitivity:list, specificity:list):
    #x aksiks is specificity
    #y aksiks is sensitivity
    roc = np.empty((len(specificity),len(sensitivity)), dtype=int)
    for iA,spec in enumerate(reversed(specificity)):
        for iB,sens in enumerate(sensitivity):
            roc[iB,iA] = arr.index(spec) > arr.index(sens) 
    return roc

if __name__ == "__main__":
    sensitivity = [0, 1, 2, 3, 4]
    specificity = [5, 6, 7, 8, 9]
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(genROC(arr, sensitivity, specificity))