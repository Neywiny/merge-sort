import numpy as np
np.seterr(all="ignore")

def continuousScale(*args):
    """creates a set of continuous scale data. 
    If provided a filename, returns the indecies for sorting, D0, and D1
    If provided a number, just creates and returns the indecies alternating from D0 and D1 indecies"""
    if isinstance(args[0], str):
        filename = args[0]
        data = []
        D0 = []
        D1 = []
        with open(filename) as f:
            for i,line in enumerate(f):
                if len(line) > 10:
                    line = line.strip().split(" ")
                    point = float(line[2])
                    data.append(point)
                    if line[1] == "1": # case should be positive
                        D1.append(i)
                    else: # case should be negative
                        D0.append(i)
        newData = [-1 for i in range(len(data))]
        #print(data)
        for i, d in enumerate(sorted(data)):
            newData[i] = data.index(d)
        D0.sort()
        D1.sort()
        return newData, D0, D1
    elif isinstance(args[0], int):
        data = list()
        for i in range(0, args[0] // 2):
            data.append(i + args[0] // 2)
            data.append(i)
        return data

if __name__ == "__main__":
    data = continuousScale(256, True)
    print(data)