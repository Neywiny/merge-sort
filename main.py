from random import randint
from tqdm import tqdm
from better import mergeSort

if __name__ == "__main__":
    lMax: int = 100
    iters: int = 1000
    with open("results.csv", "w") as f:
        for i in range(lMax):
            f.write(str(i) + ',')
        f.write('\n')
        level = 3
        if True:
        #for level in tqdm(range(4)):
            counts = dict()
            for n in range(lMax):
                counts[n] = 0
            for i in tqdm(range(iters)):
                known:dict = dict()
                l:list = [randint(0, lMax - 1) for i in range(lMax)]
                sL = sorted(l)
                global count
                comp = mergeSort(l, level)
                for comparison in comp.compHistory:
                    counts[comparison[0]] += 1
                    counts[comparison[1]] += 1
                if sL != l:
                    print("woops")
                    print(l, sL)
            for n in range(lMax):
                f.write(str(counts[n] / iters) + ',')
            f.write('\n')