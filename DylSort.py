import numpy as np

from DylMath import *
from DylMerger import *
from tqdm import tqdm

def genD0D1(d0d1: list, arr: list) -> tuple:
    D0, D1 = list(), list()
    
    for item in arr:
        if item in d0d1[0]:
            D0.append(item)
        elif item in d0d1[1]:
            D1.append(item)
    return D0, D1

def runStats(groups, d0d1, n, currLayer, nLayers):
    aucs, varOfSM, hanleyMcNeils, estimates = list(), list(), list(), list()
    start = 0
    for group in groups:
        if d0d1 != None:
            D0, D1 = genD0D1(d0d1, group)
        else:
            D0, D1 = list(sorted(group))[:len(group)//2], list(sorted(group))[len(group)//2:]
        if D0 and D1:
            sm = successMatrix(group, D0, D1)
            auc = aucSM(sm)
            if auc == auc:
                aucs.append(auc)
            hanleyMcNeils.append((len(D0), len(D1)))
            smVAR = unbiasedMeanMatrixVar(sm)
            if smVAR == smVAR and len(D0) > 2 and len(D1) > 2: # if not NaN
                varOfSM.append(smVAR)
    varOfAverageAUC = np.var(aucs, ddof=1) / len(aucs)
    aucs = np.array(aucs)
    avgAUC = np.mean(aucs)

    estimateNs = [list()]
    for i, ns in enumerate(hanleyMcNeils):
        estimateNs[0].append(ns)

    # while there are groups to 'merge'
    while len(estimateNs[-1]) != 1:
        # get the previous layer and sort by N0 + N1
        oldNs = sorted(estimateNs[-1], key=lambda x: sum(x))
        # roughly the same code as mergers creation
        estimateNs.append(list())
        while oldNs:
            i = 0
            toMerge = list()
            segments = min(n, len(oldNs) - i)
            for iSegment in range(segments):
                toMerge.append(oldNs.pop(0))
            estimateNs[-1].append([sum((x[0] for x in toMerge)), sum((x[1] for x in toMerge))])
        estimateNs[-1].sort(key=lambda x:sum(x))
        estimates.append(hanleyMcNeil(avgAUC, estimateNs[-1][-1][0], estimateNs[-1][-1][1]) / len(estimateNs[-1]))


    for i, (N0, N1) in enumerate(hanleyMcNeils):
        hanleyMcNeils[i] = hanleyMcNeil(avgAUC, N0, N1)
    if len(varOfSM) == 0:
        varEstimate = float(varOfAverageAUC)
    elif currLayer == nLayers:
        varEstimate = (sum(varOfSM) / (len(varOfSM)**2))
    else:
        varEstimate = (currLayer*(sum(varOfSM) / (len(varOfSM)**2)) + (nLayers - currLayer) * float(varOfAverageAUC)) / nLayers

    # bootstimate
    iters = min(len(aucs)**len(aucs), 1000)
    z = np.array([np.mean(aucs[np.random.randint(len(aucs), size=len(aucs))]) for i in range(iters)])
    z.sort()
    lowBoot = z[int(len(z) * 0.16)]
    highBoot = z[len(z) - int(len(z) * 0.16) - 1]
    # arcsin transform
    thingy = 1 / (2*np.sqrt(len(aucs)))
    lowSine = np.sin(np.arcsin(np.sqrt(avgAUC)) - thingy)**2
    highSine = np.sin(np.arcsin(np.sqrt(avgAUC)) + thingy)**2

    try:
        stats = [avgAUC, varEstimate, sum(hanleyMcNeils) / len(hanleyMcNeils)**2, lowBoot, highBoot, lowSine, highSine, (sum(varOfSM) / (len(varOfSM)**2)), float(varOfAverageAUC), *estimates]
    except ZeroDivisionError:
        stats = [avgAUC, varEstimate, sum(hanleyMcNeils) / len(hanleyMcNeils)**2, lowBoot, highBoot, lowSine, highSine, 0, float(varOfAverageAUC), *estimates]
    #stats = [aucs, vars, float(npvar)]
    return stats

def mergeSort(arr: list, comp=None, retStats: bool=False, retMid: bool=False, n: int=2, d0d1 = None, combGroups: bool=True, sortGroups: bool=False) -> list:
    """mergeSort(arr: list, level=3)
    Can either be provided a comparator or will make its own
    merge sorts the list arr with 'level' amount of optimization
    yields the arr after each pass through
    also yields the stats used if retStats"""
    if comp == None:
        from DylComp import Comparator
        comp = Comparator(arr, level=3, rand=False)

    # do this after comp created just in case
    if not arr:
        yield arr, None if retStats else arr
        return

    groups: list = list([arr[i]] for i in range(len(arr)))

    mergers = []

    if retMid:
        mini = min(arr)
        maxi = max(arr)
        split = (maxi - mini) // 2
        medians = []
        percentages = []
    currLayer = -1
    nLayers = calcNLayers(arr) - 1
    # while there are partitions
    while len(groups) != 1:
        currLayer += 1
        i = 0
        while len(groups) >= n:
            # last group, odd one out
            # get n arrays
            # feed the MultiMergers with them
            arrays = list()
            for iSegment in range(n):
                arrays.append(groups.pop(0))
            mergers.append(MultiMerger(arrays, comp, i, 0))
            i += 1
        #while we have active mergers
        while mergers:
            for merger in mergers:
                res = merger.inc()
                if res: #if that merger is done
                    #print(merger.output)
                    if -1 in merger.output:
                        raise FloatingPointError("it didn't actually do it")
                    for i, v in enumerate(merger.output):
                        if merger.output.count(v) > 1:
                            raise EnvironmentError(f"duplicated {v}")
                    comp.learn(merger.output)
                    if sortGroups:
                        groups.append(merger.output)
                    else:
                        groups.insert(0, merger.output)
                    mergers.remove(merger)
        if combGroups:
            arr = []
            for group in groups:
                arr.extend(group)
        else:
            arr = groups
        # run dem stats
        if retStats:
            stats = runStats(groups, d0d1, n, currLayer, nLayers)
            #stats = [aucs, vars, float(npvar)]
            yield arr, stats
        elif not retMid:
            yield arr
        else:
            yield percentages, medians

def treeMergeSort(arr: list, comp, n: int=2, retStats: bool=False, d0d1 = None, combGroups: bool=True):
    if n < 2:
        raise IOError("can't split a tree with n < 2")
    sizess = [[0, len(arr)]]
    while max(sizess[-1]) > n: #needs to be broken down further
        sizess.append([0])
        for i, size in enumerate(sizess[-2]):
            quotient, remainder = divmod(size, n)
            while size > 0:
                if remainder > 0:
                    sizess[-1].append(quotient + 1)
                    remainder -= 1
                    size -= quotient + 1
                else:
                    sizess[-1].append(quotient)
                    size -= quotient
    for sizes in sizess:
        for i, size in enumerate(sizes[1:], start=1):
            sizes[i] += sizes[i - 1]
    # do the first layer, which pulls from the array
    mergerss = [[], []]
    i = 0
    while i < len(sizess[-1]) - 1:
        segments = min(n, len(sizess[-1]) - i)
        groups = list()
        for iSeg in range(segments):
            group = arr[sizess[-1][i + iSeg]:sizess[-1][i + iSeg + 1]]
            if len(group) != 1: # such that we need to add another layer to it
                mergerss[0].append(MultiMerger([[img] for img in group], comp))
                groups.append(mergerss[0][-1].output)
            else:
                groups.append(group)
        mergerss[1].append(MultiMerger(groups, comp))
        i += segments
    # now build up layers of mergerss where the groups are the outputs of the last layer
    while len(mergerss[-1]) > 1: # while not on top level
        mergerss.append(list())
        i = 0
        while (segments == min(n, len(mergerss[-2]) - i)) != 0:
            groups = list()
            for iSeg in range(segments):
                groups.append(mergerss[-2][i + iSeg].output)
            mergerss[-1].append(MultiMerger(groups, comp))
            i += segments
    #print(mergerss)
    #for mergers in mergerss:
        #print([[len(group) for group in merger.groups] for merger in mergers])
    #print(len(mergerss[0]))
    left = True
    for layer, mergers in enumerate(mergerss, start=1):
        done = 0
        groups = list()
        while mergers:
            for merger in mergers if left else reversed(mergers):
                if merger.inc():
                    groups.append(merger.output)
                    mergers.remove(merger)
                    done += 1
        left != left
        if combGroups:
            arr = []
            for group in groups: arr.extend(group)
        else:
            arr = groups
        yield (arr, runStats(groups, d0d1, n, layer, len(mergerss))) if retStats else arr
    #print(f"n? {n} Did it sort right? {mergerss[-1][-1].output == sorted(mergerss[-1][-1].output)}. How many layers? {layer} How many comparisons? {len(comp)}")
if __name__ == "__main__":
    from DylRand import *

    test = 14
    if test == 1:
        from random import shuffle, seed
        from tqdm import trange
        from DylComp import Comparator
        #maxi = 79
        #if True:
        for maxi in trange(1000):
            seed(maxi)
            arr = list(range(maxi))#79 is prime
            shuffle(arr)
            sArr = sorted(arr)
            comp = Comparator(arr, level=0, rand=True)
            for _ in mergeSort(arr, comp=comp, retStats=False, retMid=True):
                pass
    elif test == 2:
        maxi = 1024
        l: list = randomDisease(maxi)
        for level in range(5):
            for arr, comp in mergeSort(l[:], level = level, retComp=True):
                pass
            print(level, len(comp), comp.dupHistory, min(comp.seps.values()), comp.optHistory)
    elif test == 3:
        m = Merger([0, 1, 2, 4, 5, 6, 7, 8, 9], [3], Comparator([0,1, 2, 3, 4, 5, 6, 7, 8, 9], level=3))
        while not m.inc():
            continue
            print(m.output)
        print(m.output)
        print(m.comp.compHistory)
    elif test == 4:
        data = continuousScale(128)
        arrays = [data[:]]
        print(data)
        for _ in mergeSort(data):
            arrays.append(data[:])
            print(data)
        graphROCs(arrays, True)
    elif test == 5:
        from DylComp import Comparator
        print("treeMergeSort")
        for n in range(2, 18):
            data = [*reversed(range(197))]
            comp = Comparator(data, level=0, rand=False)
            for _ in treeMergeSort(data, comp, n=n):
                pass
            print(n, len(comp))
        print("regular mergeSort")
        for n in range(2, 18):
            data = [*reversed(range(197))]
            comp = Comparator(data, level=0, rand=False)
            for _ in mergeSort(data, comp, n=n):
                pass
            print(n, len(comp))
    elif test == 6:
        from DylComp import Comparator
        from random import shuffle

        data = [i for i in range(21)]
        shuffle(data)
        comp = Comparator(data)
        for _ in mergeSort(data, comp, combGroups=False):
            print(_)
    elif test == 7:
        from DylData import continuousScale
        from DylComp import Comparator
        from tqdm import trange, tqdm

        avgs = []
        for plot in trange(100):
            data = continuousScale(256)
            comp = Comparator(data, level=0, rand=True)
            for _ in mergeSort(data, comp=comp, retMid=True):
                pass
            avgs.append(len(comp))
        print(sum(avgs) / 100)

        avgs = []
        for plot in trange(100):
            data = continuousScale(256)
            comp = Comparator(data, level=0, rand=True)
            for _ in mergeSort(data, comp=comp, retMid=False):
                pass
            avgs.append(len(comp))
        print(sum(avgs) / 100)
        avgs = []
        for plot in trange(5):
            data = continuousScale(256)
            comp = Comparator(data, level=3, rand=True)
            for _ in mergeSort(data, comp=comp, retMid=True):
                pass
            avgs.append(len(comp))
        print(sum(avgs) / 5)

        avgs = []
        for plot in trange(5):
            data = continuousScale(256)
            comp = Comparator(data, level=3, rand=True)
            for _ in mergeSort(data, comp=comp, retMid=False):
                pass
            avgs.append(len(comp))
        print(sum(avgs) / 5)
    elif test == 8:
        from DylComp import Comparator
        from DylData import continuousScale

        data = continuousScale(16)
        comp = Comparator(data, level=3, rand=True)
        print(data)
        for val in sorted(data):
            print(val, comp.getLatentScore(val))
        
        for l, (arr, stats) in enumerate(mergeSort(data, comp, retStats=True, retMid=False), start=1):
            print(l, list(map(lambda x: int(x >= 8), arr)), stats)
    elif test == 9:
        from DylComp import Comparator
        from DylData import continuousScale
        from tqdm import trange
        comp = Comparator(list(range(256)), level=3)
        for n in trange(2, 257):
            comp.clearHistory()
            arr = continuousScale(256)
            for _ in mergeSort(arr, comp=comp, n=n):
                pass
            assert arr == sorted(arr)
            print(n, len(comp), sep=',')
    elif test == 10:
        from DylComp import Comparator
        from DylData import continuousScale

        data = continuousScale(256)
        comp = Comparator(data, level=0, rand=True)
        print("AUC\tSMV\tNPV\tHmN")
        for arr in mergeSort(data, comp=comp, retStats=True):
            print(*[f"{x:0.5f}" for x in arr[1]], sep='\t')
    elif test == 11:
        from DylComp import Comparator
        from DylData import continuousScale

        data = continuousScale(256)
        arrays = [tuple(data)]
        print(data)
        comp = Comparator(data, level=0, rand=False)
        for _ in mergeSort(data, comp, n=4):
            print(data)
            arrays.append(tuple(data))
        print(data)
    elif test == 12:
        from DylComp import Comparator
        from DylRand import nearlySorted

        """ for insSort in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            data = nearlySorted(256, 10)
            comp = Comparator(data, rand=True, level=0)
            for _ in mergeSort(data, comp):
                print("done a layer")
            print(insSort, len(comp)) """
        
        data = nearlySorted(256, 10)
        for shuffle in (True, False):
            comp = Comparator(data, rand=True, level=0)
            arr = data[:]
            for _ in mergeSort(arr, comp):
                pass
            print(shuffle, len(comp), arr)
    elif test == 13:
        from DylData import continuousScale
        from DylComp import Comparator
        from DylRand import nearlySorted
        from functools import cmp_to_key
        from tqdm import trange

        class Image:
            def __init__(self, id, comp):
                self.id = id
                self.comp = comp
            def __lt__(self, other):
                return self.comp(self.id, other.id)

        shuffle = False

        #if True:
        for level in range(4):
            with open("/dev/urandom", 'rb') as file: 
                rand = [x for x in file.read(10)]
            seed = 1
            for val in rand: seed *= val
            seed %= 2**32
            near = nearlySorted(256, 40)
            cont = continuousScale(256)
            comp = Comparator(cont, level=level, rand=True, seed=seed)
            images = [Image(id, comp) for id in cont]
            images.sort()
            timCont = len(comp)

            comp.clearHistory()
            comp = Comparator(cont, level=level, rand=True, seed=seed)
            for _ in mergeSort(cont,comp):
                pass
            meCont = len(comp)

            comp.clearHistory()
            comp = Comparator(near, level=level, rand=True, seed=seed)

            images = [Image(id, comp) for id in near]
            images.sort()
            timNear = len(comp)

            comp = Comparator(near, level=level, rand=True, seed=seed)
            comp.clearHistory()
            for _ in mergeSort(near,comp):
                pass
            meNear = len(comp)

            comp = Comparator(near, level=level, rand=True, seed=seed)

            images = [Image(id, comp) for id in range(256)]
            images.sort()
            timFull = len(comp)

            data = list(range(256))

            comp = Comparator(data, level=level, rand=True, seed=seed)
            comp.clearHistory()
            for _ in mergeSort(data,comp):
                pass
            meFull = len(comp)

            print(level, timCont, meCont, timNear, meNear, timFull, meFull)
    elif test == 14:
        from DylData import continuousScale
        from DylComp import Comparator
        import matplotlib
        matplotlib.use('QT4Agg')
        import matplotlib.pyplot as plt
        font = {'size' : 24}
        matplotlib.rc('font', **font)

        data, D0, D1 = continuousScale(135, 87)
        comp = Comparator(data, rand=True)
        comps = list()
        rocs = list()
        for groups in treeMergeSort(data, comp, combGroups=False):
            rocs.append(list())
            comps.append(len(comp))
            for group in groups:
                gD0, gD1 = genD0D1((D0, D1), group)
                if gD0 and gD1:
                    rocs[-1].append(genROC(group, gD0, gD1))
            rocs[-1] = list(zip(*avROC(rocs[-1])))
            rocs[-1].reverse()
        
        if False:
            rows = int(math.ceil(math.sqrt(len(rocs))))
            cols = int(math.ceil(len(rocs) / rows))
            fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, num="plots")
            fig.suptitle("ROC Curves")
            for i,ax in enumerate(axes.flat):
                if i >= len(rocs):
                    continue
                ax.set(xlabel="False Positive Fraction", ylabel="True Positive Fraction")
                ax.label_outer()
                ax.plot((0,1),(0,1),c='red', linestyle=":")
                ax.plot(*zip(*rocs[i]), c='blue')
                ax.set_ylim(top=1.02, bottom=0)
                ax.set_xlim(left=-0.01, right=1)
                ax.set_title(f"Iteration #{i + 1} AUC: {auc(rocs[i]):.5f}")
        else:
            fig = plt.figure(figsize=(8, 8))
            plt.title("ROC Curves")
            ax = fig.add_subplot(1, 1, 1)
            linestyle_tuple = [
                ('loosely dashdotted',    (0, (3, 10, 1, 10))),
                ('dashdotted',            (0, (3, 5, 1, 5))),
                ('densely dashdotted',    (0, (3, 1, 1, 1))),

                ('loosely dashed',        (0, (5, 10))),
                ('dashed',                (0, (5, 5))),
                ('densely dashed',        (0, (5, 1))),

                ('loosely dotted',        (0, (1, 5))),
                ('dotted',                (0, (1, 2))),
                ('densely dotted',        (0, (1, 1))),

                ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
                ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
            ax.plot([], [], lw=0, label='Comparisons, AUC')
            for i, roc in enumerate(rocs):
                ax.plot(*zip(*roc), linestyle=linestyle_tuple[i][1], label=f"{comps[i]:04d}, {-auc(list(roc)):0.4f}", lw=0.4*(i + 3))
            ax.legend()
            ax.set_ylim(top=1, bottom=0)
            ax.set_xlim(left=0, right=1)
            ax.set(xlabel="False Positive Fraction", ylabel="True Positive Fraction")
        plt.tight_layout()
        plt.show()
    elif test == 15:
        from DylComp import Comparator
        from DylData import continuousScale
        from tqdm import trange

        data, D0, D1 = continuousScale(135, 87)
        comp = Comparator(data, rand=True)
        comp.genRand(len(D0), len(D1), 7.72, 'exponential')
        for groups in treeMergeSort(data, comp, combGroups=False):
            print('[', end='')
            for group in groups:
                print('[', end='')
                gD0, gD1 = genD0D1((D0, D1), group)
                for img in group[:-1]:
                    if img in gD0:
                        print(0, end=',')
                    elif img in gD1:
                        print(1, end=',')
                    else:
                        print('w', end=',')
                if group[-1] in gD0:
                    print(0, end=']')
                elif group[-1] in gD1:
                    print(1, end=']')
                else:
                    print('w', end=']')
            print(']', end='\n\n')
        
