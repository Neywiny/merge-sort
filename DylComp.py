import numpy as np
np.seterr(all="ignore")
from warnings import filterwarnings
filterwarnings("ignore")

class Comparator:
    """A class for comparing 2 values.
        Controlled with the optimizaiton level and if you want random decisions or not
        Either provide objects in init or call .genLookup before you do any comparing
        Optimization levels: will not optimize, store result, do abc association, do recursive association
        Rand: defaults to False. If True will create data from random distributions with seed parameter
    """
    def __init__(self, objects: list = None, level: int = 3, rand: bool=False, seed: int=None):
        self.clearHistory()
        self.rand: bool = rand
        self.seed: int = seed
        self.level: int = level
        self.compHistory = list()
        self.dupCount = 0
        self.optCount = 0
        self.counts = dict()
        self.minSeps = dict()
        self.bRecord = True
        if objects != None:
            self.genLookup(objects)
            if rand:
                self.n0 = self.n1 = len(objects) // 2
                self.n1 += len(objects) % 2
                self.genRand(self.n0, self.n1, 1.7, 'normal')
        self.last: tuple = None
    def __len__(self):
        """returns either the number of comparisons done"""
        return self.compHistory if isinstance(self.compHistory, int) else len(self.compHistory)
    def __call__(self, a, b):
        """returns a < b"""
        return self.compare(a,b)
    
    def genRand(self, n0, n1, sep, dist):
        from os import getpid, uname
        from time import time
        # get a random seed for each node and each process on that node, and the time
        if self.seed == None:
            self.seed = (int(str(ord(uname()[1][-1])) + str(getpid()) + str(int(time()))) % 2**31)
        np.random.seed(self.seed)
        if dist == 'normal':
            self.vals = (tuple(np.random.normal(size=n0,loc=0)) + tuple(np.random.normal(size=n1,loc=sep)))
        elif dist == 'exponential':
            self.vals = (tuple(np.random.exponential(size=n0,scale=1)) + tuple(np.random.exponential(size=n1,scale=sep)))
        else:
            raise NotImplementedError("distibution must be one of ['normal','exponential']")

    def record(self, vals):
        if not self.bRecord:
            return
        for val in vals:
            self.counts[val] += 1

            #count minimum separations
            new = len(self) - self.minSeps[val][1]
            self.minSeps[val] = [min(new, self.minSeps[val][0]), len(self)]

            if self.last:
                if val in self.last:
                    self.dupCount += 1

        self.compHistory.append(tuple(vals))
        self.last = tuple(vals)

    def compare(self, a, b) -> bool:
        """returns a < b"""
        #if not self.rand and self.level == 0:
            #return a < b
        try:
            if b in self.lookup[a].keys():
                # print("cache hit")
                return self.lookup[a][b]
            elif a in self.lookup[b].keys():
                # print("cache hit")
                return self.lookup[b][a]
            else:
                #only gets it right 80% of the time
                needComp = True
                if self.rand: # where we are for most things
                    
                    aScore, aNeg = self.getLatentScore(a)
                    bScore, bNeg = self.getLatentScore(b)

                    res: bool = aScore < bScore
                else:
                    res: bool = a < b
                if needComp:
                    self.record([a, b])
                    
                    if self.level > 0:
                        self.lookup[a][b]:bool = res
                        self.lookup[b][a]:bool = not res

                    if self.level > 1:
                        # print("optimizing")
                        # single optimization
                        for c in self.lookup[b].keys():
                            # for all c s.t. we know the relationship b/t b and c
                            if self.lookup[b][c] == res:# a < b < c or a > b > c
                                # print("optimized", a, c)
                                self.lookup[a][c]:bool = res  # a < c or a > c 
                                self.lookup[c][a]:bool = not res
                                self.optCount += 1
                        for c in self.lookup[a].keys():
                            # for all c s.t. we know the relationship b/t b and c
                            if self.lookup[a][c] == (not res):# a < b < c or a > b > c
                                # print("optimized", b, c)
                                self.lookup[b][c]:bool = not res  # a < c or a > c 
                                self.lookup[c][b]:bool = res
                                self.optCount += 1
                    if self.level > 2:
                        # O(n!) optimization. Make sure to use a copy of objects
                        self.optCount += Comparator.optimize(list(self.lookup.keys()), self.lookup, res, a, b)
                return res
        except AttributeError as e:
            raise LookupError("You need to generate the lookup first")

    def getLatentScore(self, index: int) -> float:
        """gets the latent score of a given index"""
        if self.rand:
            return self.vals[index], index <= self.n0

    def genLookup(self, objects: list):
        """generate the lookup table and statistics for each object provided"""
        self.lookup:dict = dict()
        self.objects: list = objects
        for object in objects:
            self.lookup[object] = dict()
        self.clearHistory()
    
    def clearHistory(self):
        """clear the history statistics of comparisons"""
        if hasattr(self, "objects"):
            self.compHistory = list()
            self.last = None
            self.dupCount = 0
            for object in self.objects:
                self.counts[object] = 0
                self.minSeps[object] = [2*len(self.objects),0]

    def learn(self, arr: list, img=None, maxi=False):
        """learn the order of the array provided, assuming the current optimization level allows it
        if img is provided, learns the arr w.r.t. the img and if it is max or min"""
        if img == None and self.level > 1:
            for i, a in enumerate(arr):
                for b in arr[i + 1:]:
                    self.lookup[a][b] = True
                    self.lookup[b][a] = False
                    if self.level > 2:
                        Comparator.optimize(self.objects, self.lookup, True, a, b)
        elif img != None and self.level > 1:
            for b in arr:
                if b != img:
                    self.lookup[img][b] = not maxi
                    self.lookup[b][img] = maxi
                    if self.level > 2:
                        Comparator.optimize(self.objects, self.lookup, maxi, b, img)

    def max(self, arr) -> (int, int):
        if len(arr) == 0:
            raise NotImplementedError("I can't take the max of nothing")
        self.record(arr)
        maxVal = arr[0]
        maxScore = self.getLatentScore(arr[0]) if self.rand else arr[0]
        maxInd = 0
        for i, imageID in enumerate(arr[1:], start=1):
            score = self.getLatentScore(imageID) if self.rand else arr[i]
            if score > maxScore:
                maxInd = i
                maxVal = imageID
                maxScore = score
        self.learn(arr, maxVal, True)
        return maxInd, maxVal

    def min(self, arr) -> (int, int):
        if len(arr) == 0:
            raise NotImplementedError("I can't take the min of nothing")
        self.record(arr)
        minVal = arr[0]
        minScore = self.getLatentScore(arr[0]) if self.rand else arr[0]
        minInd = 0
        for i, imageID in enumerate(arr[1:], start=1):
            score = self.getLatentScore(imageID) if self.rand else arr[i]
            if score < minScore:
                minInd = i
                minVal = imageID
                minScore = score
        self.learn(arr, minVal, False)
        return minInd, minVal

    @staticmethod
    def optimize(objects: list, lookup: dict, res: bool, a, b):
        """recursive optimization algorithm for adding a node to a fully connected graph"""
        if objects:
            nObjects: list = []
            for c in list(lookup[b]): 
                # for all c s.t. c is a neighbor of b
                if c in objects and lookup[b][c] == res and c != a and c not in lookup[a]: 
                    # s.t. a > b > c or a < b < c
                    nObjects.append(c)
                    # print("optimized", a, c)
                    lookup[a][c]:bool = res
                    lookup[c][a]:bool = not res
                    return 1 + Comparator.optimize(nObjects, lookup, res, b, c)
        return 0

if __name__ == "__main__":
    test = 5
    if test == 1:
        comp = Comparator([i for i in range(10)], 2)
        print(comp.compare(0, 1))
        print(comp.compare(1, 2))
        print(comp.compare(0, 2))
        print(comp.compare(2, 0))
        print(comp.compare(0, 3))
        print(comp.compare(2, 3))
    elif test == 2:
        comp = Comparator([i for i in range(10)], level=3, rand=True)
        print(comp.neg)
        print(comp.pos)
    elif test == 4:
        comp = Comparator([*range(8)], rand=True, seed=15)
        group1 = [0, 1, 2, 3]
        for i in range(8):
            print(i, comp.getLatentScore(i))
        print(comp.max(group1))
        print(comp.min(group1))
        group2 = [4, 5, 6, 7]
        print(comp.max(group2))
        print(comp.min(group2))
        print(comp.counts)
    elif test == 5:
        comp = Comparator([*range(8)], rand=False)
        print(list(comp.minSeps.items()))
        print(comp.min([3, 7]))
        print(comp.compHistory)
        print(list(comp.minSeps.items()))
        print(comp.min([1, 5]))
        print(comp.compHistory)
        print(list(comp.minSeps.items()))
        print(comp.min([2, 6]))
        print(comp.compHistory)
        print(list(comp.minSeps.items()))
        print(comp.min([3, 7]))
        print(comp.compHistory)
        print(list(comp.minSeps.items()))
        print(comp.min([2, 6]))
        print(comp.compHistory)
        print(list(comp.minSeps.items()))
