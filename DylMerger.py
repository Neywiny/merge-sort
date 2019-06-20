
class Merger(MultiMerger):
    """A class for merging 2 arrays into 1 array
    Must feed it a comparator object. It can also hold onto start/stop indecies if you ask
    'Toggle' parameter defines if you want to do a cocktail merge, as in switch which end you're merging from each call of inc()
    """
    def __init__(self, groupA: list, groupB: list, comp, start=0, stop=0, toggle:bool=True):
        self.groupA: list = groupA
        self.groupB: list = groupB
        comp.learn(groupA)
        comp.learn(groupB)
        self.output: list = [-1 for i in [*groupA, *groupB]]
        self.indexA: int = 0
        self.indexB: int = 0
        self.outIndex: int = 0
        self.indexARight: int = len(groupA) - 1
        self.indexBRight: int = len(groupB) - 1
        self.indexORight: int = len(self.output) - 1
        self.comp = comp
        self.left: bool = True
        self.toggle: bool = toggle

        #hold onto these for the parent object, they're a secret tool that will help us later
        self.start = start
        self.stop = stop
    
    def inc(self) -> bool:
        """do a merge, and if configured, switch which end the next merge will be done from
        returns True if the merge is completed"""
        if self.indexB >= len(self.groupB) and self.left != -3:
            while self.indexA <= self.indexARight:
                self.output[self.outIndex] = self.groupA[self.indexA]
                self.outIndex += 1
                self.indexA += 1
            return True
        if (self.indexA >= len(self.groupA) or self.indexARight < 0) and self.left != -3:
            while self.indexB <= self.indexBRight:
                self.output[self.outIndex] = self.groupB[self.indexB]
                self.outIndex += 1
                self.indexB += 1
            return True
        
        if self.left == True:
            iA = self.indexA
            iB = self.indexB
            iO = self.outIndex
            # if the element from A is less than the element from B
            if self.comp(self.groupA[iA], self.groupB[iB]):
                self.output[iO] = self.groupA[iA]
                self.indexA += 1
            else:
                self.output[iO] = self.groupB[iB]
                self.indexB += 1
            self.outIndex += 1
        elif self.left == False:
            iA = self.indexARight
            iB = self.indexBRight
            iO = self.indexORight

            if iB < 0: # done with group B
                while self.indexARight >= self.indexA:
                    self.output[self.outIndex] = self.groupA[self.indexA]
                    self.indexA += 1
                    self.outIndex += 1
                return True

            # if the element from A is less than the element from B
            if self.comp(self.groupA[iA], self.groupB[iB]):
                self.output[iO] = self.groupB[iB]
                self.indexBRight -= 1
            else:
                self.output[iO] = self.groupA[iA]
                self.indexARight -= 1
            self.indexORight -= 1
        else: # this signals no need to compare
            return True

        # go from other side
        self.left = self.toggle ^ self.left
        return (self.indexA > self.indexARight) and (self.indexB > self.indexBRight)
        #return (self.outIndex == len(self.output)) or (self.outIndex == self.indexORight)

class MultiMerger:
    def __init__(self, groups: list, comp, start=0, stop=0, toggle:bool=True):
        self.groups = [group[:] for group in groups]
        self.comp = comp
        self.start = start
        self.stop = stop
        self.toggle = toggle
        self.indecies = [0 for group in groups]
        self.indeciesRight = [len(group) - 1 for group in groups]
        self.output = [-1 for i in range(sum([len(group) for group in groups]))]
        self.OIndex = 0
        self.OIndexRight = len(self.output) - 1
        self.left = True

    def inc(self) -> bool:
        if self.left == True:
            group = [group[self.indecies[i]] for i, group in enumerate(self.groups)]
            gI, iI = self.comp.min(group)
            self.output[self.OIndex] = iI
            self.indecies[gI] += 1
            self.OIndex += 1
        elif self.left == False:
            group = [group[self.indeciesRight[i]] for i, group in enumerate(self.groups)]
            gI, iI = self.comp.max(group)
            self.output[self.OIndexRight] = iI
            self.indeciesRight[gI] -= 1
            self.OIndexRight -= 1


        # check to see if that group is doneski
        for i, group in enumerate(self.groups):
            if self.indecies[i] > self.indeciesRight[i]:
                self.groups.pop(i)
                self.indecies.pop(i)
                self.indeciesRight.pop(i)
        
        if len(self.groups) == 1:
            for i in range(self.indecies[0], self.indeciesRight[0] + 1):
                self.output[self.OIndex] = self.groups[0][i]
                self.OIndex += 1
            return True

        self.left ^= self.toggle
        return self.OIndex == len(self.output) or len(self.groups) == 0

if __name__ == '__main__':
    from DylComp import Comparator
    for test in range(1, 4):
        if test == 1:
            objs = [*range(8)]
            comp = Comparator(objs, rand=False)
            m = MultiMerger([[0,4], [1, 5], [2, 6], [3, 7]], comp, toggle=True)
            while not m.inc():
                pass
            assert m.output == [0, 1, 2, 3, 4, 5, 6, 7]
            comp.clearHistory()
            m1 = MultiMerger([[0,4], [1, 5]], comp, toggle=True)
            while not m1.inc():
                pass
            assert m1.output == [0, 1, 4, 5]
            m2 = MultiMerger([[2, 6], [3, 7]], comp, toggle=True)
            while not m2.inc():
                pass
            assert m2.output == [2, 3, 6, 7]
            m3 = MultiMerger([m1.output,m2.output], comp, toggle=True)
            while not m3.inc():
                pass
            assert m3.output == m.output

        elif test == 2:
            comp = Comparator([0, 1, 2, 3, 4, 5, 6, 7])
            m = MultiMerger([[0, 3, 5], [1, 2, 4]], comp, toggle=False)
            while not m.inc():
                pass
            assert m.output == [0, 1, 2, 3, 4, 5]
            m = MultiMerger([[0, 3, 5], [1, 2, 4]], comp, toggle=True)
            while not m.inc():
                pass
            assert m.output == [0, 1, 2, 3, 4, 5]

        elif test == 3:
            comp = Comparator([0, 1, 2, 3, 4, 5, 6, 7])
            m = MultiMerger([[0, 1, 2, 3, 5, 6, 7], [4]], comp, toggle=True)
            while not m.inc():
                pass
            assert m.output == [0, 1, 2, 3, 4, 5, 6, 7]