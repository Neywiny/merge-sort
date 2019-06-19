
class Merger:
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
