class Comparator:
    def __init__(self, objects: list = None, level: int = 2):
        self.clearHistory()
        self.level:int = level
        if objects != None:
            self.genLookup(objects)
    def compare(self, a, b) -> bool:
        """returns a < b
        based on optimization level, will not optimize, store result, or do abc association"""
        try:
            if b in self.lookup[a].keys():
                return self.lookup[a][b]
            elif a in self.lookup[b].keys():
                return self.lookup[b][a]
            else:
                res:bool = a < b
                self.compHistory.append((a,b))
                #print(a,b)
                if self.level > 0:
                    self.lookup[a][b]:bool = res
                    self.lookup[b][a]:bool = not res

                if self.level == 2:
                    #single optimization
                    for c in self.lookup[b].keys():
                        # for all c s.t. we know the relationship b/t b and c
                        if self.lookup[b][c] == res:#a < b < c or a > b > c
                            self.lookup[a][c]:bool = res  #a < c or a > c 
                            self.lookup[c][a]:bool = not res
                if self.level > 2:
                    #O(n!) optimization. Make sure to use a copy of objects
                    Comparator.optimize(list(self.lookup.keys()), self.lookup, res, a, b)
                return res
        except AttributeError as e:
            raise LookupError("You need to generate the lookup first")
    def genLookup(self, objects: list):
        self.lookup:dict = dict()
        self.objects:list = objects
        for object in objects:
            self.lookup[object] = dict()
    
    def clearHistory(self):
        self.compHistory = list()

    @staticmethod
    def optimize(objects: list, lookup: dict, res: bool, a, b):
        if objects:
            nObjects:list = []
            for c in list(lookup[b]): 
                #for all c s.t. c is a neighbor of b
                if c in objects and lookup[b][c] == res and c != a: 
                    #s.t. a > b > c or a < b < c
                    nObjects.append(c)
                    lookup[a][c]:bool = res
                    lookup[c][a]:bool = not res
                    Comparator.optimize(nObjects, lookup, res, b, c)

if __name__ == "__main__":
    comp = Comparator([i for i in range(10)], 3)
    print(    comp.compare(1, 2))
    print(not comp.compare(2, 1))
    print(    comp.compare(3, 4))
    print(not comp.compare(4, 3))
    print(    comp.compare(5, 6))
    print(not comp.compare(6, 5))
    print(    comp.compare(7, 8))
    print(not comp.compare(8, 4))
    print(    comp.compare(6, 7))
    print(not comp.compare(7, 6))