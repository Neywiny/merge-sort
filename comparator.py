class Comparator:
    def __init__(self, objects: list = None, optimizing: bool = False):
        self.clearHistory()
        self.optimizing = optimizing
        if objects != None:
            self.genLookup(objects)
    def compare(self, a, b):
        try:
            if b in self.lookup[a]:
                return self.lookup[a][b]
            elif a in self.lookup[b]:
                return self.lookup[b][a]
            else:
                res = a < b
                self.compHistory.append((a,b))
                #print(a,b)
                self.lookup[a][b] = res
                self.lookup[b][a] = not res

                if self.optimizing:
                    #single optimization
                    for c in self.lookup[b].keys():
                        # for all c s.t. we know the relationship b/t b and c
                        if self.lookup[b][c] == res:#a < b < c or a > b > c
                            self.lookup[a][c] = res  #a < c or a > c 
                            self.lookup[c][a] = not res
                return res
        except AttributeError as e:
            raise LookupError("You need to generate the lookup first")
    def genLookup(self, objects: list):
        self.lookup = dict()
        self.objects = objects
        for object in objects:
            self.lookup[object] = dict()
    def clearHistory(self):
        self.compHistory = list()

if __name__ == "__main__":
    comp = Comparator([i for i in range(10)])
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