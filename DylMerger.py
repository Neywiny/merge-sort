class MultiMerger:
	def __init__(self, groups: list, comp, start=0, stop=0, toggle:bool=True):
		groups = list(filter(lambda x: len(x) > 0, groups))
		self.groups = [group for group in groups]
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
		if len(self.groups) == 0:
			return True
		if self.left == True:
			group = [group[self.indecies[i]] for i, group in enumerate(self.groups)]
			res = self.comp.min(group)
			if res == 'done':
				return 'done'
			maxInd, maxVal = res
			self.output[self.OIndex] = maxVal
			self.indecies[maxInd] += 1
			self.OIndex += 1
		elif self.left == False:
			group = [group[self.indeciesRight[i]] for i, group in enumerate(self.groups)]
			res = self.comp.max(group)
			if res == 'done':
				return 'done'
			maxInd, maxVal = res
			self.output[self.OIndexRight] = maxVal
			self.indeciesRight[maxInd] -= 1
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
		self.left ^= self.toggle # toggle self.left if self.toggle == True
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
			if m.output != [0, 1, 2, 3, 4, 5, 6, 7]:
				raise AssertionError("wasn't right")
			comp.clearHistory()
			m1 = MultiMerger([[0,4], [1, 5]], comp, toggle=True)
			while not m1.inc():
				pass
			if m1.output != [0, 1, 4, 5]:
				raise AssertionError("wasn't right")
			m2 = MultiMerger([[2, 6], [3, 7]], comp, toggle=True)
			while not m2.inc():
				pass
			if m2.output != [2, 3, 6, 7]:
				raise AssertionError("wasn't right")
			m3 = MultiMerger([m1.output,m2.output], comp, toggle=True)
			while not m3.inc():
				pass
			if m3.output != m.output:
				raise AssertionError("wasn't right")
		elif test == 2:
			comp = Comparator([0, 1, 2, 3, 4, 5, 6, 7])
			m = MultiMerger([[0, 3, 5], [1, 2, 4]], comp, toggle=False)
			while not m.inc():
				pass
			if m.output != [0, 1, 2, 3, 4, 5]:
				raise AssertionError("wasn't right")
			m = MultiMerger([[0, 3, 5], [1, 2, 4]], comp, toggle=True)
			while not m.inc():
				pass
			if m.output != [0, 1, 2, 3, 4, 5]:
				raise AssertionError("wasn't right")
		elif test == 3:
			comp = Comparator([0, 1, 2, 3, 4, 5, 6, 7])
			m = MultiMerger([[0, 1, 2, 3, 5, 6, 7], [4]], comp, toggle=True)
			while not m.inc():
				pass
			if m.output != [0, 1, 2, 3, 4, 5, 6, 7]:
				raise AssertionError("wasn't right")