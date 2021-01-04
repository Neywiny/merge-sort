#!/usr/bin/python3.6
class MultiMerger:
	def __init__(self, groups: list, comp, start=0, stop=0, toggle:bool=True):
		"""Initializes a Merger object mergning on the groups parameter.
		
		start and stop are not needed.
		toggle determines if it merges from both ends."""
		groups: list = list(filter(lambda x: len(x) > 0, groups))
		self.groups: list = [group for group in groups]
		self.comp = comp
		self.start: int = start
		self.stop: int = stop
		self.toggle: bool = toggle
		self.indecies: list = [0 for group in groups]
		self.indeciesRight: list = [len(group) - 1 for group in groups]
		self.output: list = [-1 for i in range(sum([len(group) for group in groups]))]
		self.OIndex: int = 0
		self.OIndexRight = len(self.output) - 1
		self.left: bool = True
	def inc(self) -> bool:
		if len(self.groups) == 0:
			return True
		if self.left == True:
			group: list = [group[self.indecies[i]] for i, group in enumerate(self.groups)]
			res = self.comp.min(group)
			if res == 'done':
				return 'done'
			minInd, minVal = res
			self.output[self.OIndex] = minVal
			self.indecies[minInd] += 1
			self.OIndex += 1
		elif self.left == False:
			group: list = [group[self.indeciesRight[i]] for i, group in enumerate(self.groups)]
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

def MultiMergerGenerator(groups: list, comp, start=0, stop=0, toggle:bool=True, left:bool=True):
	"""A generator version of the MultiMerger class.

	Not in use due to the inability to see what's currently inside a generator object."""
	groups: list = list(filter(lambda x: len(x) > 0, groups))
	indecies: list = [0 for group in groups]
	indeciesRight: list = [len(group) - 1 for group in groups]
	output: list = [-1 for i in range(sum([len(group) for group in groups]))]
	OIndex: int = 0
	OIndexRight: int = len(output) - 1
	while not (OIndex == len(output) or len(groups) == 0):
		if len(groups) == 0:
			yield output
		if left == True:
			group: list = [group[indecies[i]] for i, group in enumerate(groups)]
			res: int = comp.min(group)
			if res == 'done':
				yield 'done'
			minInd, minVal = res
			output[OIndex] = minVal
			indecies[minInd] += 1
			OIndex += 1
		elif left == False:
			group: list = [group[indeciesRight[i]] for i, group in enumerate(groups)]
			res = comp.max(group)
			if res == 'done':
				yield 'done'
			maxInd, maxVal = res
			output[OIndexRight] = maxVal
			indeciesRight[maxInd] -= 1
			OIndexRight -= 1
		# check to see if that group is doneski
		for i, group in enumerate(groups):
			if indecies[i] > indeciesRight[i]:
				groups.pop(i)
				indecies.pop(i)
				indeciesRight.pop(i)
		if len(groups) == 1:
			for i in range(indecies[0], indeciesRight[0] + 1):
				output[OIndex] = groups[0][i]
				OIndex += 1
			yield output
		left ^= toggle # toggle left if toggle == True
		yield False

if __name__ == '__main__':
	from DylComp import Comparator
	for test in range(1, 7):
		if test == 1:
			objs: list = [*range(8)]
			comp: Comparator = Comparator(objs, rand=False)
			m: MultiMerger = MultiMerger([[0,4], [1, 5], [2, 6], [3, 7]], comp, toggle=True)
			while not m.inc():
				pass
			if m.output != [0, 1, 2, 3, 4, 5, 6, 7]:
				raise AssertionError("wasn't right")
			comp.clearHistory()
			m1: MultiMerger = MultiMerger([[0,4], [1, 5]], comp, toggle=True)
			while not m1.inc():
				pass
			if m1.output != [0, 1, 4, 5]:
				raise AssertionError("wasn't right")
			m2: MultiMerger = MultiMerger([[2, 6], [3, 7]], comp, toggle=True)
			while not m2.inc():
				pass
			if m2.output != [2, 3, 6, 7]:
				raise AssertionError("wasn't right")
			m3: MultiMerger = MultiMerger([m1.output,m2.output], comp, toggle=True)
			while not m3.inc():
				pass
			if m3.output != m.output:
				raise AssertionError("wasn't right")
		elif test == 2:
			comp: Comparator = Comparator([0, 1, 2, 3, 4, 5, 6, 7])
			m: MultiMerger = MultiMerger([[0, 3, 5], [1, 2, 4]], comp, toggle=False)
			while not m.inc():
				pass
			if m.output != [0, 1, 2, 3, 4, 5]:
				raise AssertionError("wasn't right")
			m: MultiMerger = MultiMerger([[0, 3, 5], [1, 2, 4]], comp, toggle=True)
			while not m.inc():
				pass
			if m.output != [0, 1, 2, 3, 4, 5]:
				raise AssertionError("wasn't right")
		elif test == 3:
			comp: Comparator = Comparator([0, 1, 2, 3, 4, 5, 6, 7])
			m: MultiMerger = MultiMerger([[0, 1, 2, 3, 5, 6, 7], [4]], comp, toggle=True)
			while not m.inc():
				pass
			if m.output != [0, 1, 2, 3, 4, 5, 6, 7]:
				raise AssertionError("wasn't right")
		if test == 4:
			objs: list = [*range(8)]
			comp: Comparator = Comparator(objs, rand=False)
			m = MultiMergerGenerator([[0,4], [1, 5], [2, 6], [3, 7]], comp, toggle=True)
			res = next(m)
			while not res:
				res = next(m)
			if res != [0, 1, 2, 3, 4, 5, 6, 7]:
				raise AssertionError("wasn't right")
			comp.clearHistory()
			m1 = MultiMergerGenerator([[0,4], [1, 5]], comp, toggle=True)

			res1 = next(m1)
			while not res1:
				res1 = next(m1)
			if res1 != [0, 1, 4, 5]:
				raise AssertionError("wasn't right")
			m2 = MultiMergerGenerator([[2, 6], [3, 7]], comp, toggle=True)
			res2 = next(m2)
			while not res2:
				res2 = next(m2)
			if res2 != [2, 3, 6, 7]:
				raise AssertionError("wasn't right")
			m3= MultiMergerGenerator([res1, res2], comp, toggle=True)
			res3 = next(m3)
			while not res3:
				res3 = next(m3)
			if res != res3:
				raise AssertionError("wasn't right")
		elif test == 5:
			comp: Comparator = Comparator([0, 1, 2, 3, 4, 5, 6, 7])
			m = MultiMergerGenerator([[0, 3, 5], [1, 2, 4]], comp, toggle=False)
			res = next(m)
			while not res:
				res = next(m)
			if res != [0, 1, 2, 3, 4, 5]:
				raise AssertionError("wasn't right")
			m = MultiMergerGenerator([[0, 3, 5], [1, 2, 4]], comp, toggle=True)
			res = next(m)
			while not res:
				res = next(m)
			if res != [0, 1, 2, 3, 4, 5]:
				raise AssertionError("wasn't right")
		elif test == 6:
			comp: Comparator = Comparator([0, 1, 2, 3, 4, 5, 6, 7])
			m = MultiMergerGenerator([[0, 1, 2, 3, 5, 6, 7], [4]], comp, toggle=True)
			res = next(m)
			while not res:
				res = next(m)
			if res != [0, 1, 2, 3, 4, 5, 6, 7]:
				raise AssertionError("wasn't right")