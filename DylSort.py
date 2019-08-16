#!/usr/bin/python3.6
import numpy as np
from sys import argv
from DylMath import runStats, graphROCs
from DylMerger import MultiMerger
from DylComp import Comparator
from DylData import continuousScale

def validate(arr: list):
	"""Throws errors if the array to be validated is invalid.
	Invalid is if there is a -1 in the array or if there are duplicates."""
	if -1 in arr:
		raise FloatingPointError("it didn't actually do it")
	for v in arr:
		if arr.count(v) > 1:
			raise EnvironmentError(f"duplicated {v}")

def mergeSort(arr: list, comp: Comparator, statParams: list=None, n: int=2, combGroups: bool=True, sortGroups: bool=False) -> list:
	"""MergeSort(arr: list)
	statParams must be the format ((D0, D1), dist, target AUC)
	combGroups determins if the returned array is one list or each group as its own list.
	sortGroups determins if groups will be sorted by size in the sort.
	yields the arr after each pass through
	also yields the stats if given statParams"""
	groups: list = list([arr[i]] for i in range(len(arr)))
	mergers: list = []
	currLayer: int = -1
	# while there are partitions
	while len(groups) != 1:
		currLayer += 1
		i: int = 0
		while len(groups) >= n:
			# last group, odd one out
			# get n arrays
			# feed the MultiMergers with them
			arrays: list = [groups.pop(0) for _ in range(n)]
			mergers.append(MultiMerger(arrays, comp, i, 0))
			i += 1
		#while we have active mergers
		while mergers:
			for merger in mergers:
				res = merger.inc()
				if res: #if that merger is done
					validate(merger.output)
					comp.learn(merger.output)
					if sortGroups:
						groups.append(merger.output)
					else:
						groups.insert(0, merger.output)
					mergers.remove(merger)
		if combGroups:
			arr: list = []
			for group in groups:
				arr.extend(group)
		else:
			arr: list = groups
		# run dem stats
		if statParams:
			stats: list = runStats(groups, statParams + [n, currLayer, len(mergerss)], comp)
			yield arr, stats
		else:
			yield arr
def treeMergeSort(arr: list, comp, statParams=None, n: int=2, combGroups: bool=True):
	"""Sorts an array with the provided comparator.
	statParams must be the format ((D0, D1), dist, target AUC) if it is provided.
	If n is provided, does at most nAFC type comparisons (ex if n=4, may do most 4AFC, maybe some 3AFC, rest 2AFC)
	combGroups determins if the returned array is one list or each group as its own list.
	Yields the current layer and the statistics if statParams is not None."""
	if n < 2:
		raise IOError("can't split a tree with n < 2")
	sizess: list = [[0, len(arr)]]
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
	mergerss: list = [[], []]
	i: int = 0
	while i < len(sizess[-1]) - 1:
		segments: int = min(n, len(sizess[-1]) - i)
		groups: list = list()
		for iSeg in range(segments):
			group: list = arr[sizess[-1][i + iSeg]:sizess[-1][i + iSeg + 1]]
			if len(group) != 1: # such that we need to add another layer to it
				merger = MultiMerger([[img] for img in group], comp)
				merger.left = bool(iSeg % 2)
				mergerss[0].append(merger)
				groups.append(mergerss[0][-1].output)
			else:
				groups.append(group)
		mergerss[1].append(MultiMerger(groups, comp))
		i += segments
	# now build up layers of mergerss where the groups are the outputs of the last layer
	while len(mergerss[-1]) > 1: # while not on top level
		mergerss.append(list())
		i: int = 0
		while (segments == min(n, len(mergerss[-2]) - i)) != 0:
			groups: list = list()
			for iSeg in range(segments):
				groups.append(mergerss[-2][i + iSeg].output)
			mergerss[-1].append(MultiMerger(groups, comp))
			i += segments

	left: bool = True
	for layer, mergers in enumerate(mergerss, start=1):
		done: int = 0
		groups: list = list()
		while mergers:
			for merger in mergers if left else reversed(mergers):
				res = merger.inc()
				if res == True:
					validate(merger.output)
					groups.append(merger.output)
					mergers.remove(merger)
					done += 1
				elif res == 'done':
					raise StopIteration()
		left ^= 1 #flip left
		if combGroups:
			arr: list = []
			for group in groups: arr.extend(group)
		else:
			arr: list = groups
		yield (arr, runStats(groups, statParams + [n, layer, len(mergerss)], comp)) if statParams else arr

if __name__ == "__main__":
	test: int = int(argv[1]) if len(argv) > 1 else 1
	if test == 1:
		if len(argv) > 5:
			print("Usage:")
			print(f"{__file__} 1 <n0> <n1> <directory to save file into (optional)>")
		else:
			import matplotlib.pyplot as plt
			plt.rcParams["font.size"]: int = 10
			data, D0, D1 = continuousScale(int(argv[2]), int(argv[3]))
			comp: Comparator = Comparator(data, rand=True)
			for arr in treeMergeSort(data, comp):
				pass
			arrays: list = [arr]
			D0.sort(key=arr.index)
			D1.sort(key=arr.index)
			plt = graphROCs(arrays, True, D0=D0, D1=D1)
			ax = plt.gca()
			ax.set_title("")
			plt.title("")
			plt.gcf().suptitle("")
			if len(argv) > 4:
				plt.savefig(argv[4] + "/patches.pdf", bbox_inches = 'tight', pad_inches = 0)
			else:
				plt.show()
	elif test == 2:
		print("treeMergeSort")
		for n in range(2, 18):
			data: list = [*reversed(range(197))]
			comp: Comparator = Comparator(data, level=0, rand=False)
			for _ in treeMergeSort(data, comp, n=n):
				pass
			print(n, len(comp))
		print("regular mergeSort")
		for n in range(2, 18):
			data: list = [*reversed(range(197))]
			comp: Comparator = Comparator(data, level=0, rand=False)
			for _ in mergeSort(data, comp, n=n):
				pass
			print(n, len(comp))
	elif test == 3:
		if len(argv) > 3:
			print("Usage:")
			print(f"{__file__} 3 <overlapping? True/False>")
		else:
			import matplotlib
			matplotlib.use("QT4Agg")
			import matplotlib.pyplot as plt
			from DylMath import genROC, avROC, auc
			data, D0, D1 = continuousScale(16, 16)
			comp: Comparator = Comparator(data, rand=True)
			comp.genRand(len(D0), len(D1), 7.72, "exponential")
			comps: list = list()
			rocs: list = list()
			overlapping: bool = argv[2] == "True" if len(argv) == 3 else True
			for groups in treeMergeSort(data, comp, combGroups=False):
				rocs.append(list())
				comps.append(len(comp))
				arr: list = []
				for group in groups:
					arr.extend(group)
					rocs[-1].append(genROC(group, D0, D1))
				rocs[-1]: tuple = list(zip(*avROC(rocs[-1])))
				#rocs[-1].reverse()
				#print(comp.kendalltau(arr), MSE(7.72, rocs[-1], comp.empiricROC()))
			if not overlapping:
				rows: int = int(np.ceil(np.sqrt(len(rocs))))
				cols: int = int(np.ceil(len(rocs) / rows))
				fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, num="plots")
				fig.suptitle("ROC Curves")
				for i,ax in enumerate(axes.flat):
					if i >= len(rocs):
						continue
					ax.set_aspect('equal', 'box')
					ax.set(xlabel="FPF", ylabel="TPF")
					ax.label_outer()
					ax.plot((0,1),(0,1),c='red', linestyle=":")
					ax.plot(*zip(*rocs[i]), c='blue')
					ax.set_ylim(top=1.02, bottom=0)
					ax.set_xlim(left=-0.01, right=1)
					ax.set_title(f"Iteration #{i + 1} AUC: {auc(rocs[i]):.5f}")
			else:
				font: dict = {"size" : 24}
				matplotlib.rc("font", **font)
				fig = plt.figure(figsize=(8, 8))
				plt.title("ROC Curves")
				ax = fig.add_subplot(1, 1, 1)
				ax.plot(comp.empiricROC()['x'], comp.empiricROC()['y'], 'b-', lw=3, label="empiric")
				linestyle_tuple: list = [':', '--', '-.', (0, (1, 1)), (0, (1, 1, 1, 0))]
				ax.plot([], [], lw=0, label='Comparisons, AUC')
				for i, roc in enumerate(rocs):
					ax.plot(*zip(*roc), linestyle=linestyle_tuple[i], label=f"{comps[i]:03d}, {auc(list(roc)):0.4f}", lw=(i + 3))
				ax.set_aspect('equal', 'box')
				ax.legend()
				ax.set_ylim(top=1, bottom=0)
				ax.set_xlim(left=0, right=1)
				ax.set(xlabel="False Positive Fraction", ylabel="True Positive Fraction")
			plt.tight_layout()
			plt.show()
	elif test == 4:
		data, D0, D1 = continuousScale(135, 87)
		comp: Comparator = Comparator(data, rand=True)
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

	elif test == 5:
		import matplotlib.pyplot as plt
		from matplotlib.patches import Rectangle
		import matplotlib
		font: dict = {'size' : 24}
		matplotlib.rc('font', **font)
		power: int = 9
		length: int = int(2**power*(2/3))
		power += 1
		yStep: float = length / power
		fig, axes = plt.subplots(ncols=2, nrows=1)
		for ax in axes:
			if ax == axes[0]:
				color: str = 'r'
				sorter = mergeSort
			else:
				color: str = 'lime'
				sorter = treeMergeSort
			data: list = list(range(int(2**(power - 1)*(2/3))))
			img: np.ndarray = np.zeros((power, length))
			np.random.shuffle(data)
			img[0]: list = data[:]
			comp: Comparator = Comparator(data, level=0)

			for gIndex, group in enumerate(data):
				ax.add_patch(Rectangle((gIndex, (power) * yStep), 1, -yStep, color=color, lw=1/power, fill=False))

			for y, groups in enumerate(sorter(data, comp=comp, combGroups=False), start=1):
				arr: list = []
				x: int = 0
				for group in groups:
					ax.add_patch(Rectangle((x, (power - y) * yStep), len(group), -yStep, color=color, lw=3*y/power, fill=False))
					arr.extend(group)
					x += len(group)
				if len(arr) < len(img[0]):
					arr.extend([0 for i in range(len(img[0]) - len(arr))])
				img[y]: list = arr
			ax.imshow(img, cmap='Greys', extent=[0, length, 0, length], aspect=1)
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_ylabel("Layer")
			ax.set_xlabel("Position")
		plt.show()