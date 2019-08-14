import numpy as np
from sys import argv
from scipy.stats import norm
from DylMath import *
from DylMerger import MultiMerger
from tqdm import tqdm
def genD0D1(d0d1: list, arr: list) -> tuple:
	"""Generates filtered D0 and D1 vectors.
	d0d1 is (D0, D1) together as a tuple/list."""
	D0, D1 = list(), list()
	for item in arr:
		if item in d0d1[0]:
			D0.append(item)
		elif item in d0d1[1]:
			D1.append(item)
	return D0, D1

def validate(arr: list):
	"""Throws errors if the array to be validated is invalid.
	Invalid is if there is a -1 in the array or if there are duplicates."""
	if -1 in arr:
		raise FloatingPointError("it didn't actually do it")
	for v in arr:
		if arr.count(v) > 1:
			raise EnvironmentError(f"duplicated {v}")

def runStats(groups: list, params: list, comp=None) -> list:
	"""Runs stats on the groups provided.
	Params must be: ((d0d1), dist, targetAUC, n, currLayer, len(mergers))"""

	aucs, varOfSM, hanleyMcNeils, estimates = list(), list(), list(), list()
	d0d1, dist, targetAUC, n, currLayer, *_ = params
	rocs: list = list()
	for group in groups:
		D0, D1 = genD0D1(d0d1, group)
		if D0 and D1:
			rocs.append(genROC(group, D0, D1))
			sm: np.ndarray = successMatrix(group, D0, D1)
			auc: float = np.mean(sm)
			if auc == auc:
				aucs.append(auc)
			hanleyMcNeils.append((len(D0), len(D1)))
			smVAR: float = unbiasedMeanMatrixVar(sm)
			if smVAR == smVAR and len(D0) > 3 and len(D1) > 3: # if not NaN
				varOfSM.append(smVAR)
	rocs: list = list(filter(lambda roc: np.min(np.isfinite(roc)), rocs))
	varOfAverageAUC = np.var(aucs, ddof=1) / len(aucs)
	aucs: np.ndarray = np.array(aucs)
	avgAUC: float = np.mean(aucs)
	estimateNs: list = [list()]
	for ns in hanleyMcNeils:
		estimateNs[0].append(ns)
	# while there are groups to 'merge'
	while len(estimateNs[-1]) != 1:
		# get the previous layer and sort by N0 + N1
		oldNs: list = sorted(estimateNs[-1], key=sum)
		# roughly the same code as mergers creation
		estimateNs.append(list())
		while oldNs:
			i: int = 0
			toMerge: list = list()
			segments: int = min(n, len(oldNs) - i)
			for _ in range(segments):
				toMerge.append(oldNs.pop(0))
			estimateNs[-1].append([sum((x[0] for x in toMerge)), sum((x[1] for x in toMerge))])
		estimateNs[-1].sort(key=sum)
		estimates.append(hanleyMcNeil(avgAUC, estimateNs[-1][-1][0], estimateNs[-1][-1][1]) / len(estimateNs[-1]))
	for i, (N0, N1) in enumerate(hanleyMcNeils):
		hanleyMcNeils[i]: float = hanleyMcNeil(avgAUC, N0, N1)
	if len(varOfSM) == 0:
		varEstimate: float = float(varOfAverageAUC)
	else:
		varEstimate: float = (sum(varOfSM) / (len(varOfSM)**2))

	avgROC: tuple = avROC(rocs)
	empiricROC: tuple = comp.empiricROC()
	sep: float = genSep(dist, float(targetAUC)) # float in case it's a string
	
	stats: list = [avgAUC, varEstimate, sum(hanleyMcNeils) / len(hanleyMcNeils)**2, estimates, *MSE(sep, dist, avgROC, empiricROC)[:2]]

	return stats

def mergeSort(arr: list, statParams: list=None, n: int=2, combGroups: bool=True, sortGroups: bool=False) -> list:
	"""mergeSort(arr: list)
	statParams must be the format ((D0, D1), dist, target AUC)
	combGroups determins if the returned array is one list or each group as its own list.
	sortGroups determins if groups will be sorted by size in the sort.
	yields the arr after each pass through
	also yields the stats if given statParams"""
	groups: list = list([arr[i]] for i in range(len(arr)))
	mergers: list = []
	currLayer: int = -1
	nLayers: int = calcNLayers(arr) - 1
	# while there are partitions
	while len(groups) != 1:
		currLayer += 1
		i: int = 0
		while len(groups) >= n:
			# last group, odd one out
			# get n arrays
			# feed the MultiMergers with them
			arrays: list = list()
			for iSegment in range(n):
				arrays.append(groups.pop(0))
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
			stats: list = runStats(groups, statParams + [n, layer, len(mergerss)], comp)
			yield arr, stats
		else:
			yield arr
def treeMergeSort(arr: list, comp, statParams=None, n: int=2, combGroups: bool=True):
	"""sorts an array with the provided comparator.
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
	test: int = int(arv[1]) if len(argv) > 1 else 1
	if test == 1:
		if len(argv) < 3:
			print("Usage:")
			print(f"{__file__} 1 <directory to save file into>")
		else:
			from DylComp import Comparator
			plt.rcParams["font.size"]: int = 10
			data, D0, D1 = continuousScale(32, 32)
			comp: Comparator = Comparator(data, rand=True)
			#arrays = [data[:]]
			print(data)
			for arr in treeMergeSort(data, comp):
				#arrays.append(arr)
				#print(arr)
				pass
			arrays: list = [arr]
			D0.sort(key=arr.index)
			D1.sort(key=arr.index)
			plt = graphROCs(arrays, True, D0=D0, D1=D1)
			ax = plt.gca()
			ax.set_title("")
			plt.title("")
			plt.gcf().suptitle("")
			plt.savefig(arv[2] + "/patches.pdf", bbox_inches = 'tight', pad_inches = 0)
	elif test == 2:
		from DylComp import Comparator
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
		if len(argv) < 3:
			print("Usage:")
			print(f"{__file__} 3 <overlapping? True/False>")
		else:
			from DylData import continuousScale
			from DylComp import Comparator
			import matplotlib
			matplotlib.use("QT4Agg")
			import matplotlib.pyplot as plt
			font: dict = {"size" : 24}
			matplotlib.rc("font", **font)
			data, D0, D1 = continuousScale(16, 16)
			comp: Comparator = Comparator(data, rand=True)
			comp.genRand(len(D0), len(D1), 7.72, "exponential")
			comps: list = list()
			rocs: list = list()
			overlapping: bool = argv[2] == "True"
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
				ax.plot(comp.empiricROC()['x'], comp.empiricROC()['y'], 'b-', lw=3, label="empiric")
				linestyle_tuple: list = [':', '--', '-.', (0, (1, 1)), (0, (1, 1, 1, 0))]
				ax.plot([], [], lw=0, label='Comparisons, AUC')
				for i, roc in enumerate(rocs):
					ax.plot(*zip(*roc), linestyle=linestyle_tuple[i], label=f"{comps[i]:03d}, {auc(list(roc)):0.4f}", lw=(i + 3))
				ax.legend()
				ax.set_ylim(top=1, bottom=0)
				ax.set_xlim(left=0, right=1)
				ax.set(xlabel="False Positive Fraction", ylabel="True Positive Fraction")
			plt.tight_layout()
			plt.show()
	elif test == 4:
		from DylComp import Comparator
		from DylData import continuousScale
		from tqdm import trange
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
		from random import shuffle
		import numpy as np
		import matplotlib.pyplot as plt
		from matplotlib.patches import Rectangle
		import matplotlib
		font: dict = {'size' : 24}
		matplotlib.rc('font', **font)
		power: int = 9
		length: int = int(2**power*(2/3))
		power += 1
		yStep: float = length / power
		print(yStep)
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
			shuffle(data)
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