import ROC1
import numpy as np
np.set_printoptions(threshold=np.inf)
np.seterr(all="ignore")
from tqdm import tqdm
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.stats import norm
#from p_tqdm import p_map
try:
	import matplotlib
	matplotlib.use('QT4Agg')
	import matplotlib.pyplot as plt
	font: dict = {'size' : 56}
	#matplotlib.rc('font', **font)
	from matplotlib.collections import PatchCollection
	from matplotlib.patches import Rectangle
except BaseException as e:
	pass
from DylData import *
unbiasedMeanMatrixVar = ROC1.unbiasedMeanMatrixVar
def paramToParams(predicted: list, D0: list=None, D1: list=None) -> (list, list, list):
	"""Takes one parameter and splits it into three if predicted is a 2d list"""
	if isinstance(predicted[0], (list, tuple)):
		return predicted[0], predicted[1], predicted[2]
	else:
		return predicted, D0, D1

def auc(results: tuple, D0: list=None, D1: list=None) -> float:
	""" Takes an ROC curve from genROC and returns the AUC.
	If results is a prediction not an ROC curve, generates the ROC curve."""
	if not isinstance(results[0], (list, tuple)):
		results: list = genROC(results, D0, D1)
	total: float = 0.0
	for i,(x,y) in enumerate(results[:-1], start=1):
		# start=1 means i is actually i + 1
		total += 0.5*(y + results[i][1]) * (x - results[i][0])
	return -total

def hanleyMcNeil(auc: float, n0: int, n1: int) -> float:
	"""The very good power-law variance estimate from Hanley/McNeil"""
	auc2=auc*auc
	q1=auc/(2.-auc)
	q2=2.*auc2/(1.+auc)
	return( (auc-auc2+(n1-1.)*(q1-auc2)+(n0-1.)*(q2-auc2))/n0/n1 )

def calcNLayers(arr: list) -> int:
	"""Returns the number of layers that would be needed to sort.
	If arr is the a tuple or list, uses the length.
	If arr is already the length, uses that."""
	if isinstance(arr, int):
		length: int = arr
	else:
		length: int = len(arr)
	return np.ceil(np.log2(length))

def genSep(dist: str, auc: float) -> float:
	"""Returns the sep parameter needed for the target AUC for the given distribution."""
	if dist == 'exponential':
		return abs(auc/(1-auc))
	elif dist == 'normal':
		return norm.ppf(auc)*(2**0.5)
	raise NotImplementedError("Cannot gen sep for that distribution")

def MSE(sep: float, dist: str, ROC: list, rocEmpiric: list=None) -> (float, float, float):
	"""Returns the MSE of the given ROC with respect to:
	If sep and dist are not None: the true ROC from sep and dist
	If rocEmpiric is not None: the MSE between the Empiric and ROC
	If sep and dist are None, the first value returned is always 0
	The last value returned is always the AUC of the ROC"""
	step: float = 10**-4
	fpf = np.arange(0, 1, step)
	if len(ROC) == 2:
		approx = interp1d(*((ROC['x'], ROC['y']) if isinstance(ROC, dict) else ROC))(fpf)
	else:
		approx = interp1d(*zip(*ROC))(fpf)
	if dist == 'exponential':
		mseTrue: float = np.mean((approx - (fpf**(1/sep)))**2)
	elif dist == 'normal':
		mseTrue: float = np.mean((approx - (1-norm.cdf(norm.ppf(1-fpf) - sep)))**2)
	else:
		mseTrue: float = 0.0
	if rocEmpiric != None:
		if len(rocEmpiric) == 2:
			trueApprox = interp1d(rocEmpiric['x'], rocEmpiric['y'])
		else:
			trueApprox = interp1d(*zip(*rocEmpiric))
		mseEmpiric: float = np.mean((approx - (trueApprox(fpf)))**2)
	calcAUC: float = np.trapz(approx) / (1/step)
	return (mseTrue, calcAUC) if rocEmpiric == None else (mseTrue, mseEmpiric, calcAUC)

def genX0X1(predicted: tuple, D1: tuple=None, D0: tuple=None) -> (list, list):
	"""Generates x0 and x1 vectors out of the given parameters.
	D1 and D0 should never be smaller than the predicted array, but are often bigger."""
	predicted, D0, D1 = paramToParams(predicted, D0, D1)
	x0, x1 = genD0D1((D0, D1), predicted)
	return np.array(x0), np.array(x1)

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

def genROC(predicted: tuple, D1: list=None, D0: list=None) -> list:
	"""Returns a list of collections of x,y coordinates in order of the threshold"""
	x0, x1 = genX0X1(predicted, D1, D0)
	roc: dict = ROC1.rocxy(x1, x0)
	return list(zip(roc['x'], roc['y']))

def graphROC(predicted: tuple, D0: list=None, D1: list=None):
	"""Generates and graphs a single ROC curve and displays the results."""
	predicted, D0, D1 = paramToParams(predicted, D0, D1)
	fig = plt.figure(figsize=(4,4))
	ax = fig.add_subplot(111)
	ax.plot(*zip(*genROC(predicted, D0, D1)))
	ax.plot((0,1),(0,1),c="r", linestyle="--")
	ax.set_ylim(top=1.1,bottom=-0.1)
	ax.set_xlim(left=-0.1,right=1.1)
	ax.set_title(f"AUC: {auc(predicted, D0, D1):.5f}")
	ax.set_xlabel("FPF")
	ax.set_ylabel("TPF")
	plt.show()

def graphROCs(arrays: list, withPatches: bool=False, withLine: bool=True, D0: list=None, D1: list=None):
	"""Graphs a collection of array predictions. Takes the arrays as they would come out of DylSort sorts.
	If withPatches, puts a color coded success matrix behind the line.
	If withLine, graphs the line.
	Returns the plt handle, does not display the results."""
	rows: int = int(np.ceil(np.sqrt(len(arrays))))
	cols: int = int(np.ceil(len(arrays) / rows))
	fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, num="plots")
	fig.suptitle("ROC curves")
	if withLine:
		params: list = [(array, D0, D1) for array in arrays]
		if len(arrays[0]) < 1024:
			results: list = list(map(genROC, params))
		else:
			with Pool() as p:
				results: list = list(p.imap(genROC,params))
	if withPatches:
		pbar = tqdm(total=len(arrays)*(len(arrays[0])//2)**2)
	for i, ax in enumerate(axes.flat if (rows * cols > 1) else [axes]):
		if i >= len(arrays):
			continue
		ax.set(xlabel="False Positive Fraction", ylabel="True Positive Fraction")
		ax.label_outer()
		ax.plot((0,1),(0,1),c='red', linestyle=":")
		if withLine:
			ax.plot(*zip(*results[i]), c='blue')
			ax.set_ylim(top=1.02, bottom=0)
			ax.set_xlim(left=-0.01, right=1)
			if not withPatches:
				ax.set_title(f"Iteration #{i} AUC: {auc(results[i]):.5f}")
		if withPatches:
			sm: np.ndarray = successMatrix(arrays[i], D0, D1)
			yes: list = []
			no: list = []
			yLen: int = len(D1)
			xLen: int = len(D0)
			for (y,x), value in np.ndenumerate(sm):
				if value:
					yes.append(Rectangle((x/xLen,y/yLen),1/xLen,1/yLen))
				else:
					no.append(Rectangle((x/xLen,y/yLen),1/xLen,1/yLen))
			pbar.update(1)
			patches = PatchCollection(no, facecolor = 'r', alpha=0.75, edgecolor='None')
			ax.add_collection(patches)
			patches = PatchCollection(yes, facecolor = 'g', alpha=0.75, edgecolor='None')
			ax.add_collection(patches)
			area = len(yes) / (len(yes) + len(no))
			ax.set_ylim(top=1, bottom=0)
			ax.set_xlim(left=0, right=1)
			ax.set_title(f"Iteration #{i} AUC: {area:.5f}")
			ax.set_aspect('equal', 'box')
	if withPatches:
		pbar.close()
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	#plt.show()
	return plt

def avROC(rocs: list) -> tuple:
	""" Averages ROC curves. Rocs parameter are ROC curves from genROC."""
	#hard coded SeSp
	#e = 9*sys.float_info.epsilon
	# convert [(x1, y1), (x2, y2) ...] into np array for better arithmatic
	rocs: list = [np.array(roc) for roc in rocs]
	rotrocs: list = [{'u': tuple((roc[:,0] + roc[:,1])/2), 'v': tuple((roc[:,1]-roc[:,0])/2)} for roc in rocs]
	stdA: list = list()
	for roc in rotrocs:
		stdA.extend(roc['u'])
	stdA: np.ndarray = np.array(sorted(set(stdA)))
	aprotrocs: np.ndarray = np.zeros((len(rotrocs), len(stdA)))
	for iRoc, roc in enumerate(rotrocs):
		inter = interp1d(roc['u'], roc['v'])
		for iU, u in enumerate(stdA):
			aprotrocs[iRoc][iU]: float = inter(u)
	ymean: np.ndarray = np.zeros((1, len(stdA)))
	for apro in aprotrocs:
		ymean += apro
	ymean /= len(aprotrocs)
	fpout: np.ndarray = stdA - ymean
	tpout: np.ndarray = stdA + ymean
	ret = tpout.tolist(), fpout.tolist()
	return ret[0][0], ret[1][0]

def successMatrix(predicted: list, D0: list, D1: list):
	"""Creates the success matrix for the predicted ordering.
	Checks to make sure it got every entry filled."""
	arr: np.ndarray = np.full((len(D1), len(D0)), -1)
	indecies: dict = dict()
	for val in D0 + D1:
		indecies[val]: int = predicted.index(val)
	for col, x in enumerate(reversed(D0)):
		for row, y in enumerate(reversed(D1)):
			arr[row, col]: bool = indecies[x] < indecies[y]
	if -1 in arr:
		raise EnvironmentError("failed to create success matrix")
	return arr

def runStats(groups: list, params: list, comp) -> list:
	"""Runs stats on the groups provided.
	Params parameter must be: ((d0d1), dist, targetAUC, n, currLayer, len(mergers))"""

	aucs, varOfSM, hanleyMcNeils, estimates = list(), list(), list(), list()
	d0d1, dist, targetAUC, n, *_ = params
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

if __name__ == "__main__":
	from DylSort import mergeSort
	test: int = 9
	if test == 1:
		#print(D0, D1)
		newData, D0, D1 = continuousScale("sampledata.csv")
		print(auc(genROC(newData)))
		arrays: list = [newData[:]]
		for _ in mergeSort(newData):
			arrays.append(newData[:])
		print(arrays)
		graphROCs(arrays)
	elif test == 3:
		predicted: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
		print(aucSM(successMatrix(predicted, [*range(10)], [*range(10,20)])))
	elif test == 4:
		arrays: list = [[0, 1, 4, 2, 5, 3, 6],
			[0, 1, 2, 4, 3, 5, 6],
			[0, 1, 2, 4, 3, 5, 6],
			[0, 1, 2, 3, 4, 5, 6]]
		graphROCs(arrays, D0=[0, 1, 2, 3], D1=[4, 5, 6])
	elif test == 5:
		graphROC([4, 1, 2, 3], [1, 2], [3, 4])
	elif test == 6:
		from DylSort import treeMergeSort
		from DylComp import Comparator
		from DylData import continuousScale
		import matplotlib
		font: dict = {'size' : 10}
		matplotlib.rc('font', **font)
		data, D0, D1 = continuousScale(128, 128)
		comp: Comparator = Comparator(data, rand=True, level=0, seed=15)
		for arr in treeMergeSort(data, comp=comp):
			pass
		D0.sort(key = comp.getLatentScore)
		D1.sort(key = comp.getLatentScore)
		roc: dict = ROC1.rocxy(comp.getLatentScore(D1), comp.getLatentScore(D0))
		graphROCs([arr], True, True, D0, D1)
	elif test == 7:
		roc1: list = [[0, 0], [0, 1], [1, 1]]
		roc3 = roc2 = roc1
		roc4: list = [[0, 0], [0.5, 0], [0.5, 0.5], [1, 1]]
		avgROC: tuple = avROC([roc1, roc2, roc3, roc4])
		fig = plt.figure(figsize=(4,4))
		ax = fig.add_subplot(111)
		ax.plot(*zip(*roc1), 'm', label='chunk1', ls='-')
		ax.plot(*zip(*roc2), 'b', label='chunk2', ls='--')
		ax.plot(*zip(*roc3), 'g', label='chunk3', ls=':')
		ax.plot(*zip(*roc4), 'c', label='chunk4')
		ax.plot(*avgROC, 'orange', label='avg')
		ax.plot((0,1),(0,1),c="r", linestyle="--")
		ax.set_ylim(top=1.1,bottom=-0.1)
		ax.set_xlim(left=-0.1,right=1.1)
		ax.set_xlabel("FPF")
		ax.set_ylabel("TPF")
		ax.legend()
		plt.show()
	elif test == 8:
		roc1: list = [[0,0],[0,0.05],[0,0.1],[0,0.15],[0,0.2],[0,0.25],[0,0.3],[0,0.35],[0,0.4],[0,0.45],[0.1,0.45],[0.1,0.5],[0.1,0.55],[0.1,0.6],[0.1,0.65],[0.2,0.65],[0.3,0.65],[0.3,0.7],[0.4,0.7],[0.5,0.7],[0.5,0.75],[0.5,0.8],[0.5,0.85],[0.5,0.9],[0.5,0.95],[0.5,1],[0.6,1],[0.7,1],[0.8,1],[0.9,1],[1,1]]
		roc2: list = [[0,0],[0,0.1],[0,0.2],[0,0.3],[0,0.4],[0,0.5],[0.06666667,0.5],[0.13333333,0.5],[0.2,0.5],[0.26666667,0.5],[0.26666667,0.6],[0.26666667,0.7],[0.33333333,0.7],[0.4,0.7],[0.4,0.8],[0.4,0.9],[0.4,1],[0.46666667,1],[0.53333333,1],[0.6,1],[0.66666667,1],[0.73333333,1],[0.8,1],[0.86666667,1],[0.93333333,1],[1,1]]
		avgROC: tuple = avROC([roc1, roc2])
		fig = plt.figure(figsize=(4,4))
		ax = fig.add_subplot(111)
		ax.plot(*zip(*roc1), 'm', label='chunk1', ls=':', marker='o')
		ax.plot(*zip(*roc2), 'b', label='chunk2', ls='--', marker='o')
		ax.plot(*avgROC, 'orange', label='avg', marker='o')
		ax.legend()
		plt.show()
	elif test == 9:
		from DylSort import treeMergeSort
		from DylComp import Comparator
		import matplotlib.pyplot as plt
		from time import time
		t1: float = time()
		data, D0, D1 = continuousScale(2048, 2048)
		comp: Comparator = Comparator(data, rand=True)
		comp.genRand(len(D0), len(D1), 7.72, 'exponential')
		fig = plt.figure()
		for level, groups in enumerate(treeMergeSort(data, comp, combGroups=False)):
			rocs: list = list()
			for group in groups:
				roc: list = genROC(group, D0, D1)
				rocs.append(roc)
			avgROC:tuple = avROC(rocs)
			rocs: list = list(zip(*avgROC))
			rocs.reverse()
			mse: float = MSE(7.72, 'exponential', rocs)
			#print(*mse, auc(rocs))
			print(f"{mse[0]:03.3e}, {-auc(rocs):0.3f}, {len(comp)}")
			ax = fig.add_subplot(3, 4, level + 1)
			ax.set_aspect('equal', 'box')
			approx = interp1d(*zip(*rocs), 'linear')
			ax.plot(list(np.arange(0, 1 - 10**-4, 10**-4)), [approx(fp) for fp in np.arange(0, 1 - 10**-4, 10**-4)])
			ax.plot(list(np.arange(0, 1 - 10**-4, 10**-4)), [fp**(1/7.72) for fp in np.arange(0, 1 - 10**-4, 10**-4)])
			ax.set(title=f"{mse[0]:03.6e}:{len(comp)}")
		#plt.subplots_adjust(hspace=0.25)
		plt.show()
		print(time() - t1)
