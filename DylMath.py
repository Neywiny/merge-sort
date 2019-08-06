import ROC1
import numpy as np
import math
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
	font = {'size' : 56}
	#matplotlib.rc('font', **font)
	from matplotlib.collections import PatchCollection
	from matplotlib.patches import Rectangle
except BaseException as e:
	pass
from DylData import *
unbiasedMeanMatrixVar = ROC1.unbiasedMeanMatrixVar
def paramToParams(predicted, D0=None, D1=None):
	if isinstance(predicted[0], (list, tuple)):
		return predicted[0], predicted[1], predicted[2]
	else:
		return predicted, D0, D1
def se(inp: list, n=None) -> float:
	"""se(inp) -> standard error of the input
	inp can be a list or the stdev of the list, in which case
	n needs to be provided"""
	return stdev(inp) / math.sqrt(len(n) if n != None else len(inp))
def pc(arr: list, D0: list, D1: list) -> float:
	# calc % correct
	# add up all the times a number is on the correct side
	# divide by total to get the average
	pc: float = 0.0
	for i,val in enumerate(arr):
		if val in D0:
			if i < len(D0):
				pc += 1
		elif val in D1:
			if i > len(D0):
				pc += 1
	return pc / len(arr)
def auc(results: tuple, D0=None, D1=None) -> float:
	if not isinstance(results[0], (list, tuple)):
		results = genROC(results, D0, D1)
	total: float = 0.0
	for i,(x,y) in enumerate(results[:-1], start=1):
		total += 0.5*(y + results[i][1]) * (x - results[i][0])
	return -total
def hanleyMcNeil(auc, n0, n1):
	# The very good power-law variance estimate from Hanley/McNeil
	auc2=auc*auc
	q1=auc/(2.-auc)
	q2=2.*auc2/(1.+auc)
	return( (auc-auc2+(n1-1.)*(q1-auc2)+(n0-1.)*(q2-auc2))/n0/n1 )
def aucSM(sm) -> float:
	return np.mean(sm)
def calcNLayers(arr) -> int:
	return math.ceil(math.log2(len(arr)))
def fRange(stop, step):
	i = 0
	while i < stop:
		yield i
		i += step
def MSE(sep, dist, ROC, rocEmpiric=None):
	step = 10**-4
	fpf = np.arange(0, 1, step)
	if len(ROC) == 2:
		approx = interp1d(*((ROC['x'], ROC['y']) if isinstance(ROC, dict) else ROC))(fpf)
	else:
		approx = interp1d(*zip(*ROC))(fpf)
	if dist == 'exponential':
		mseTrue = np.mean((approx - (fpf**(1/sep)))**2)
	elif dist == 'normal':
		mseTrue = np.mean((approx - (1-norm.cdf(norm.ppf(1-fpf) - sep)))**2)
	if rocEmpiric != None:
		if len(rocEmpiric) == 2:
			trueApprox = interp1d(rocEmpiric['x'], rocEmpiric['y'])
		else:
			trueApprox = interp1d(*zip(*rocEmpiric))
		mseEmperic = np.mean((approx - (trueApprox(fpf)))**2)
	calcAUC = np.trapz(approx) / (1/step)
	return (mseTrue, calcAUC) if rocEmpiric == None else (mseTrue, mseEmperic, calcAUC)
def genROC(predicted: tuple, D1: tuple=None, D0: tuple=None) -> tuple:
	predicted, D0, D1 = paramToParams(predicted, D0, D1)
	x0 = list()
	x1 = list()
	for i, val in enumerate(predicted):
		if val in D1:
			x1.append(i)
		elif val in D0:
			x0.append(i)
	roc = ROC1.rocxy(x1, x0)
	return list(zip(roc['x'], roc['y']))
def graphROC(predicted: tuple, D0=None, D1=None):
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
def graphROCs(arrays: list, withPatches=False, withLine=True, D0=None, D1=None):
	rows = int(math.ceil(math.sqrt(len(arrays))))
	cols = int(math.ceil(len(arrays) / rows))
	fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, num="plots")
	fig.suptitle("ROC curves")
	if withLine:
		params = [(array, D0, D1) for array in arrays]
		if len(arrays[0]) < 1024:
			results = list(map(genROC, params))
		else:
			with Pool() as p:
				results = list(p.imap(genROC,params))
	if withPatches:
		pbar = tqdm(total=len(arrays)*(len(arrays[0])//2)**2)
	for i,ax in enumerate(axes.flat if (rows * cols > 1) else [axes]):
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
			sm = successMatrix(arrays[i], D0, D1)
			yes = []
			no = []
			yLen = len(D1)
			xLen = len(D0)
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
			ax.set_title(f"Iteration #{i} PC: {int(pc(arrays[i], D0, D1)*100)}% AUC: {area:.5f}")
			ax.set_aspect('equal', 'box')
	if withPatches:
		pbar.close()
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()
def avROC(rocs):
	#hard coded SeSp
	#e = 9*sys.float_info.epsilon
	# convert [(x1, y1), (x2, y2) ...] into np array for better arithmatic
	rocs = [np.array(roc) for roc in rocs]
	rotrocs = [{'u': tuple((roc[:,0] + roc[:,1])/2), 'v': tuple((roc[:,1]-roc[:,0])/2)} for roc in rocs]
	stdA = list()
	for roc in rotrocs:
		stdA.extend(roc['u'])
	stdA = np.array(sorted(set(stdA)))
	aprotrocs = np.zeros((len(rotrocs), len(stdA)))
	for iRoc, roc in enumerate(rotrocs):
		inter = interp1d(roc['u'], roc['v'])
		for iU, u in enumerate(stdA):
			aprotrocs[iRoc][iU] = inter(u)
	ymean = np.zeros((1, len(stdA)))
	for apro in aprotrocs:
		ymean += apro
	ymean /= len(aprotrocs)
	fpout = stdA - ymean
	tpout = stdA + ymean
	ret = tpout.tolist(), fpout.tolist()
	return ret[0][0], ret[1][0]
def successMatrix(predicted: list, D0: list=None, D1: list=None):
	if D0 == None:
		D0 = [i for i in range(len(predicted) // 2)]
	if D1 == None:
		D1 = [i for i in range(len(predicted) // 2, len(predicted))]
	arr = np.full((len(D1), len(D0)), -1)
	indecies = dict()
	for val in D0 + D1:
		indecies[val] = predicted.index(val)
	for col, x in enumerate(reversed(D0)):
		for row, y in enumerate(reversed(D1)):
			arr[row, col] = indecies[x] < indecies[y]
	if -1 in arr:
		raise EnvironmentError("failed to create success matrix")
	return arr
if __name__ == "__main__":
	from DylSort import mergeSort
	test = 9
	if test == 1:
		#print(D0, D1)
		newData, D0, D1 = continuousScale("sampledata.csv")
		print(auc(genROC(newData)))
		arrays = [newData[:]]
		for _ in mergeSort(newData):
			arrays.append(newData[:])
		print(arrays)
		graphROCs(arrays)
	elif test == 2:
		predicted = [0, 1, 5, 2, 3, 6, 4, 7, 8, 9]
		mat = successMatrix(predicted)
		print(mat)
		graphROC(predicted)
	elif test == 3:
		predicted = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
		print(aucSM(successMatrix(predicted, [*range(10)], [*range(10,20)])))
	elif test == 4:
		arrays = [[0, 1, 4, 2, 5, 3, 6],
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
		from ROC1 import rocxy
		import matplotlib
		font = {'size' : 10}
		matplotlib.rc('font', **font)
		data, D0, D1 = continuousScale(128, 128)
		comp = Comparator(data, rand=True, level=0, seed=15)
		for arr in treeMergeSort(data, comp=comp):
			pass
		D0.sort(key = comp.getLatentScore)
		D1.sort(key = comp.getLatentScore)
		roc = rocxy(comp.getLatentScore(D1), comp.getLatentScore(D0))
		graphROCs([arr], True, True, D0, D1)
	elif test == 7:
		roc1 = [[0, 0], [0, 1], [1, 1]]
		roc3 = roc2 = roc1
		roc4 = [[0, 0], [0.5, 0], [0.5, 0.5], [1, 1]]
		avgROC = avROC([roc1, roc2, roc3, roc4])
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
		roc1 = [[0,0],[0,0.05],[0,0.1],[0,0.15],[0,0.2],[0,0.25],[0,0.3],[0,0.35],[0,0.4],[0,0.45],[0.1,0.45],[0.1,0.5],[0.1,0.55],[0.1,0.6],[0.1,0.65],[0.2,0.65],[0.3,0.65],[0.3,0.7],[0.4,0.7],[0.5,0.7],[0.5,0.75],[0.5,0.8],[0.5,0.85],[0.5,0.9],[0.5,0.95],[0.5,1],[0.6,1],[0.7,1],[0.8,1],[0.9,1],[1,1]]
		roc2 = [[0,0],[0,0.1],[0,0.2],[0,0.3],[0,0.4],[0,0.5],[0.06666667,0.5],[0.13333333,0.5],[0.2,0.5],[0.26666667,0.5],[0.26666667,0.6],[0.26666667,0.7],[0.33333333,0.7],[0.4,0.7],[0.4,0.8],[0.4,0.9],[0.4,1],[0.46666667,1],[0.53333333,1],[0.6,1],[0.66666667,1],[0.73333333,1],[0.8,1],[0.86666667,1],[0.93333333,1],[1,1]]
		avgROC = avROC([roc1, roc2])
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
		t1 = time()
		data, D0, D1 = continuousScale(1000, 1000)
		comp = Comparator(data, rand=True)
		comp.genRand(len(D0), len(D1), 7.72, 'exponential')
		fig = plt.figure()
		for level, groups in enumerate(treeMergeSort(data, comp, combGroups=False)):
			rocs = list()
			for group in groups:
				rocs.append(genROC(group, D0, D1))
				rocs = list(zip(*avROC(rocs)))
				rocs.reverse()
			mse = MSE(7.72, 'exponential', rocs)
			#print(*mse, auc(rocs))
			print(f"{mse[0]:03.3e}, {-auc(rocs):0.3f}, {len(comp)}")
			ax = fig.add_subplot(3, 4, level + 1)
			approx = interp1d(*zip(*rocs), 'linear')
			ax.plot(list(fRange(1 - 10**-4, 10**-4)), [approx(fp) for fp in fRange(1 - 10**-4, 10**-4)])
			ax.plot(list(fRange(1 - 10**-4, 10**-4)), [fp**(1/7.72) for fp in fRange(1 - 10**-4, 10**-4)])
			ax.set(title=f"{mse[0]:03.6e}:{len(comp)}")
		#plt.subplots_adjust(hspace=0.25)
		plt.show()
		print(time() - t1)
