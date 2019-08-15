import pickle
import json
import ROC1
import tqdm
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from os import stat
from sys import getsizeof, argv
from scipy import stats
from multiprocessing import Pool
from DylComp import Comparator
from DylData import continuousScale
from DylSort import treeMergeSort
from DylMath import genX0X1, MSE, auc

def analyzeMergeSims(fileName: str, length: int, layers: int, justOne: bool=False, bar: bool=False) -> tuple:
	""" analyze a merge sort results file.
	If justOne is True, only does the first simulation
	If bar is True, shows a tqdm progress bar"""

	# Game plan:
	# create vectors for each statistic for each layer
	# add each stat, then divide by num of iters to get avg
	# also record each AUC and var estimate for variance
	avgAUC: np.ndarray = np.zeros((layers,))
	avgComps: np.ndarray = np.zeros((layers,))
	avgHanleyMNeil: np.ndarray = np.zeros((layers,))
	avgMSETrues: np.ndarray = np.zeros((layers,))
	avgMSEEmpiric: np.ndarray = np.zeros((layers,))
	avgPC: np.ndarray = np.zeros((layers,))
	avgEstimates: np.ndarray = np.array([np.zeros((i)) for i in range(layers)])
	avgMinSeps: np.ndarray = np.ones((layers, length))
	varEstimates: np.ndarray = np.zeros((layers, 0))
	aucs: np.ndarray = np.zeros((layers, 0))
	iters: int = 0
	fileLength: int = stat(fileName).st_size
	old: int = 0
	with open(fileName, "rb") as f, tqdm.tqdm(total=fileLength, unit="B", unit_scale=True, disable=not bar) as pBar:
		# each simulation is one pickle, so need to depickle one at a time
		unpickler = pickle.Unpickler(f)
		reshapeCount: int = 0 # just for keeping track
		# while more stuff to get
		while f.tell() < fileLength:
			iteration: list = unpickler.load()
			iters += 1
			# extrapolate how many iterations are in the file
			iterEstimate: int = int(fileLength /(f.tell() / iters))
			# if there is not space for another iteration:
			if iters > len(aucs[0]):
				reshapeCount += 1 # for keeping track
				# for aucs and varEstimates, create a new array then copy the old data

				# do this for an array of the predicted iters avoid repeatedly calling the function
				# use max() to make sure there's at least one extra space
				new: np.ndarray = np.zeros((layers, max(iterEstimate, iters + 1)))
				new[:aucs.shape[0], :aucs.shape[1]] = aucs
				aucs = new
				new: np.ndarray = np.zeros((layers, max(iterEstimate, iters + 1)))
				new[:varEstimates.shape[0], :varEstimates.shape[1]] = varEstimates
				varEstimates = new
				del new # we don't need it anymore
			for iLevel, (auc, varEstimate, hanleyMcNeil, estimates, mseTrue, mseEmpiric, compLen, minSeps, pc) in enumerate(iteration):
				# store results
				varEstimates[iLevel][iters - 1]: float = varEstimate
				aucs[iLevel][iters - 1]: float = auc

				# add to running total
				avgHanleyMNeil[iLevel] += hanleyMcNeil
				avgMSEEmpiric[iLevel] += mseEmpiric
				avgMSETrues[iLevel] += mseTrue
				avgMinSeps[iLevel] += minSeps
				avgComps[iLevel] += compLen
				avgAUC[iLevel] += auc
				avgPC[iLevel] += pc
				avgEstimates[layers - iLevel - 1] += estimates

				# update how many bytes were read
				pBar.update(f.tell() - old)
				pBar.desc: str = f"{iters}/{iterEstimate}, {reshapeCount}, {getsizeof(unpickler)}"
				old: int = f.tell()

				if justOne:
					break
	# get rid of the first layer because it's not needed and transpose to be the same shape as the others
	avgMinSeps = avgMinSeps.transpose()[:,1:,]

	# iterEstimate can overshoot, so trim them back
	aucs = aucs[:,:iters]
	varEstimates = varEstimates[:,:iters]

	# need to transpose because numpy is weird
	avgAUC = (avgAUC / iters).transpose()

	# divide vectors by iters to get average
	# // avgComps because can't have fraction of a comparison
	avgComps = avgComps // iters
	avgHanleyMNeil /= iters
	avgMSEEmpiric /= iters
	avgEstimates /= iters
	avgMSETrues /= iters
	avgMinSeps /= iters
	avgPC   /= iters

	# axis=1 is across the simulations
	varEstimate: float = np.mean(varEstimates, axis=1)
	varAUCnp: float = np.var(aucs, ddof=1, axis=1)
	stdVarEstimate: float = np.sqrt(np.var(varEstimates, axis=1, ddof=1))

	return varEstimate, avgAUC, avgMSETrues, avgMSEEmpiric, avgComps, avgHanleyMNeil, avgEstimates, avgMinSeps, varAUCnp, stdVarEstimate, avgPC, iters

def analyzeEloSims(filename: str, passes) -> list:
	"""Analyze an ELO simulation.
	Returns AUC, true Var, bad Var, the lsit of bad vars, mse True, mse Emperic and PC"""
	aucs = [list() for _ in range(passes)]
	masVars = [list() for _ in range(passes)]
	mseTruths = np.zeros((passes,), dtype=float)
	MSEEmpirics = np.zeros((passes,), dtype=float)
	pcs = np.zeros((passes,), dtype=float)
	iters = -1
	with open(filename, "rb") as f:
		unpickler = pickle.Unpickler(f)
		while True:
			try: # because Unpickler can't just raise a StopIterationError...
				iteration = unpickler.load()
				iters += 1
			except EOFError:
				break
			for N, _, ncmp, var, auc, mseTruth, mseEmpiric, pc in iteration:
				idx = ncmp // N - 1
				aucs[idx].append(auc)
				masVars[idx].append(var)
				pcs[idx] += pc
				mseTruths[idx] += mseTruth
				MSEEmpirics[idx] += mseEmpiric
	masAvgAUC = np.mean(aucs, axis=1)
	masVarAUC = np.var(aucs, ddof=1, axis=1)
	masAvgVAR = np.mean(masVars, axis=1)
	masAvgMSETruth = mseTruths / iters
	masAvgMSEEmpiric = MSEEmpirics / iters
	avgPC = pcs / iters
	return masAvgAUC, masVarAUC, masAvgVAR, masVars, masAvgMSETruth, masAvgMSEEmpiric, avgPC

def analyzeScaleStudy(fileName:str, names:list=None) -> tuple:
	"""Analyzes a scale study.
	If names parameter given, filters for only those names.
	Names can be any type of iterable."""
	times = list()
	x0 = list()
	x1 = list()
	scores = dict()
	with open(fileName) as f:
		posDir, negDir = f.readline().strip().split()
		for line in f:
			line: list = line.rstrip().split()
			times.append(float(line[-1]))
			score: int = int(line[1])
			# DylScale may do more ratings than are needed
			# this ensures (if the user wants) that they aren't included
			if not names or line[0] in names:
				scores[line[0]]: int = score

	for name in sorted(scores.keys()):
		if negDir in name:
			x0.append(scores[name])
		elif posDir in name:
			x1.append(scores[name])
	x1, x0 = np.array(x1), np.transpose(x0)
	return times, x0, x1, scores

def analyzeAFCStudies(log: str, results: str, n0: int, n1: int) -> tuple:
	"""extracts the times out of the log file generated from DylAFC
	extracts the x0 and x1 vectors and the ranks from the results file from DylComp"""
	times = list()
	with open(log) as f:
		for line in f:
			line: list = line.strip().split()
			times.append(float(line[-1]))

	data, D0, D1 = continuousScale(n0, n1)
	comp = Comparator(data, rand=True)
	comp.learn(results)
	for arr in treeMergeSort(data[:], comp):
		pass
	indeciesAFC: list = [arr.index(i) for i in range(256)]
	x0, x1 = genX0X1(arr, D1, D0)
	x0: np.ndarray = np.array([indeciesAFC[i] for i in range(128)])
	x1: np.ndarray = np.array([indeciesAFC[i] for i in range(128, 256)])
	return times, x0, x1, indeciesAFC

def analyzeReaderStudies(resultsFile, directory, n0):
	roc8s = list()
	roc4s = list()
	rocScales = list()
	AUCss = list()
	VARss = list()
	PCss = list()
	with open("results.json") as f:
		results = json.load(f)
	results = {"Reader A":("resGabi/scaleGabi.csv1565102893.2022426", "resGabi/log2.csv", "resGabi/rocs", "resGabi/compGabi.csv"), "Reader B":("resFrank/scaleFrank.csv1565098562.1623092", "resFrank/log2.csv", "resFrank/rocs", "resFrank/results.csv"), "Reader C":("resDylan/scaleDylan2.csv", "resDylan/log2.csv", "resDylan/rocs", "resDylan/compDylan.csv")}
	readers = results.keys()
	for reader, val in results.items():
		AUCs = list()
		VARs = list()
		PCs = list()
		c = 0
		possibleC = 0
		with open(val[3]) as f:
			for comps, line in enumerate(f, start=-1):
				line = line.split(',')
				comps = comps - len(AUCs)
				if len(line) > 5: # stats line
					AUCs.append((comps, float(line[0])))
					VARs.append((comps, float(line[1])))
					PCs.append((comps, c / (possibleC)))
					possibleC = 0
					c = 0
				elif comps > -1:
					id0 = int(line[0])
					id1 = int(line[1])
					if (id0 < n0) ^ (id1 < n0):
						possibleC += 1
						if int(line[2]) == max(id0, id1):
							c += 1

		AUCss.append(AUCs)
		VARss.append(VARs)
		PCss.append(PCs)
		scaleTimes, x0, x1, _ = analyzeScaleStudy(val[0])
		scaleROC = ROC1.rocxy(x1, x0)
		scaleVAR = ROC1.unbiasedAUCvar(x1, x0)
		scaleAUC = ROC1.auc(x1, x0)
		scaleTimes = list(filter(lambda x: x < 10, scaleTimes))
		mergeTimes, *_ = analyzeAFCStudies(val[1], val[3], 128, 128)
		mergeTimes = list(filter(lambda x: x < 5, mergeTimes))
		with open(val[2], "rb") as f:
			roc8, roc4 = pickle.load(f)
		roc8s.append((roc8, reader, auc(list(zip(*roc8)))))
		roc4s.append((roc4, reader, auc(list(zip(*roc4)))))
		rocScales.append((scaleROC, reader, scaleAUC, scaleVAR))
	#plt.axis('equal')
	return AUCss, VARss, PCss, readers, rocScales, roc8s, roc4s


if __name__ == "__main__":
	if len(argv) > 1:
		if argv[1] == '2':
			test: int = 2
		elif argv[1] == '1':
			test: int = 1
		else:
			test: int = -1
	else:
		arv.append("1")
		argv.append("resultsMerge85") # default value
		test: int = 1
	if test == 1:
		# Shows the 5 plot dashboard for studies
		length: int = 256
		layers: int = 8
		varEstimate, avgAUC, avgMSETrues, avgMSEEmpiric, avgComps, avgHanleyMNeil, avgEstimates, avgMinSeps, varAUCnp, stdVarEstimate, avgPC, iters = analyzeMergeSims(arv[2], length, layers, bar=True)
		labels: list = [f'{np.median(list(filter(lambda x: x != 0, avgMinSeps[0]))):3.02f}']
		for val in np.median(avgMinSeps, axis=0)[1:]:
			labels.append(f'{val:3.02f}')
		tickLabels = ['0'] + [str(int(n)) for n in avgComps]
		xVals: list = avgComps

		fig = plt.figure()
		ax1 = fig.add_subplot(2, 3, 1)
		ax1.errorbar(xVals, avgAUC, yerr=np.sqrt(varEstimate), capsize=5, c='r', lw=1, elinewidth=2, label="$\pm\sqrt{var_{estimate}}$")
		ax1.set_ylim(top=0.96)
		ax1.legend()
		ax1.set_ylabel('AUC', color='b')
		ax1.set_title("Average AUC per Layer")

		ax2 = fig.add_subplot(2, 3, 2)
		ax2.plot(xVals, varAUCnp, 'g.', ls='--', lw=5, label='$var_{real}$')
		ax2.errorbar(xVals, varEstimate, yerr=stdVarEstimate, c='r', marker='.', ls=':', lw=2, label='$var_{estimate}$')
		ax2.plot(xVals, avgHanleyMNeil, 'c.', ls='-', lw=2, label='HmN Variance')
		for layer in range(1, layers):
			# estimate is a list of where that layer estimated the HmN variances would be
			estimate: list = avgEstimates[layer - 1]
			for i, point in enumerate(estimate):
				pass #uncomment next line and comment this one to draw text where the estimates were
				#ax2.text(layer + 1, point, str(i), fontsize=12, horizontalalignment='center', verticalalignment='center')
		ax2.legend()
		ax2.set_title("Variance Estimate per Layer")

		ax3 = fig.add_subplot(2, 3, 3)
		info: list = [-1 for i in range(layers - 1)]
		for layer in range(layers - 1):
			try:
				info[layer]: float = ((1/varEstimate[layer + 1]) - (1/varEstimate[layer]))/(avgComps[layer + 1] - avgComps[layer])
			except ZeroDivisionError:
				print(varEstimate, avgComps)
		ax3.plot(xVals[1:], info, marker='.')
		ax3.set_title("Information Gained per Comparison per Layer")

		ax4 = fig.add_subplot(2, 2, 3)
		ax4.plot([0, len(avgComps)], [0, max(avgComps)], 'b:')
		ax4.plot(list(range(9)), [0, *avgComps], 'r.-', label='comparisons')
		ax4.set_ylabel('Comparisons', color='r')
		ax4.set_yticks([0] + [int(avgComp) for avgComp in avgComps])
		ax4.set_title("Average Comparisons per Layer")

		ax5 = fig.add_subplot(2, 2, 4)
		plot = ax5.imshow(avgMinSeps,norm=LogNorm(), extent=[0, length, 0, length], aspect=0.5)
		ax5.set_xticks([*range(length//(2*layers), length, int((layers / (layers - 1))*(length//layers)))])
		start, end = ax5.get_xlim()
		step: float = length / (layers - 1)
		ax5.set_xticks(np.arange(start + (step / 2), end + (step / 2), step))
		ax5.set_xticklabels(tickLabels[1:])
		cbaxes = fig.add_axes([0.91, 0.13, 0.01, 0.31])
		cbar = fig.colorbar(plot, cax=cbaxes)
		ax5.set_title("Average Distance Between Compairisons per ID per Layer")

		plt.subplots_adjust(wspace=0.45)
		plt.show()
	elif test == 2:

		def bootstrapTau(arr: list):
			"""function for permuting the columns of the array with replacement"""
			ranks: np.ndarray = arr[:,np.random.randint(len(arr[0]), size=len(arr[0]))]
			return stats.kendalltau(ranks[0], ranks[1])[0]

		def permutation(arr: list, D0: list, D1: list):
			"""permutation test for MSEs"""
			indecies: np.ndarray = np.random.randint(2, size=len(arr[0]))
			scales: np.ndarray = arr[indecies, range(len(arr[0]))]
			afcs: np.ndarray = arr[1 - indecies, range(len(arr[0]))]
			x0 = [scales[i] + 1 for i in range(128)]
			x1 = [scales[i] + 1 for i in range(128, 256)]
			scaleROC: dict = ROC1.rocxy(x1, x0)
			x0 = [afcs[i] + 1 for i in range(128)]
			x1 = [afcs[i] + 1 for i in range(128, 256)]
			afcROC: dict = ROC1.rocxy(x1, x0)
			return MSE(None, None, scaleROC, afcROC)[1]

		n0: int = 128
		n1: int = 128

		with open(argv[2]) as f:
			results: dict = json.load(f)
		with open(argv[3]) as f:
			names: list = f.read().split()
		if max((len(files) for files in results.values())) == 4:
			fig, (scatterAxes, timeAxes, tauAxes) = plt.subplots(ncols=3, nrows=3)
		else:
			fig, (timeAxes) = plt.subplots(ncols=3, nrows=1)
		fontSize: int = 8
		plt.rcParams["font.size"] = fontSize
		line = "reader\t(scaleTimes)\tstd(scaleTimes)\tmean(mergeTimes)\tstd(mergeTimes)\ttau\tstd(taus)"
		print(line)
		print('-' * int(len(line) * 1.2))
		for i, (reader, files) in enumerate(results.items()):

			afcTime, afcX0, afcX1, afcRanks = analyzeAFCStudies(files[0], files[2], n0, n1)
			mergeTimes: list = list(filter(lambda x: x < 5, afcTime))

			if len(files) == 4:
				scaleTimes, x0, x1, scoresScale = analyzeScaleStudy(files[3], names=names)
				scaleSMData: np.ndarray = ROC1.successmatrix(x1, x0)
				scaleTimes: list = list(filter(lambda x: x < 10, scaleTimes))

			if len(files) == 4:
				xmax: float = np.append(scaleTimes, mergeTimes).max()
				kernal = stats.gaussian_kde(scaleTimes)
				xVals: np.ndarray = np.linspace(0, xmax, 1000)
				timeAxes[i].fill_between(xVals, kernal(xVals), label="scale", alpha=0.5)
			else:
				xmax = max(mergeTimes)

			kernal = stats.gaussian_kde(mergeTimes)
			xVals: np.ndarray = np.linspace(0, xmax, 1000)
			timeAxes[i].fill_between(xVals, kernal(xVals), label="merge", alpha=0.5)

			timeAxes[i].legend()
			timeAxes[i].set_ylim(bottom=0)
			timeAxes[i].set_xlim(left=0, right=xmax)
			timeAxes[i].set_ylabel("Percentage")
			timeAxes[i].set_xlabel("Time")
			timeAxes[i].set_title("Times")
			if len(files) == 4:
				ranks: np.ndarray = np.zeros((2, n0 + n1))
				for x, name in enumerate(sorted(names)):
					ranks[0, x] = scoresScale[name]
					ranks[1, x] = afcRanks[x]
				ranks[0] = stats.rankdata(ranks[0])
				ranks[1] = stats.rankdata(ranks[1])
				scaleROC: dict = ROC1.rocxy(x1, x0)

				x0 = ranks[0][0:n0]
				x1 = ranks[0][n0:n0 + n1]
				scaleSM = ROC1.successmatrix(x1, x0)
				scaleAUC = ROC1.auc(x1, x0)

				afcSM = ROC1.successmatrix(afcX1, afcX0)
				afcAUC = ROC1.auc(afcX1, afcX0)
				afcROC = ROC1.rocxy(x1, x0)

				mse: float = MSE(None, None, scaleROC, afcROC)[1]

				scatterAxes[i].plot([0, n0 + n1], [0, n0 + n1], 'r:')
				for x in range(n0):
					scatterAxes[i].scatter(ranks[0][x], ranks[1][x], c="b", marker="^", s=2)
				for x in range(n0, n0 + n1):
					scatterAxes[i].scatter(ranks[0][x], ranks[1][x], c="g", marker="o", s=2)
				#scatterAxes[i].text(20, (n0 + n1)*0.9, reader[-1])
				scatterAxes[i].set_aspect('equal', 'box')
				tau: float = stats.kendalltau(ranks[0], ranks[1])[0]
				scatterAxes[i].set_xticks([1, (n0 + n1) // 2, n0 + n1])
				scatterAxes[i].set_yticks([1, (n0 + n1) // 2, n0 + n1])
				scatterAxes[i].set_xticklabels([str(x) for x in [1, (n0 + n1) // 2, n0 + n1]], fontsize=fontSize)
				scatterAxes[i].set_yticklabels([str(x) for x in [1, (n0 + n1) // 2, n0 + n1]], fontsize=fontSize)
				scatterAxes[i].set_title(reader)
				scatterAxes[i].set_xlabel("Image Ranks from Rating Data", fontsize=fontSize)
				scatterAxes[i].set_ylabel("Image Ranks from 2AFC Merge", fontsize=fontSize)

				with Pool(8, initializer=np.random.seed) as p:
						taus = p.map(bootstrapTau, (ranks for _ in range(1_000)))
						mses = p.starmap(permutation, ((ranks, list(range(n0)), list(range(n0, n1))) for _ in range(1_000)))

				xmax: float = max(taus)
				kernal = stats.gaussian_kde(taus)
				xVals: np.ndarray = np.linspace(0, xmax, 1000)
				tauAxes[i].set_title("Kendall's Tau")
				tauAxes[i].fill_between(xVals, kernal(xVals))
				tauAxes[i].set_ylim(bottom=0)

				p = np.mean([mse < permutted for permutted in mses])

				dSM = afcSM - scaleSM
				varDSM: float = ROC1.unbiasedMeanMatrixVar(dSM)
				dAUC: float = np.mean(afcSM) - np.mean(scaleSM)
				stdDSM: float = np.sqrt(varDSM)
				z: float = np.abs(dAUC / stdDSM) # how many standard deviations away it is
				wald: float = 2 * (1 - stats.norm.cdf(z))
			else:
				scaleTimes: list = [0]
				taus: list = [0]
				tau: int = 0

			print(f"{reader} {np.mean(scaleTimes):0.3f}\t\t{np.std(scaleTimes):0.3f}\t\t{np.mean(mergeTimes):0.3f}\t\t\t{np.std(mergeTimes):0.3f}\t\t{tau:0.3f}\t{np.std(taus):0.3f}")
		fig.set_size_inches(12, 8)
		if len(argv) == 4:
			plt.savefig(argv[3], bbox_inches = 'tight', pad_inches = 0)
		else:
			plt.show()
	else:
		print("Usage:")
		print(f"{__file__} [json results file] [optional output directory]")
		print(f"{__file__} [simulation results file]")