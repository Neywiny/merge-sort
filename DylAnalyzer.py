#!/usr/bin/python3.6
import pickle
import json
from typing import Dict, List
import ROC1
import tqdm
import sys
import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from multiprocessing import Pool
from DylComp import Comparator
from DylData import continuousScale
from DylSort import treeMergeSort
from DylMath import genX0X1, MSE, auc

def analyzeMergeSims(fileName: str, length: int, layers: int, justOne: bool=False, bar: bool=False) -> tuple:
	"""Analyze a merge sort results file.

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
	avgEstimates: np.ndarray[float] = np.array([np.zeros((i)) for i in range(layers)])
	avgMinSeps: np.ndarray[float] = np.ones((layers, length))
	varEstimates: np.ndarray[float] = np.zeros((layers, 0))
	aucs: np.ndarray = np.zeros((layers, 0))
	iters: int = 0
	fileLength: int = os.stat(fileName).st_size
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
				varEstimates[iLevel][iters - 1] = varEstimate
				aucs[iLevel][iters - 1] = auc

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
				pBar.desc = f"{iters}/{iterEstimate}, {reshapeCount}, {sys.getsizeof(unpickler)}"
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
	scores: Dict[int] = dict()
	with open(fileName) as f:
		posDir, negDir = f.readline().strip().split()
		for line in f:
			line: list = line.rstrip().split()
			times.append(float(line[-1]))
			score: int = int(line[1])
			# DylScale may do more ratings than are needed
			# this ensures (if the user wants) that they aren't included
			if not names or line[0] in names:
				scores[line[0]] = score

	for name in sorted(scores.keys()):
		if negDir in name:
			x0.append(scores[name])
		elif posDir in name:
			x1.append(scores[name])
	x1, x0 = np.array(x1), np.transpose(x0)
	return times, x0, x1, scores

def analyzeAFCStudies(log: str, results: str, n0: int, n1: int) -> tuple:
	"""Extracts the times out of the log file generated from DylAFC.

	extracts the x0 and x1 vectors and the ranks from the results file from DylComp"""
	times = list()
	with open(log) as f:
		for line in f:
			line: list = line.strip().split()
			times.append(float(line[-1]))

	data, D0, D1 = continuousScale(n0, n1)
	comp = Comparator(data, rand=True)
	# this redoes the study with the decisions of the reader
	comp.learn(results)
	for arr in treeMergeSort(data[:], comp):
		pass
	indeciesAFC: list = [arr.index(i) for i in range(n0 + n1)]
	x0, x1 = genX0X1(arr, D1, D0)
	x0: np.ndarray = np.array([indeciesAFC[i] for i in range(n0)])
	x1: np.ndarray = np.array([indeciesAFC[i] for i in range(n0, n0 + n1)])
	return times, x0, x1, indeciesAFC

def analyzeReaderStudies(resultsFile, directory, n0):
	roc8s = list()
	roc4s = list()
	rocScales = list()
	AUCss = list()
	VARss = list()
	PCss = list()
	with open(resultsFile) as f:
		results = json.load(f)
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
	return AUCss, VARss, PCss, readers, rocScales, roc8s, roc4s

def bootstrapTau(arr: list):
	"""Function for permuting the columns of the array with replacement."""
	ranks: np.ndarray = arr[:,np.random.randint(len(arr[0]), size=len(arr[0]))]
	return stats.kendalltau(ranks[0], ranks[1])[0]

def permutation(arr: list, D0: list, D1: list, n0: int, n1: int):
	"""Permutation test for MSEs."""
	indecies: np.ndarray = np.random.randint(2, size=len(arr[0]))
	scales: np.ndarray = arr[indecies, range(len(arr[0]))]
	afcs: np.ndarray = arr[1 - indecies, range(len(arr[0]))]
	x0 = [scales[i] + 1 for i in range(n0)]
	x1 = [scales[i] + 1 for i in range(n0, n0 + n1)]
	scaleROC: dict = ROC1.rocxy(x1, x0)
	x0 = [afcs[i] + 1 for i in range(n0)]
	x1 = [afcs[i] + 1 for i in range(n0, n0 + n1)]
	afcROC: dict = ROC1.rocxy(x1, x0)
	return MSE(None, None, scaleROC, afcROC)[1]


if __name__ == "__main__":
	# if the first argument is a 1, analyze simulation
	# if the first argument is a 2, analyze study
	if len(sys.argv) > 1:
		if sys.argv[1] == '2' and len(sys.argv) >= 4:
			test: int = 2
		elif sys.argv[1] == '1' and len(sys.argv) == 5:
			test: int = 1
		elif sys.argv[1] == '3' and len(sys.argv) == 3:
			test: int = 3
		else:
			test: int = -1
	else:
		test: int = -1
	if test == 1:
		# Shows the 5 plot dashboard for studies
		length: int = int(sys.argv[3])
		layers: int = int(sys.argv[4])
		varEstimate, avgAUC, avgMSETrues, avgMSEEmpiric, avgComps, avgHanleyMNeil, avgEstimates, avgMinSeps, varAUCnp, stdVarEstimate, avgPC, iters = analyzeMergeSims(sys.argv[2], length, layers, bar=True)
		labels: list = [f'{np.median(list(filter(lambda x: x != 0, avgMinSeps[0]))):3.02f}']
		for val in np.median(avgMinSeps, axis=0)[1:]:
			labels.append(f'{val:3.02f}')
		tickLabels = ['0'] + [str(int(n)) for n in avgComps]
		xVals: list = avgComps

		fig = plt.figure()
		ax1 = fig.add_subplot(2, 3, 1)
		ax1.errorbar(xVals, avgAUC, yerr=np.sqrt(varEstimate), capsize=5, c='r', lw=1, elinewidth=2, label="$\pm\sqrt{var_{estimate}}$")
		#ax1.set_ylim(top=0.96)
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
		info: List[float] = [-1 for i in range(layers - 1)]
		for layer in range(layers - 1):
			try:
				info[layer] = ((1/varEstimate[layer + 1]) - (1/varEstimate[layer]))/(avgComps[layer + 1] - avgComps[layer])
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

		n0: int = -1
		n1: int = -1

		with open(sys.argv[2]) as f:
			results: dict = json.load(f)
		with open(sys.argv[3]) as f:
			# files are separated with... the file separator character. amazing
			names: list = f.read().rstrip('\x1C').split('\x1C')
		if min((len(files) for files in results.values())) == 4:
			fig, (scatterAxes, timeAxes, tauAxes) = plt.subplots(ncols=len(results), nrows=3)
		else:
			fig, (timeAxes) = plt.subplots(ncols=len(results), nrows=1)
			scatterAxes = None; tauAxes = None # appease the linter
		fontSize: int = 16
		plt.rcParams["font.size"] = fontSize
		line = "reader\t(scaleTimes)\tstd(scaleTimes)\tmean(mergeTimes)\tstd(mergeTimes)\ttau\tstd(taus)"
		print(line)
		print('-' * int(len(line) * 1.2))
		for i, (reader, files) in enumerate(results.items()):
			if n0 == -1: # run n0/n1 detection
				with open(files[2]) as f:
					header = True
					for line in f:
						if header:
							header = False
						elif n0 == -1:
							# the first comparison is between image 0 and image [n0]
							n0 = int(line.split(',')[1])
						else:
							#the last comparison is between image [n0-1] and image [n1]
							# so keep feeding lines until we get it
							if len(line.split(',')) == 3:
								lastLine = line
							else:
								n1 = int(lastLine.split(',')[1]) - n0 + 1
								break

			afcTime, afcX0, afcX1, afcRanks = analyzeAFCStudies(files[0], files[2], n0, n1)
			mergeTimes: list = list(filter(lambda x: x < 5, afcTime))
			xmax = max(mergeTimes)

			timeAxes[i].set_ylim(bottom=0, top=100)
			timeAxes[i].set_xlim(left=0, right=xmax)
			timeAxes[i].set_ylabel("Percentage", fontsize=fontSize)
			timeAxes[i].set_xlabel("Time", fontsize=fontSize)
			timeAxes[i].set_title("Times")

			if len(files) == 4:
				scaleTimes, x0, x1, scoresScale = analyzeScaleStudy(files[3], names=names)
				scaleSMData: np.ndarray = ROC1.successmatrix(x1, x0)
				scaleTimes: list = list(filter(lambda x: x < 10, scaleTimes))
				xmax: float = np.append(scaleTimes, mergeTimes).max()
				kernal = stats.gaussian_kde(scaleTimes)
				xVals: np.ndarray = np.linspace(0, xmax, 1000)
				timeAxes[i].fill_between(xVals, kernal(xVals) * 100, label="scale", alpha=0.5)
				kernal = stats.gaussian_kde(mergeTimes)
				xVals: np.ndarray = np.linspace(0, xmax, 1000)
				timeAxes[i].fill_between(xVals, kernal(xVals) * 100, label="merge", alpha=0.5)
				timeAxes[i].set_xlim(left=0, right=xmax)
				timeAxes[i].legend()
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
				for x in range(n0, n0 + n1):
					#only apply the label if it's the last marker, because I'm not doing just one scatter plot
					scatterAxes[i].scatter(ranks[0][x], ranks[1][x], c="g", marker="o", linestyle='None', s=4, label='+' if x == n0 + n1 - 1 else '')
				for x in range(n0):
					scatterAxes[i].scatter(ranks[0][x], ranks[1][x], c="b", marker="^", linestyle='None', s=4, label='-' if x == n0 - 1 else '')
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
				#tighten up legend. No need for so much white space
				scatterAxes[i].legend(loc='lower right', numpoints=1, handletextpad=0.1, borderaxespad=0.05, labelspacing=0.1)

				with Pool(initializer=np.random.seed) as p:
						taus = p.map(bootstrapTau, (ranks for _ in range(1_000)))
						mses = p.starmap(permutation, ((ranks, list(range(n0)), list(range(n0, n1)), n0, n1) for _ in range(1_000)))

				xmax: float = max(taus)
				kernal = stats.gaussian_kde(taus)
				xVals: np.ndarray = np.linspace(0, xmax, 1000)
				tauAxes[i].set_title("Kendall's Tau")
				tauAxes[i].fill_between(xVals, kernal(xVals))
				tauAxes[i].set_ylim(bottom=0)
				tauAxes[i].set_xlabel("Tau", fontsize=fontSize)
				tauAxes[i].set_ylabel("Percentage", fontsize=fontSize)

				p = np.mean([mse < permutted for permutted in mses])

				dSM = afcSM - scaleSM
				varDSM: float = ROC1.unbiasedMeanMatrixVar(dSM)
				dAUC: float = np.mean(afcSM) - np.mean(scaleSM)
				stdDSM: float = np.sqrt(varDSM)
				z: float = np.abs(dAUC / stdDSM) # how many standard deviations away it is
				wald: float = 2 * (1 - stats.norm.cdf(z))
			else:
				timeAxes[i].set_title(reader)
				scaleTimes: list = [0]
				taus: list = [0]
				tau: int = 0
				timeAxes[i].legend()
				kernal = stats.gaussian_kde(mergeTimes)
				xVals: np.ndarray = np.linspace(0, xmax, 1000)
				timeAxes[i].fill_between(xVals, kernal(xVals) * 100, label="merge", alpha=0.5)

			print(f"{reader} {np.mean(scaleTimes):0.3f}\t\t{np.std(scaleTimes):0.3f}\t\t{np.mean(mergeTimes):0.3f}\t\t\t{np.std(mergeTimes):0.3f}\t\t{tau:0.3f}\t{np.std(taus):0.3f}")
		fig.set_size_inches(24, 16)
		fig.tight_layout()
		if len(sys.argv) == 5:
			plt.savefig(sys.argv[4], bbox_inches = 'tight', pad_inches = 0)
		else:
			plt.show()
	elif test == 3:
		with open(sys.argv[2]) as f:
			results: dict = json.load(f)
		statistics = list()
		readers = list()
		for i, (reader, files) in enumerate(results.items()):
			statistics.append(list())
			readers.append(reader)
			with open(files[2]) as f:
				for line in f:
					line = line.replace('[', '').replace(']', '').rstrip()
					split = line.split(',')
					if len(split) > 3:
						statistics[-1].append(split[:2]) # only AUC and variance
		print('reader\tlayer\tauc\tvariance')
		for (reader, statistic) in zip(readers, statistics):
			for layer, (_auc, var) in enumerate(statistic):
				print(reader, layer, _auc, var, sep='\t')
	else:
		print("Usage:")
		print(f"{__file__} 1 [simulation results file]")
		print(f"{__file__} 2 [json results file for reader studies] [names.txt filename] [optional output directory]")
		print(f"{__file__} 3 [json results file for reader studies]")