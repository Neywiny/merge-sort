import pickle
import ROC1
import tqdm
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from os import stat
from sys import getsizeof
from scipy import stats

def analyze(fileName, length, layers, justOne=False, bar=False):
	""" analyze a merge sort results file.
	If justOne is True, only does the first simulation
	If bar is True, shows a tqdm progress bar"""

	# Game plan:
	# create vectors for each statistic for each layer
	# add each stat, then divide by num of iters to get avg
	# also record each AUC and var estimate for variance
	avgAUC = np.zeros((layers,))
	avgSMVAR = np.zeros((layers,))
	avgnpVARs = np.zeros((layers,))
	avgComps = np.zeros((layers,))
	avgHanleyMNeil = np.zeros((layers,))
	avgMSETrues = np.zeros((layers,))
	avgMSEEmperic = np.zeros((layers,))
	avgPC = np.zeros((layers,))
	avgErrorBars = np.zeros((layers, 6))
	avgEstimates = np.array([np.zeros((i)) for i in range(layers)])
	avgMinSeps = np.ones((layers, length))
	varEstimates = np.zeros((layers, 0))
	aucs = np.zeros((layers, 0))
	iters = 0
	fileLength = stat(fileName).st_size
	old = 0
	with open(fileName, "rb") as f, tqdm.tqdm(total=fileLength, unit="B", unit_scale=True, disable=not bar) as pBar:
		# each simulation is one pickle, so need to depickle one at a time
		unpickler = pickle.Unpickler(f)
		reshapeCount = 0 # just for keeping track
		# while more stuff to get
		while f.tell() < fileLength:
			iteration = unpickler.load()
			iters += 1
			# extrapolate how many iterations are in the file
			iterEstimate = int(fileLength /(f.tell() / iters))
			# if there is not space for another iteration:
			if iters > len(aucs[0]):
				reshapeCount += 1 # for keeping track
				# for aucs and varEstimates, create a new array then copy the old data

				# do this for an array of the predicted iters avoid repeatedly calling the function
				# use max() to make sure there's at least one extra space
				new = np.zeros((layers, max(iterEstimate, iters + 1)))
				new[:aucs.shape[0], :aucs.shape[1]] = aucs
				aucs = new
				new = np.zeros((layers, max(iterEstimate, iters + 1)))
				new[:varEstimates.shape[0], :varEstimates.shape[1]] = varEstimates
				varEstimates = new
				del new # we don't need it anymore
			for iLevel, (auc, varEstimate, hanleyMcNeil, lowBoot, highBoot, lowSine, highSine, smVAR, npVAR, *estimates, mseTrue, mseEmperic, compLen, minSeps, pc) in enumerate(iteration):
				# store results
				varEstimates[iLevel][iters - 1] = varEstimate
				aucs[iLevel][iters - 1] = auc

				# add to running total
				avgHanleyMNeil[iLevel] += hanleyMcNeil
				avgMSEEmperic[iLevel] += mseEmperic
				avgMSETrues[iLevel] += mseTrue
				avgMinSeps[iLevel] += minSeps
				avgComps[iLevel] += compLen
				avgnpVARs[iLevel] += npVAR
				avgSMVAR[iLevel] += smVAR
				avgAUC[iLevel] += auc
				avgPC[iLevel] += pc
				avgErrorBars[iLevel] += [lowBoot, highBoot, lowSine, highSine, 0, 0]
				avgEstimates[layers - iLevel - 1] += estimates

				# update how many bytes were read
				pBar.update(f.tell() - old)
				pBar.desc = f"{iters}/{iterEstimate}, {reshapeCount}, {getsizeof(unpickler)}"
				old = f.tell()

				if justOne:
					break
	# get rid of the first layer because it's not needed and transpose to be the same shape as the others
	avgMinSeps = avgMinSeps.transpose()[:,1:,]

	# iterEstimate can overshoot, so trim them back
	aucs = aucs[:,:iters]
	varEstimates = varEstimates[:,:iters]

	# need to transpose because numpy is weird
	avgErrorBars = np.transpose(avgErrorBars / iters)
	avgAUC = (avgAUC / iters).transpose()

	# divide vectors by iters to get average
	# // avgComps because can't have fraction of a comparison
	avgComps = avgComps // iters
	avgHanleyMNeil /= iters
	avgMSEEmperic /= iters
	avgEstimates /= iters
	avgMSETrues /= iters
	avgMinSeps /= iters
	avgnpVARs /= iters
	avgSMVAR /= iters
	avgPC   /= iters

	# axis=1 is across the simulations
	varEstimate = np.mean(varEstimates, axis=1)
	varAUCnp = np.var(aucs, ddof=1, axis=1)
	stdVarEstimate = np.sqrt(np.var(varEstimates, axis=1, ddof=1))


	# arcsine transform, this isn't currently used

	#remainder = int(bin(length)[3:], 2)
	#thingies = [remainder, length / 2]
	#for _ in range(2, layers):
	#	thingies.append(thingies[-1] // 2)
	#thingies = [1/(2*np.sqrt(thingy)) for thingy in thingies]
	#avgErrorBars[4] = [np.sin(np.arcsin(np.sqrt(auc)) - thingies[i])**2 for i, auc in enumerate(avgAUC)]
	#avgErrorBars[5] = [np.sin(np.arcsin(np.sqrt(auc)) + thingies[i])**2 for i, auc in enumerate(avgAUC)]
	return varEstimate, avgAUC, avgSMVAR, avgnpVARs, avgMSETrues, avgMSEEmperic, avgComps, avgHanleyMNeil, avgErrorBars, avgEstimates, avgMinSeps, varAUCnp, stdVarEstimate, avgPC, iters

def analyzeScale(fileName, names=None):
	"""Analyzes a scale study.
	If names parameter given, filters for only those names"""
	times = list()
	x0 = list()
	x1 = list()
	scores = dict()
	with open(fileName) as f:
		posDir, negDir = f.readline().strip().split()
		for line in f:
			line = line.rstrip().split()
			times.append(float(line[-1]))
			score = int(line[1])
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

def analyzeAFC(fileName):
	times = list()
	with open(fileName) as f:
		for line in f:
			line = line.strip().split()
			times.append(float(line[-1]))
	return times

if __name__ == "__main__":
	test = 1
	if test == 1:
		# first shows the plot of MSEs
		# then shows the 5 plot dashboard for studies
		length = 256
		layers = 8
		varEstimate, avgAUC, avgSMVAR, avgnpVARs, avgMSETrues, avgMSEEmperic, avgComps, avgHanleyMNeil, avgErrorBars, avgEstimates, avgMinSeps, varAUCnp, stdVarEstimate, avgPC, iters = analyze("resultsMergeExponential85", length, layers, bar=True)
		plt.plot(avgComps, avgMSEEmperic, label='emperic')
		plt.plot(avgComps, avgMSETrues, label='true')
		plt.legend()
		plt.show()
		labels = [f'{np.median(list(filter(lambda x: x != 0, avgMinSeps[0]))):3.02f}']
		for val in np.median(avgMinSeps, axis=0)[1:]:
			labels.append(f'{val:3.02f}')
		slopeFirst = (varEstimate[1]/avgHanleyMNeil[1]) - (varEstimate[0]/avgHanleyMNeil[0])
		slopeTotal = slopeFirst / 3
		hanleyMcNeilToVarEstimate = [avgHanleyMNeil[i] * (1 + i * slopeTotal) for i in range(layers)]
		tickLabels = ['0'] + [str(int(n)) for n in avgComps]
		xVals = avgComps

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
			estimate = avgEstimates[layer - 1]
			for i, point in enumerate(estimate):
				pass #uncomment next line and comment this one to draw text where the estimates were
				#ax2.text(layer + 1, point, str(i), fontsize=12, horizontalalignment='center', verticalalignment='center')
		ax2.legend()
		ax2.set_title("Variance Estimate per Layer")

		ax3 = fig.add_subplot(2, 3, 3)
		info = [-1 for i in range(layers - 1)]
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
		step = length / (layers - 1)
		ax5.set_xticks(np.arange(start + (step / 2), end + (step / 2), step))
		ax5.set_xticklabels(tickLabels[1:])
		cbaxes = fig.add_axes([0.91, 0.13, 0.01, 0.31])
		cbar = fig.colorbar(plot, cax=cbaxes)
		ax5.set_title("Average Distance Between Compairisons per ID per Layer")

		plt.subplots_adjust(wspace=0.45)
		plt.show()
	elif test == 2:
		from DylComp import Comparator
		from DylData import continuousScale
		from DylSort import treeMergeSort
		from DylMath import genX0X1, MSE
		from multiprocessing import Pool

		def bootstrapTau(arr):
			ranks = arr[:,np.random.randint(len(arr[0]), size=len(arr[0]))]
			return stats.kendalltau(ranks[0], ranks[1])[0]

		def eachreader(x):
			rA1=x[0][128:(2*128)]
			rA0=x[0][0:128]
			rB1=x[1][128:(2*128)]
			rB0=x[1][0:128]
			sA=ROC1.successmatrix(rA1,rA0)
			sB=ROC1.successmatrix(rB1,rB0)
			return ROC1.unbiasedMeanMatrixVar(sB-sA)

		def permutation(arr, D0, D1):
			indecies = np.random.randint(2, size=len(arr[0]))
			scales = arr[indecies, range(len(arr[0]))]
			afcs = arr[1 - indecies, range(len(arr[0]))]
			x0 = [scales[i] + 1 for i in range(128)]
			x1 = [scales[i] + 1 for i in range(128, 256)]
			scaleROC = ROC1.rocxy(x1, x0)
			x0 = [afcs[i] + 1 for i in range(128)]
			x1 = [afcs[i] + 1 for i in range(128, 256)]
			afcROC = ROC1.rocxy(x1, x0)
			return MSE(None, None, scaleROC, afcROC)[1]

		data, D0, D1 = continuousScale(128, 128)
		results = {	"Reader A":("resGabi/scaleGabi.csv1565102893.2022426", "resGabi/log2.csv", "resGabi/rocs", "resGabi/compGabi.csv"),
					"Reader B":("resFrank/scaleFrank.csv1565098562.1623092", "resFrank/log2.csv", "resFrank/rocs", "resFrank/results.csv"),
					"Reader C":("resDylan/scaleDylan2.csv", "resDylan/log2.csv", "resDylan/rocs", "resultsBackup/resultsDylan.csv")}
		with open("names.txt") as f:
			names = f.read().split()
		ids = list()
		for i, name in enumerate(sorted(names)):
			names[i] = ''.join(filter(lambda x: not x in "',][ ", name))
			ids.append(names[i])
		fig, (scatterAxes, timeAxes) = plt.subplots(ncols=3, nrows=2)
		#masterRanks = np.zeros((len(results.keys()), 2, 256))
		print("reader   \tscale mean\tscale std\tmerge mean\tmerge std")
		print('-'*90)
		for i, (reader, files) in enumerate(results.items()):
			comp = Comparator(data, rand=True)
			comp.learn(files[3])
			for arr, sortstats in treeMergeSort(data[:], comp, statParams=[(D0, D1)], retStats=True):
				pass
			indeciesAFC = [arr.index(i) for i in range(256)]
			x0, x1 = genX0X1(arr, D1, D0)
			x0 = np.array([indeciesAFC[i] for i in range(128)])
			x1 = np.array([indeciesAFC[i] for i in range(128, 256)])
			#print(x0)
			afcSMData = ROC1.successmatrix(x1, x0)
			afcROC = ROC1.rocxy(x1, x0)
			sortstats[0] = ROC1.auc(x1, x0)
			scaleTimes, x0, x1, scoresScale = analyzeScale(files[0], names=names)
			#print(x0,x1)
			scaleSMData = ROC1.successmatrix(x1, x0)

			mergeTimes = analyzeAFC(files[1])
			scaleTimes = list(filter(lambda x: x < 10, scaleTimes))
			mergeTimes = list(filter(lambda x: x < 5, mergeTimes))
			xmax = np.append(scaleTimes, mergeTimes).max()
			kernal = stats.gaussian_kde(scaleTimes)
			xVals = np.linspace(0, xmax, 1000)
			timeAxes[i].fill_between(xVals, kernal(xVals), label="scale", alpha=0.5)
			kernal = stats.gaussian_kde(mergeTimes)
			xVals = np.linspace(0, xmax, 1000)
			timeAxes[i].fill_between(xVals, kernal(xVals), label="merge", alpha=0.5)

			timeAxes[i].legend()
			timeAxes[i].set_ylim(bottom=0)
			timeAxes[i].set_xlim(left=0, right=xmax)
			#timeAxes[i].set_xscale("log")
			timeAxes[i].set_ylabel("percentage")
			timeAxes[i].set_xlabel("time")

			ranks = np.zeros((2, 256))
			#print(*enumerate(sorted(names)) )
			for x, name in enumerate(sorted(names)):
				ranks[0, x] = scoresScale[name]
				ranks[1, x] = indeciesAFC[x]
			ranks[0] = stats.rankdata(ranks[0])
			ranks[1] = stats.rankdata(ranks[1])
			#for x, name in enumerate(sorted(names)):
			#	timeAxes[i].scatter(scoresScale[name], ranks[0, x], c='b', s=x/256)
			#timeAxes[i].set_xlabel("score")
			#timeAxes[i].set_ylabel("rank")
			#timeAxes[i].set_aspect('equal', 'box')

			scaleROC = ROC1.rocxy(x1, x0)

			x0 = ranks[0][0:128]
			x1 = ranks[0][128:256]
			#print(x0,x1)
			scaleSM = ROC1.successmatrix(x1, x0)
			scaleAUC = ROC1.auc(x1, x0)
			x0 = ranks[1][0:128]
			x1 = ranks[1][128:256]
			#print(x0)
			afcSM = ROC1.successmatrix(x1, x0)
			afcAUC = ROC1.auc(x1, x0)

			mse = MSE(None, None, scaleROC, afcROC)[1]

			scatterAxes[i].plot([0, 256], [0, 256], 'r:')
			for x in range(256):
				scatterAxes[i].scatter(ranks[0][x], ranks[1][x], c='b', s=1)
			scatterAxes[i].set_aspect('equal', 'box')
			tau = stats.kendalltau(ranks[0], ranks[1])[0]
			scatterAxes[i].set_title(reader)
			scatterAxes[i].set_xlabel("scale")
			scatterAxes[i].set_ylabel("AFC")

			with Pool(8, initializer=np.random.seed) as p:
				taus = p.map(bootstrapTau, (ranks for _ in range(1_000)))
				mses = p.starmap(permutation, ((ranks, D0, D1) for _ in range(1)))
			#tauAxes[i].hist(taus, alpha=0.5, color='red', density=True)
			#xmax = max(taus)
			#kernal = stats.gaussian_kde(taus)
			#xVals = np.linspace(0, xmax, 1000)

			#mseAxes[i].hist(mses, alpha=0.5, color='red', density=True)
			#print(mses)
			p = np.mean([mse < permutted for permutted in mses])
			#xmax = max(mses)
			#kernal = stats.gaussian_kde(mses)
			#xVals = np.linspace(0, xmax, 1000)

			dSM = afcSM - scaleSM
			varDSM = ROC1.unbiasedMeanMatrixVar(dSM)
			dAUC = np.mean(afcSM) - np.mean(scaleSM)
			stdDSM = np.sqrt(varDSM)
			z = np.abs(dAUC / stdDSM) # how many standard deviations away it is
			wald = 2 * (1 - stats.norm.cdf(z))
			#print(dAUC, varDSM)
			scatterAxes[i].set_title(f"$\\tau_k={tau:0.3f}\\pm{np.std(taus):0.3f}$")
			print(f"{reader} & ${np.mean(scaleTimes):0.3f}\\pm{np.std(scaleTimes):0.3f}$ & ${np.mean(mergeTimes):0.3f}\\pm{np.std(mergeTimes):0.3f}$\t{np.std(taus):0.7f}")
			#print(reader, wald, sep='\t')
			#tauAxes[i].imshow(scaleSM - scaleSMData)
			#tauAxes[i].set_title(np.mean(scaleSM - scaleSMData))
			#mseAxes[i].imshow(afcSM - afcSMData)
			#mseAxes[i].set_title(np.mean(afcSM - afcSMData))

			#masterRanks[i] = ranks

		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		plt.subplots_adjust(top=0.957,
							bottom=0.066,
							left=0.039,
							right=0.991,
							hspace=0.161,
							wspace=0.136)
		plt.show()
		#for ranks in masterRanks:
			#print(eachreader(ranks))
