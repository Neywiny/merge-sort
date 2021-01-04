#!/usr/bin/python3.6
import numpy as np
from numpy import matlib as mb
from ROC1 import rocxy, successmatrix, unbiasedMeanMatrixVar
from warnings import filterwarnings
from DylMath import MSE, genSep
from sys import argv
from tqdm import trange, tqdm

filterwarnings('ignore')

def simulation_ELO_targetAUC(args: list, rounds: int=14, retRoc=False):
	"""Args is of the form (dist, auc, n0, n1).
	Rounds is how many rounds of (n0 + n1)/2 comparisons it will so.
	retRoc determines if the function will return its ROC curve or its statistics.

	@Author: Francesc Massanes (fmassane@iit.edu)
	@Version: 0.1 (really beta)"""
	
	dist, auc, n0, n1 = args
	if n0 != n1:
		raise NotImplementedError("n0 must equal n1")
	N = n0
	##
	# DATA GENERATION
	#
	K1 = 400
	K2 = 32

	sep = genSep(dist, float(auc))
	if dist == 'exponential':
		neg = np.random.exponential(size=(N, 1))
		plus = np.random.exponential(scale=sep, size=(N, 1))
	elif dist == 'normal':
		neg = np.random.normal(0, 1, (N, 1))
		plus = np.random.normal(sep, 1, (N, 1))
	else:
		print("invalid argument", dist)
		return None

	x0 = np.array(neg)[:,0]
	x1 = np.array(plus)[:,0]
	empiricROC = rocxy(x1, x0)
	scores = np.append(neg, plus)
	truth = np.append(mb.zeros((N, 1)), mb.ones((N, 1)), axis=0)

	rating = np.append(mb.zeros((N, 1)), mb.zeros((N, 1)), axis=0)
	pc = list()
	cnt = 0
	ncmp = 0
	results = list()
	for eloRound in range(1, rounds+1):
		toCompare = mb.zeros((2*N, 1))
		if eloRound == 1:
			# option A: only compare + vs -
			arr = list(range(N))
			#np.random.shuffle(arr)
			toCompare[0::2] = np.array(arr, ndmin=2).transpose()
			arr = list(range(N, 2 * N))
			#np.random.shuffle(arr)
			toCompare[1::2] = np.array(arr, ndmin=2).transpose()
		else:
			# option B: everything is valid
			arr = list(range(2 * N))
			np.random.shuffle(arr)
			toCompare = np.array(arr, ndmin=2).transpose()
		for i in range(1, 2*N, 2):
			a = int(toCompare[i - 1])
			b = int(toCompare[i])
			QA = 10**(int(rating[a]) / K1)
			QB = 10**(int(rating[b]) / K1)
			EA = QA / (QA+QB)
			EB = QB / (QA+QB)
			if scores[a] < scores[b]:
				SA = 0
				SB = 1
			else:
				SA = 1
				SB = 0
			if bool(truth[a]) ^ bool(truth[b]):
				if ( SA == 1 and truth[a] == 1 ):
					cnt = cnt + 1
				if ( SB == 1 and truth[b] == 1 ):
					cnt = cnt +1
				pc.append(cnt / (len(pc) + 1))
			ncmp = ncmp+1
			rating[a] = rating[a] + K2 * ( SA - EA )
			rating[b] = rating[b] + K2 * ( SB - EB )

		x0 = np.array(rating[0:N])[:,0]
		x1 = np.array(rating[N:])[:,0]
		roc = rocxy(x1, x0)
		if retRoc:
			results.append((roc, ncmp))
		else:
			sm = successmatrix(x1, np.transpose(x0))
			auc, var = np.mean(sm), unbiasedMeanMatrixVar(sm)
			mseTruth, mseEmpiric, auc = MSE(sep, dist, roc, empiricROC)
			results.append((N, cnt, ncmp, var, auc, mseTruth, mseEmpiric, pc[-1]))
	return results
if __name__ == '__main__':
	if len(argv) > 1:
		test = 1
	else:
		test = 3
	if test == 1:
		from main import multiRunner
		multiRunner(simulation_ELO_targetAUC, "Elo")
	elif test == 2:
		animation = False
		if animation:
			import matplotlib.pyplot as plt
			from matplotlib.animation import FuncAnimation
			from matplotlib.animation import PillowWriter
			frames = 100
			results = simulation_ELO_targetAUC(('normal', 0.8853, 128, 128), rounds=frames)
			fig, ax = plt.subplots()
			fig.set_tight_layout(True)
			pbar = tqdm(total=len(results))
			def update(i):
				"""Update the frame"""
				pbar.update()
				label = f"timestep {i}"
				ax.clear()
				roc = rocxy(*results[i])
				ax.plot(roc['x'], roc['y'])
				ax.set_title(f"{i:02d}")
				return label, ax
			anim = FuncAnimation(fig, update, frames=np.arange(0, frames), interval=100)
			anim.save("rocs.gif", writer=PillowWriter(fps=10))
			pbar.close()
		else:
			import matplotlib.pyplot as plt
			from apng import APNG
			from DylSort import treeMergeSort
			from DylComp import Comparator
			from DylData import continuousScale
			from DylMath import genROC, avROC
			seed = 15
			data, D0, D1 = continuousScale(128, 128)
			comp = Comparator(data, level=0, rand=True, seed=seed)
			comp.genRand(len(D0), len(D1), 7.72, 'exponential')
			np.random.seed(seed)
			im = APNG()
			fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
			fig.suptitle("Pass number, MSE true, MSE empirical")
			x = np.linspace(0, 1, num=200)
			y = x**(1/7.72)
			ax1.set_aspect('equal', 'box')
			ax2.set_aspect('equal', 'box')
			elo = simulation_ELO_targetAUC(True)
			merge = treeMergeSort(data, comp, statParams=[(D0, D1)], combGroups=False)
			plt.tight_layout()
			for i in trange(8):
				roc, mseTheo, mseEmp, empiricROC = next(elo)
				ax1.plot(x, y, linestyle='--', label='true', lw=3)
				ax1.plot(empiricROC['x'], empiricROC['y'], linestyle=':', lw=2, label='empirical')
				ax1.plot(roc['x'], roc['y'], label='predicted')
				ax1.legend(loc=4)
				ax1.set_title(f"ELO\n{i+1}, {mseTheo[0]*1000:02.3f}E(-3), {mseEmp[0]*1000:02.3f}E(-3)")
				groups = next(merge)
				rocs = []
				for group in groups:
					rocs.append(genROC(group, D0, D1))
				roc = avROC(rocs)
				mseTheo, mseEmp, auc = MSE(7.72, 'exponential', zip(*roc)), MSE(7.72, 'exponential', zip(*roc), zip(empiricROC['x'], empiricROC['y']))
				ax2.plot(x, y, linestyle='--', label='true', lw=3)
				ax2.plot(empiricROC['x'], empiricROC['y'], linestyle=':', lw=2, label='empirical')
				ax2.plot(*roc, label='predicted')
				ax2.legend()
				ax2.set_title(f"merge\n{i+1}, {mseTheo[0]*1000:02.3f}E(-3), {mseEmp[0]*1000:02.3f}E(-3)")
				plt.savefig("both")
				im.append_file("both.png", delay=1000)
				ax1.clear()
				ax2.clear()
			im.save("both.png")
	elif test == 3:
		from DylComp import Comparator
		from DylSort import treeMergeSort
		from DylData import continuousScale
		from DylMath import genROC, avROC
		import matplotlib.pyplot as plt
		data, D0, D1 = continuousScale(128, 128)
		comp = Comparator(data, rand=True, level=0, seed=20)
		results = [res for res in simulation_ELO_targetAUC(('normal', 0.8853, 128, 128), retRoc=True)]
		mergeResults = list()
		for groups in treeMergeSort(data, comp, combGroups=False):
			rocs = list()
			for group in groups:
				roc = genROC(group, D1, D0)
				rocs.append(roc)
			mergeResults.append((avROC(rocs), len(comp)))
		matches = [(np.argmin([abs(res[1] - mergeLen) for res in results])) for (groups, mergeLen) in mergeResults]
		fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True)
		for axn, layer in enumerate([0, 3, 5, 7]):
			axes[axn][0].plot(mergeResults[layer][0][1], mergeResults[layer][0][0], 'r')
			axes[axn][1].plot(results[matches[layer]][0]['x'], results[matches[layer]][0]['y'], 'r')
			axes[axn][0].plot(comp.empiricROC()['x'], comp.empiricROC()['y'], 'b:')
			axes[axn][1].plot(comp.empiricROC()['x'], comp.empiricROC()['y'], 'b:')
			axes[axn][0].set_aspect('equal', 'box')
			axes[axn][1].set_aspect('equal', 'box')
			axes[axn][0].set_title(str(mergeResults[layer][1]))
			axes[axn][1].set_title(str(results[matches[layer]][1]))
		fig.suptitle("Merge    Massanes")
		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		fig.subplots_adjust(hspace=0.24)
		for i in range(10):
			fig.tight_layout()
		plt.show()