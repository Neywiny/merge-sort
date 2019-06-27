import numpy as np
from numpy import matlib as mb
from ROC1 import *
from scipy.special import erfinv
from math import sqrt
from warnings import filterwarnings
filterwarnings('ignore')

def AUC(x1, x0):
	sm = successmatrix(x1, np.transpose(x0))
	return np.mean(sm), unbiasedMeanMatrixVar(sm)

def simulation_ELO_targetAUC(N):
	results = []
	N = 256
	##
	# Function for the simulation of ELO rating given an AUC of 0.8 (most of it, hard-coded), 
	# the input to the function is N (the number of samples on the rating study).
	#
	#
	# @Author: Francesc Massanes (fmassane@iit.edu)
	# @Version: 0.1 (really beta)

	## 
	# DATA GENERATION 
	#
	
	auc = 0.8
	K = ( 2 * erfinv(2*auc-1) ) ** 2
	K1 = 400
	K2 = 32

	
	mu1 = 1
	mu0 = 0
	si0 = 0.84
	si1 = sqrt( 2*((mu1-mu0)**2/K - si0**2 /2 ) )
	
	
	neg = np.random.normal(mu0, si0, (N, 1))
	plus = np.random.normal(mu1, si1, (N, 1))
	
	scores = np.append(neg, plus)
	truth = np.append(mb.zeros((N, 1)), mb.ones((N, 1)), axis=0)
	
	
	
	AUC_orig = AUC(neg, plus)
	#print(f'AUC original: {AUC_orig:.3f}\n')
	
	#
	## 
	# PRE-STABLISHED COMPARISONS
	#
	rounds = 100; 
	M = rounds*N
	rating = np.append(mb.zeros((N, 1)), mb.zeros((N, 1)), axis=0)
	
	cnt = 0
	ncmp = 0

	for round in range(1, rounds+1):
		toCompare = mb.zeros((2*N, 1))
	
		if round == 1:
			# option A: only compare + vs -
			arr = list(range(N))
			np.random.shuffle(arr)
			toCompare[0::2] = np.array(arr, ndmin=2).transpose()
			arr = list(range(N, 2 * N))
			np.random.shuffle(arr)
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

			if ( SA == 1 and truth[a] == 1 ):
				cnt = cnt + 1
			if ( SB == 1 and truth[b] == 1 ):
				cnt = cnt +1
			ncmp = ncmp+1
	
			rating[a] = rating[a] + K2 * ( SA - EA )
			rating[b] = rating[b] + K2 * ( SB - EB )
	
		x0 = rating[0:N]
		x1 = rating[N:(2*N) + 1]
		auc1 = AUC(x1, x0)

		results.append(f"{N}, {cnt}, {ncmp}, {auc1[0]}, {auc1[1]}\n")
	return results
if __name__ == '__main__':
	test = 1
	if test == 1:
		#simulation_ELO_targetAUC(200)
		from tqdm import tqdm, trange
		from p_tqdm import p_umap

		n = list(range(1000))
		resultss = p_umap(simulation_ELO_targetAUC, n)

		with open("res.csv", "w") as f:
			for results in resultss:
				for result in results:
					f.write(result)
	elif test == 2:
		import matplotlib.pyplot as plt

		mavgVAR = [0.00079233, 0.00062563, 0.00053144, 0.00048861, 0.00046847, 0.00045563, 0.00045076, 0.00044814]
		mVar = [0.0007922935327750722, 0.0006219975781022436, 0.0005373716225063664, 0.000494514994197877, 0.00047087097180417714, 0.00046050219813277343, 0.0004541607057533005, 0.00045509292906403695]
		mAUC = [0.8854691354851973, 0.8855060778166118, 0.8856959292763158, 0.885616663882607, 0.8856648093775699, 0.8856661344829359, 0.885683250427246, 0.8856790893956235]
		mComp = [128.0, 291.8201480263158, 494.91825657894736, 722.3478618421053, 963.271052631579, 1211.5181743421053, 1463.5886513157895, 1717.5917763157895]

		aucs = dict()
		vars = dict()

		for i in range(256, 25860, 256):
			aucs[i] = list()
			vars[i] = list()

		with open("res.csv") as f:
			for line in f:
				line = line.split(", ")
				line[2] = int(line[2])
				aucs[line[2]].append(float(line[3]))
				vars[line[2]].append(float(line[4]))

		avgAUC = [np.mean(aucs[layer]) for layer in aucs.keys()]
		varAUC = [np.var(aucs[layer], ddof=1) for layer in aucs.keys()]
		avgVAR = [np.mean(vars[layer]) for layer in aucs.keys()]


		fig = plt.figure()
		ax1 = fig.add_subplot(1, 2, 1)
		ax1.errorbar(list(range(256, 2048, 256)), avgAUC[:7],c='r', ls='-', yerr=np.sqrt(avgVAR[:7]), capsize=10, label='elo')
		ax1.plot(mComp, mAUC, 'b.-', label='merge')
		ax1.legend()
		ax1.set_title("AUC")

		ax2 = fig.add_subplot(1, 2, 2)
		ax2.plot(list(range(256, 2048, 256)), varAUC[:7], 'r.-', label='elo var of auc')
		ax2.plot(list(range(256, 2048, 256)), avgVAR[:7], 'r.--', label='elo mean of var')
		ax2.plot(mComp, mVar, 'b.-', label='merge var of auc')
		ax2.plot(mComp, mavgVAR, 'b.--', label='merge mean of var')
		ax2.legend()
		ax2.set_title("VAR")
		plt.show()


