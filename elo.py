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
	
	auc = 0.80
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
	runs = 100; 
	M = runs*N
	rating = np.append(mb.zeros((N, 1)), mb.zeros((N, 1)), axis=0)
	
	cnt = 0
	ncmp = 0

	for run in range(1, runs+1):
		toCompare = mb.zeros((2*N, 1))
	
		if run == 1:
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
	simulation_ELO_targetAUC(200)
	"""from tqdm import tqdm, trange
	from p_tqdm import p_umap

	n = list(range(2, 201, 1))
	resultss = p_umap(simulation_ELO_targetAUC, n)

	with open("res.csv", "w") as f:
		for results in resultss:
			for result in results:
				f.write(result)"""