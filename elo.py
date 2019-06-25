import numpy as np
from numpy import matlib as mb
from ROC1 import *
from random import shuffle
from scipy.special import erfinv
from math import sqrt
from warnings import filterwarnings
filterwarnings('ignore')

def AUC(x1, x0):
	sm = successmatrix(x1, x0)
	return np.mean(sm), unbiasedMeanMatrixVar(sm)

def simulation_ELO_targetAUC(N):
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
	L = 100; 
	M = L*N
	rating = np.append(mb.zeros((N, 1)), mb.zeros((N, 1)), axis=0)
	

	for l in range(1, L+1):
		vals = mb.zeros((2*N, 1))
	
		if l == 1:
			# option A: only compare + vs -
			arr = list(range(N))
			shuffle(arr)
			vals[0::2] = np.array(arr, ndmin=2).transpose()
			arr = list(range(N, 2 * N))
			shuffle(arr)
			vals[1::2] = np.array(arr, ndmin=2).transpose()
		else:
			# option B: everything is valid
			arr = list(range(2 * N))
			shuffle(arr)
			vals = np.array(arr, ndmin=2).transpose()
	
		cnt = 0
		ncmp = 0
		for i in range(1, 2*N, 2):
			a = int(vals[i - 1])
			b = int(vals[i])
	
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

		#print(f'AUC: {auc1:.3f}\n')
		print(cnt, ncmp, *auc1, sep=',')
if __name__ == '__main__':
	from tqdm import tqdm, trange
	""" from multiprocess import Pool

	with Pool() as p:
		n = list(range(0, 201, 1))
		results = p.map(simulation_ELO_targetAUC, n)
	for i, res in enumerate(results):
		print(i, *res) """

	for n in trange(201):
		simulation_ELO_targetAUC(n)