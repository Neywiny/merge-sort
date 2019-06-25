#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
def unbiasedMeanMatrixVar(sm,df=1):
	# Estimate the unbiased variance of the mean of a 2D matrix with 
	# two way random effects + residuals
	n0,n1 = sm.shape
	x0est=sm.mean(axis=0)  # Means of rows,
	x1est=sm.mean(axis=1)  # Means of columns
	MST=numpy.var(sm,ddof=df)  
	MSA=n1*numpy.var(x1est,ddof=df)
	MSB=n0*numpy.var(x0est,ddof=df)
	ev=( (n0*n1-1)*MST - (n0-1)*MSA-(n1-1)*MSB)/((n0-1)*(n1-1))
	sig1=(MSA-ev)/n1;   sig2=(MSB-ev)/n0
	vout=(MSA+MSB-ev)/n0/n1
	return vout

def successmatrix(x1,x0):
	# Create a "success" matrix from a vector of signal present scores (x1)
	# and signal-absent scores (x0).
	b=x1.copy()
	b.shape=(-1,1)
	dm=b-x0
	return((numpy.sign(dm)+1.)/2)  # Success matrix -- actual kernel

def unbiasedAUCvar(x1, x0,df=1): # unbiased estimate of AUC variance.
	sm=successmatrix(x1,x0)
	return(unbiasedMeanMatrixVar(sm,df))

