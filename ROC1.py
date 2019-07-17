#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
def table(x,levels=None):   # Like the R table function
        if not levels: levels=set(x) 
        tbl={}
        for i in levels: tbl[i]=0
        for i in x: tbl[i]+=1
        return(tbl)   # Returns a table that is a dictionary.

def numtable(x,levels=None):
        # Create a numerical table that is a numpy array.
        tbl=numpy.array(list(table(x,levels).items()),
                        dtype=[('value','f4'),('number','i4')])
        tbl.sort(order='value')  # put the table in order by value
        return(tbl)

def rocxy(apos,aneg):  # Data points for an empirical ROC curve.
        def cmsm(x,lvls):
                return 1.-numpy.append(0,numpy.cumsum(
                          numtable(x,levels=lvls)['number']))/float(len(x))
        lvls= set(apos) | set(aneg)
        return { 'x':cmsm(aneg,lvls),'y':cmsm(apos,lvls) }
def auc(apos,aneg):  # Calculate AUC

        n=len(aneg)

        return(1.-(numpy.sum(stats.rankdata(numpy.c((aneg,apos)))[0:n])-

               n*(n+1)/2.)/n/len(apos))
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

