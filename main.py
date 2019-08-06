#!/usr/bin/python3.6
import math
import pickle
import os
import sys
from multiprocessing import Pool
from warnings import filterwarnings
from DylComp import Comparator
from DylMath import *
from DylSort import mergeSort, treeMergeSort
def sort(tid, i=0, seed=None):
	results = list()
	data, D0, D1 = continuousScale(128, 128)
	comp = Comparator(data, level=0, rand=True, seed=seed)
	comp.genRand(len(D0), len(D1), sep, dist)
	for arr, stats in treeMergeSort(data, comp, [(D0, D1)], retStats=True, n=2):
		stats.extend([len(comp), comp.genSeps(), comp.pc[-1]])
		comp.pc = list()
		comp.c = 0
		results.append(stats)
	if arr != sorted(arr, key=lambda x: comp.getLatentScore(x)[0]):
		print(arr)
		print(sorted(arr, key=lambda x: comp.getLatentScore(x)[0]))
		raise EOFError("did not sort")
	return results
if __name__ == "__main__":
	filterwarnings('ignore')
	test = 3
	if test == 1:
		lMax: int = 2**8
		iters: int = 1
		levelMax: int = 0
		#data, D0, D1 = continuousScale("sampledata.csv")
		data = continuousScale(lMax)
		#print(data)
		lMax = len(data)
		#exit()
		with open("sample_results.csv", "w") as f:
			level: int = levelMax
			sorter = mergeSort
			if True:
			#for sorter in tqdm((mergeSort, combsort)):
			#for n in range(levelMax + 1):
			#	counts[n] = list((0, 0, 0))
				if True:
				#for level in trange(levelMax + 1):
					for i in trange(iters):
						known:dict = dict()
						arrs = [data[:]]
						comp = Comparator(data, level=0, rand=True)
						for arr in tqdm(sorter(data, comp=comp), total=math.ceil(math.log(lMax, 2))):
							#f.write(str(list(comp.minSeps.values())) + ', ' + str(unbiasedMeanMatrixVar(successMatrix(data))) + '\n')
							arrs.append(data[:])
						#for key, value in comp.counts.items():
						#	f.write(str(key))
						#	f.write(",")
						#	f.write(str(value))
						#	f.write(",")
						#	f.write(str(comp.minSeps[key]))
						#	f.write("\n")
		#print(arrs)
		graphROCs(arrs)
	elif test == 2:
		from DylData import continuousScale
		power = 8
		data = continuousScale(2**power)
		arrays = [data[:]]
		#graphROC(data)
		#print(successMatrix(data))
		comp = Comparator(data, level=0, rand=True)
		comp.genRand(2**(power - 1), 2**(power - 1), 1.7, 'normal')
		#comp.bRecord = False
		for _ in tqdm(mergeSort(data, comp=comp), total=power):
			arrays.append(data[:])
			#print(successMatrix(data))
		graphROCs(arrays, withLine=True,withPatches=False, D0=list(range(2**(power - 1))), D1=range(2**(power - 1), 2**(power)))
	elif test == 3:
		from tqdm import tqdm
		from time import sleep
		from scipy.stats import norm
		for dist in ['normal', 'exponential']:
			for AUC in [0.65, 0.85, 0.95]:
				if dist == 'normal':
					sep = norm.ppf(AUC)*(2**0.5)
				elif dist == 'exponential':
					sep = abs(AUC/(1-AUC))
				results = list()
				if len(sys.argv) > 1:
					iters = int(sys.argv[1])
					if len(sys.argv) < 3:
						sys.argv.append(AUC)
						sys.argv.append(dist)
					else:
						sys.argv[-2] = AUC
						sys.argv[-1] = dist
					ids = [*range(iters)]
					topBar = tqdm(total=iters, smoothing=0, bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} {remaining}, {rate_fmt}")
					botBar = tqdm(total=iters, smoothing=0, bar_format="{bar}")
					with Pool() as p:
						for result in p.imap_unordered(sort, ids):
							topBar.update()
							botBar.update()
							results.append(pickle.dumps(result))
					botBar.close()
					topBar.close()
					print('\n')
				else:
					retMid = False
					iters = 100
					results = [pickle.dumps(sort(0, i)) for i in range(iters)]
				#change output file if requested to do so
				print("waiting for lock")
				locked = False
				while not locked:
					try:
						lock = open(".lock", "x")
						print("made lock")
						locked = True
					except FileExistsError:
						sleep(0.1)
				try:
					with open(f'test{dist.title()}{int(AUC*100)}','ab') as f:
						print("have lock")
						f.writelines(results)
				except BaseException as err:
					print(err)
				finally:
					lock.close()
					os.remove(".lock")
	elif test == 4:
		from random import shuffle
		import numpy as np
		import matplotlib.pyplot as plt
		from matplotlib.patches import Rectangle
		import matplotlib
		font = {'size' : 24}
		matplotlib.rc('font', **font)
		power = 9
		length = int(2**power*(2/3))
		power += 1
		yStep = length / power
		print(yStep)
		fig, axes = plt.subplots(ncols=2, nrows=1)
		for ax in axes:
			if ax == axes[0]:
				color = 'r'
				sorter = mergeSort
			else:
				color = 'lime'
				sorter = treeMergeSort
			data = list(range(int(2**(power - 1)*(2/3))))
			img = np.zeros((power, length))
			shuffle(data)
			img[0] = data[:]
			comp = Comparator(data, level=0)
			for gIndex, group in enumerate(data):
				ax.add_patch(Rectangle((gIndex, (power) * yStep), 1, -yStep, color=color, lw=1/power, fill=False))
			for y, groups in enumerate(sorter(data, comp=comp, combGroups=False), start=1):
				arr = []
			x = 0
			for gIndex, group in enumerate(groups):
				ax.add_patch(Rectangle((x, (power - y) * yStep), len(group), -yStep, color=color, lw=3*y/power, fill=False))
				arr.extend(group)
				x += len(group)
			if len(arr) < len(img[0]):
				arr.extend([0 for i in range(len(img[0]) - len(arr))])
			img[y] = arr
			ax.imshow(img, cmap='Greys', extent=[0, length, 0, length], aspect=1)
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_ylabel("Layer")
			ax.set_xlabel("Position")
		plt.show()
