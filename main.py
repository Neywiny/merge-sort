#!/usr/bin/python3.6
import pickle
import os
import sys
from numpy.random import seed
from tqdm import tqdm
from time import sleep
from multiprocessing import Pool
from warnings import filterwarnings
filterwarnings('ignore')
from DylComp import Comparator
from DylMath import genSep
from DylSort import treeMergeSort
from DylData import continuousScale
def sort(args) -> list:
	"""Performs a sort based on the given args.
	Args is of the format (dist, auc, n0, n1) and is one tuple/list.
	Throws an error if the array did not sort correctly.
	Returns the results."""
	dist, auc, n0, n1 = args
	results = list()
	data, D0, D1 = continuousScale(n0, n1)
	comp = Comparator(data, level=0, rand=True)
	sep = genSep(dist, auc)
	comp.genRand(n0, n1, sep, dist)
	for arr, stats in treeMergeSort(data, comp, [(D0, D1), dist, auc], n=2):
		stats.extend([len(comp), comp.genSeps(), comp.pc[-1]])
		comp.resetPC()
		results.append(stats)
	if arr != sorted(arr, key=lambda x: comp.getLatentScore(x)[0]):
		print(arr)
		print(sorted(arr, key=lambda x: comp.getLatentScore(x)[0]))
		raise AssertionError("did not sort")
	return results

def multiRunner(sorter, sorterName: str, distributions: list=None, aucs: list=None):
	"""calls the given sorter for either the provided distributions and aucs or the command line arguments:
	command line args as: distributions and aucs each separated by commas no spaces, separated by a space
	sorter must take: one argument that equals (unique threadID, distribution, auc, n0, n1)
	sorter must return: a list of results to be pickled and appended to the file"""
	if distributions == None:
		if len(sys.argv) == 6: #distributions and AUCs given
			distributions = sys.argv[2].split(',')
		else:
			distributions = ['normal', 'exponential']
	if aucs == None:
		if len(sys.argv) == 6: #distributions and AUCs given
			aucs = [float(auc) for auc in sys.argv[3].split(',')]
		else:
			aucs = [0.65, 0.85, 0.95]

	for dist in distributions:
		for AUC in aucs:
			results = list()
			iters = int(sys.argv[1])
			if len(sys.argv) == 6:
				ids = [(dist, AUC, int(sys.argv[4]), int(sys.argv[5])) for _ in range(iters)]
			else:
				ids = [(dist, AUC, 128, 128) for _ in range(iters)]
			topBar = tqdm(total=iters, smoothing=0, bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} {remaining}, {rate_fmt}")
			botBar = tqdm(total=iters, smoothing=0, bar_format="{bar}")
			with Pool(initializer=seed) as p:
				for result in p.imap_unordered(sorter, ids):
					topBar.update()
					botBar.update()
					results.append(pickle.dumps(result))
			botBar.close()
			topBar.close()
			print('\n')
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
				with open(f'results{sorterName}{dist.title()}{int(AUC*100)}','ab') as f:
					print("have lock")
					f.writelines(results)
			except BaseException as err:
				print(err)
			finally:
				lock.close()
				os.remove(".lock")

if __name__ == "__main__":
	multiRunner(sort, "Merge")
