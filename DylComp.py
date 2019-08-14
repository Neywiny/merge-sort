#! /usr/bin/python3
import numpy as np
from scipy.stats import kendalltau
from ROC1 import rocxy
np.seterr(all="ignore")
from warnings import filterwarnings
filterwarnings("ignore")
import socket
from numpy.random import random
from pickle import dump
class Comparator:
	"""A class for comparing 2 values.
	Controlled with the optimizaiton level and if you want random decisions or not
	Either provide objects in init or call .genLookup before you do any comparing
	Optimization levels: will not optimize, store result, do abc association, do recursive association
	Rand: defaults to False. If True will create data from random distributions with seed parameter
	"""
	def __init__(self, objects: list = None, level: int = 3, rand: bool=False, seed: int=None):
		self.clearHistory()
		self.rand: bool = rand
		self.seed: int = seed
		self.level: int = level
		self.compHistory: list = list()
		self.dupCount: int = 0
		self.optCount: int = 0
		self.counts: dict = dict()
		self.seps: dict = dict()
		self.bRecord: bool = True
		if objects != None:
			self.genLookup(objects)
			if rand:
				self.n0 = self.n1 = len(objects) // 2
				self.n1 += len(objects) % 2
				self.genRand(self.n0, self.n1, 1.7, 'normal')
		self.last: tuple = None
		self.resetPC()
		self.desHist: list = list()

	def __len__(self) -> int:
		"""Returns either the number of comparisons done"""
		return self.compHistory if isinstance(self.compHistory, int) else len(self.compHistory)

	def resetPC(self):
		"""Resets the pc statistics. Call this once per layer if you only want that layer's PC"""
		self.c: int = 0
		self.pc: list = list()

	def kendalltau(self, predicted: list) -> float:
		"""Returns the kendalltau statistic between the predicted image ID ordering and the true ordering of the image IDs with respect to latent score.
		This method filters image IDs by what's in predicted, so only the ids in predicted are used."""
		return kendalltau(self.getLatentScore(predicted), list(filter(lambda x: x in self.getLatentScore(predicted), sorted(self.vals))))[0]

	def genRand(self, n0: int, n1: int, sep: float, dist: str):
		"""Generates the random data. If a seed has not previously been provided, it will be assigned here.
		This new seeding may not work on Windows, so Windows users should assign the seed on their own."""
		# get a random seed for each node and each process on that node, and the time
		self.n0: int = n0
		self.n1: int = n1
		if self.seed == None:
			from os import getpid, uname
			from time import time
			self.seed: int = (int(str(ord(uname()[1][-1])) + str(getpid()) + str(int(time()))) % 2**31)
		np.random.seed(self.seed)
		if dist == 'normal':
			self.vals: tuple = (tuple(np.random.normal(size=n0,loc=0)) + tuple(np.random.normal(size=n1,loc=sep)))
		elif dist == 'exponential':
			self.vals: tuple = (tuple(np.random.exponential(size=n0,scale=1)) + tuple(np.random.exponential(size=n1,scale=sep)))
		else:
			raise NotImplementedError("distibution must be one of ['normal','exponential']")

	def empiricROC(self) -> dict:
		"""Generates and stores the empiric ROC if it needs to. 
		Returns the stored ROC curve."""
		empiric: dict = getattr(self, 'empiric', None)
		if empiric == None:
			self.empiric = rocxy(self.vals[self.n0:], self.vals[:self.n0])
		return self.empiric

	def record(self, vals: list):
		"""Record that these values were seen.
		This is automatically called by min and max."""
		if not self.bRecord:
			return
		for val in vals:
			self.counts[val] += 1
			#count minimum separations
			self.seps[val].append(len(self))
			if self.last:
				if val in self.last:
					self.dupCount += 1
		self.compHistory.append(tuple(vals))
		self.last = tuple(vals)

	def getLatentScore(self, imgID: int) -> float:
		"""gets the latent score of a given imgID or array of imgIDs.
		If only one index is provided, also returns if the image is from the disease negative distribution."""
		if isinstance(imgID, (tuple, list)):
			return [self.getLatentScore(val)[0] for val in imgID]
		if self.rand:
			return self.vals[imgID], imgID < self.n0
		else:
			return imgID

	def genSeps(self) -> list:
		"""Goes throguh the stored records and returns a list of the minimum separations.
		If there is no minimum separation (the image has not been seen more than once), uses 2*(n0+n1) as a palceholder"""
		minseps: list = [2*len(self.objects) for i in range(len(self.objects))]
		for img, times in self.seps.items():
			if len(times) > 1:
				minseps[img]: int = min(map(lambda x: times[x + 1] - times[x], range(len(times) - 1)))
		return minseps

	def genLookup(self, objects: list):
		"""Generate the lookup table for each object provided"""
		self.lookup:dict = dict()
		self.objects: list = objects
		for datum in objects:
			self.lookup[datum]: dict = dict()
		self.clearHistory()

	def clearHistory(self):
		"""Clears the history statistics of comparisons"""
		if hasattr(self, "objects"):
			self.compHistory: list = list()
			self.last: tuple = None
			self.dupCount: int = 0
			for datum in self.objects:
				self.counts[datum]: int = 0
				self.seps[datum]: list = list()

	def learn(self, arr: list, img: int=None, maxi: bool=False):
		"""Learn the order of the array provided, assuming the current optimization level allows it
		if img is provided, learns the arr w.r.t. the img and if it is max or min. arr can also be
		a filename, in whichcase it will read the file to learn"""
		if isinstance(arr, str):
			with open(arr) as f:
				f.readline()
				for line in f:
					line: list = line.rstrip().replace(' ,', ', ').split(', ')
					if len(line) == 3: # valid comparison
						self.learn([int(line[0]), int(line[1])], int(line[2]), maxi=True)
		else:
			if img == None and self.level > 1:
				for i, a in enumerate(arr):
					for b in arr[i + 1:]:
						self.lookup[a][b]: bool = True
						self.lookup[b][a]: bool = False
						if self.level > 2:
							Comparator.optimize(self.objects, self.lookup, True, a, b)
			elif img != None and self.level > 1:
				for b in arr:
					if b != img:
						self.lookup[img][b]: bool = not maxi
						self.lookup[b][img]: bool = maxi
						if self.level > 2:
							Comparator.optimize(self.objects, self.lookup, maxi, b, img)

	def max(self, arr, tryingAgain=False) -> (int, int):
		"""Gets the maximum of the array with respect to the latent scores.
		tryingAgain should always be False unless a network comparator is used.
		Returns the undex of the maximum ID and the maximum ID."""
		if len(arr) == 0 or tryingAgain:
			raise NotImplementedError("I can't take the max of nothing")
		if len(arr) == 2:
			a,b = arr
			if b in self.lookup[a].keys():
				# cache hit
				if self.lookup[a][b]: # a < b
					return 1, b
				else:
					return 0, a
			elif a in self.lookup[b].keys():
				# cache hit
				if self.lookup[b][a]:
					return 0, a
				else:
					return 1, b
		self.record(arr)
		maxVal: int = arr[0]
		maxScore: float = self.getLatentScore(arr[0])[0] if self.rand else arr[0]
		maxInd: int = 0
		for i, imageID in enumerate(arr[1:], start=1):
			score = self.getLatentScore(imageID)[0] if self.rand else arr[i]
			if score > maxScore:
				maxInd: int = i
				maxVal: int = imageID
				maxScore: float = score
		self.learn(arr, maxVal, True)
		self.updatePC(arr, maxVal, max(arr))
		self.desHist.append(maxVal)
		return maxInd, maxVal

	def min(self, arr) -> (int, int):
		"""Gets the minimum of the array with respect to the latent scores.
		Returns the undex of the minimum ID and the minimum ID."""
		if len(arr) == 0:
			raise NotImplementedError("I can't take the min of nothing")
		if len(arr) == 2:
			a,b = arr
			if b in self.lookup[a].keys():
				# cache hit
				if self.lookup[a][b]: # a < b
					return 0, a
				else:
					return 1, b
			elif a in self.lookup[b].keys():
				# cache hit
				if self.lookup[b][a]:
					return 1, b
				else:
					return 0, a
		self.record(arr)
		minVal: int = arr[0]
		minScore: float = self.getLatentScore(arr[0])[0] if self.rand else arr[0]
		minInd: int = 0
		for i, imageID in enumerate(arr[1:], start=1):
			score = self.getLatentScore(imageID)[0] if self.rand else arr[i]
			if score < minScore:
				minInd: int = i
				minVal: int = imageID
				minScore: float = score
		self.learn(arr, minVal, False)
		self.updatePC(arr, minVal, min(arr))
		self.desHist.append(arr[int(not minInd)])
		return minInd, minVal

	def updatePC(self, arr: list, guess, answer):
		"""If the ids in arr are from different distibutions, adds 1 to the pc denominator.
		If the guess was the answer, adds 1 to the pc numerator."""
		if self.rand and (arr[0] < self.n0) ^ (arr[1] < self.n0):
			if guess == answer:
				self.c += 1
			self.pc.append(self.c / (len(self.pc) + 1))

	@staticmethod
	def optimize(objects: list, lookup: dict, res: bool, a, b) -> int:
		"""Recursive optimization algorithm for adding a node to a fully connected graph.
		Returns the number of optimizations it did."""
		if objects:
			nObjects: list = []
			for c in list(lookup[b]):
			# for all c s.t. c is a neighbor of b
				if c in objects and lookup[b][c] == res and c != a and c not in lookup[a]:
					# s.t. a > b > c or a < b < c
					nObjects.append(c)
					# print("optimized", a, c)
					lookup[a][c]:bool = res
					lookup[c][a]:bool = not res
					return 1 + Comparator.optimize(nObjects, lookup, res, b, c)
		return 0

class NetComparator(Comparator):
	"""A class for doing comparisons over a network."""
	# keep payloads to 10 bytes, try for little endian
	# 'op codes'
	# cmd -> [0010, 8 bytes, 0011]
	# max -> [0010 (image 1 32 bits) (image 2 32 bits) 0011], receive 2 32 bit ints denoting index and val respectively
	def __init__(self, ip: str, port: int, recorder=None, objects: list = None, level: int = 3):
		super(NetComparator, self).__init__(objects, level)
		self.ip: str = ip
		self.port: int = port
		self.currLayer: int = 1
		self.aucs: list = list()
		self.recorder = recorder
		self.recorder.write('Image 1,Image 2,Chosen\n')
		self.desHist: list = list()
		self.plots: list = list()
		
	def __enter__(self):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		print("getting connected")
		self.s.bind(('', self.port))
		self.s.listen(1)
		conn, addr = self.s.accept()
		print('Connection address:', addr)
		print("connection established")
		conn.send(b"I'm ready!")
		data: bytes = conn.recv(10)
		if data[0] == 2 and data[9] == 3: #valid frame
			self.n0, self.n1 = int.from_bytes(data[1:5], 'little'), int.from_bytes(data[5:9], 'little')
			print("go flight", self.n0, self.n1)
		self.conn = conn
		return self
	
	def __exit__(self, *args):
		self.conn.send(b"I'm going!")
		self.conn.close()
		self.s.close()

	def min(self, arr: list) -> (int, int):
		"""Gets the minimum of the array with respect to the latent scores as the opposite of the maximum.
		Returns the undex of the minimum ID and the minimum ID."""
		res = self.max(arr)
		if res != 'done':
			maxi, _ = res
			mini: int = maxi ^ 1
			self.learn(arr, arr[mini], False)
			return mini, arr[mini]
		else:
			return 'done'
	def max(self, arr: list, tryingAgain=False) -> (int, int):
		"""Gets the maximum of the array with respect to the latent scores.
		tryingAgain is only used for if there was a hiccup in the network.
		Returns the undex of the maximum ID and the maximum ID."""
		if not tryingAgain:
			data: bytes = self.conn.recv(10)
			if not data:
				return 'done'
			if data != b"send pics!":
				print(data, self.desHist)
				raise ConnectionError("shoulda gotten that")
			self.record(arr)
		flipped: bool = random() > 0.5
		if flipped:
			payload: bytes = b'\x02' + arr[1].to_bytes(4, 'little') + arr[0].to_bytes(4, 'little') + b'\x03'
		else:
			payload: bytes = b'\x02' + arr[0].to_bytes(4, 'little') + arr[1].to_bytes(4, 'little') + b'\x03'
		self.conn.send(payload)
		results: bytes = self.conn.recv(10)
		if len(results) == 0:
			return 'done'
		if results[0] == 2 and results[9] == 3: #valid frame
			maxInd: int = int.from_bytes(results[1:5], 'little')
			maxVal: int = int.from_bytes(results[5:9], 'little')
		elif results == b"send pics!":
			return self.max(arr, True)
		else:
			raise ConnectionError("didn't get a response " + results.decode("utf-8"))
		maxInd ^= flipped
		self.updatePC(arr, maxVal, max(arr))
		self.learn(arr, maxVal, True)
		self.desHist.append(maxVal)
		self.recorder.write(str(self.compHistory[-1])[1:-1] + f" ,{maxVal}\n")
		return maxInd, maxVal

if __name__ == "__main__":
	from sys import argv
	if len(argv != 4):
		print("Usage:")
		print(f"{__file__} <log file output> <port> <roc file output>")
	else:
		from DylData import continuousScale
		from DylSort import treeMergeSort
		from DylMath import avROC, genROC, calcNLayers
		import matplotlib.pyplot as plt
		from os import replace
		fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
		with open(argv[1], "w") as f, NetComparator('127.0.0.1', int(arv[2]), f) as comp:
			data, D0, D1 = continuousScale(comp.n0, comp.n1)
			comp.genLookup(data)
			comp.layers = layers = calcNLayers(comp.n0 + comp.n1)
			xVals: list = list(range(1, layers + 1))
			xLabels: list = ['' for _ in xVals]
			aucs: np.ndarray = np.full((layers,), np.nan)
			varEstimates: np.ndarray = np.full((layers,), np.nan)
			hmnEstimates: np.ndarray = np.full((layers, layers), np.nan)
			compLens: np.ndarray = np.full((layers,), np.nan)
			info: list = [np.nan for i in range(layers)]
			comp.aucs = aucs
			comp.pax = ax1
			comp.plt = plt
			ax1.set_ylabel("AUC")
			ax1.set_xlabel("comparisons")
			ax1.set_xticks(xVals)
			ax1.set_xticklabels(xLabels, rotation="vertical")
			ax1.set_ylim(top=1, bottom=0.4)
			ax2.plot([], [], 'b-', lw=5, label="predictions")
			ax2.plot([], [], 'r.-', label="measured")
			ax2.legend()
			ax2.set_ylabel("variance")
			ax2.set_xlabel("comparisons")
			ax2.set_xticks(xVals)
			ax2.set_xticklabels(xLabels, rotation="vertical")
			ax3.set_ylabel("${\Delta \mathrm{var^{-1}}}/{\Delta \mathrm{Comparisons}}$")
			ax3.set_xlabel("comparisons")
			ax3.set_xticklabels(xLabels, rotation="vertical")
			ax3.set_xticks(xVals)
			ax4.set_ylabel("Comparisons")
			ax4.set_xlabel("Layer")
			ax5.set_xticks(range(1, layers + 1))
			ax5.set_xlim(left=-0.01, right=1.01)
			ax5.set_ylim(bottom=-0.01, top=1.01)
			ax5.set_aspect('equal', 'box')
			fig.delaxes(ax6)
			plt.tight_layout()
			plt.savefig("temp.svg")
			replace("temp.svg", "figure.svg")
			comp.xVals = xVals
			comp.xLabels = xLabels
			print(data)
			plots = list()
			for currLayer, (groups, stats) in enumerate(treeMergeSort(data, comp, [(D0, D1)], combGroups=False)):
				print(groups)
				rocs = list()
				for group in groups:
					rocs.append(genROC(group, D0, D1))
				avgROC = avROC(rocs)
				xLabels[currLayer] = len(comp)
				auc, varEstimate, hanleyMcNeil, lowBoot, highBoot, lowSine, highSine, smVAR, npVAR, *estimates = stats
				f.write(''.join([str(val)+',' for val in stats]))
				f.write('\n')
				aucs[currLayer]: float = auc
				varEstimates[currLayer]: float = varEstimate
				hmnEstimates[currLayer]: np.ndarray = np.append(np.full((layers - len(estimates)), np.nan), estimates)
				compLens[currLayer]: int = len(comp)
				for plot in plots:
					for line in plot:
						try:
							line.remove()
						except ValueError:
							pass
				comp.currLayer += 1
				plots.append(ax1.plot(xVals, aucs, 'b.-', label="Layer AUC"))
				ax1.set_xticklabels(xLabels, rotation="vertical")
				plots.append(ax2.plot(xVals, varEstimates, 'r.-', label="measured"))
				hmnEstimates[currLayer][currLayer] = varEstimate
				ax2.plot(xVals, hmnEstimates[currLayer], 'b-', lw=5, label=f"prediction {currLayer + 1}", alpha=0.2)
				ax2.set_xticklabels(xLabels, rotation="vertical")
				if currLayer > 0:
					info[currLayer]: float = ((1 / varEstimates[currLayer]) - (1 / varEstimates[currLayer - 1])) / (compLens[currLayer] - compLens[currLayer - 1])
				else:
					plots.append(ax3.plot(xVals, xVals, lw=0))
				plots.append(ax3.plot(xVals, info, c='orange', marker='.', ls='-'))
				ax3.set_xticklabels(xLabels, rotation="vertical")
				plots.append(ax4.plot(xVals, compLens, '.-'))
				plots.append(ax5.plot(*avgROC))
				if len(groups) == 16:
					roc4: dict = avgROC
				plt.tight_layout()
				ax5.set_aspect('equal', 'box')
				plt.savefig("temp.svg")
				replace("temp.svg", "figure.svg")
		with open(argv[3], "wb") as f:
			dump((avgROC, roc4), f)