import numpy as np
np.seterr(all="ignore")
def continuousScale(*args):
	"""creates a set of continuous scale data.
	If provided a filename, returns the indecies for sorting, D0, and D1
	If provided a number, just creates and returns the indecies alternating from D0 and D1 indecies
	If provided 2 numbers, returns D0 and D1 interleaved evenly into arr, D0, and D1"""
	if len(args) == 1:
		if isinstance(args[0], str):
			filename = args[0]
			data = []
			D0 = []
			D1 = []
			with open(filename) as f:
				for i, line in enumerate(f):
					if len(line) > 10:
						line = line.strip().split(" ")
						point = float(line[2])
						data.append(point)
					if line[1] == "1": # case should be positive
						D1.append(i)
					else: # case should be negative
						D0.append(i)
			newData = [-1 for i in range(len(data))]
			#print(data)
			for i, d in enumerate(sorted(data)):
				newData[i] = data.index(d)
				D0.sort()
				D1.sort()
			return newData, D0, D1
		elif isinstance(args[0], int):
			data = list()
			for i in range(0, args[0] // 2):
				data.append(i + args[0] // 2)
				data.append(i)
			return data
	elif len(args) == 2:
		D0 = list(range(args[0]))
		D1 = list(range(args[0], args[0] + args[1]))
		arr = list()
		negI = 0
		posI = 0
		ratio = args[1] / args[0]
		while negI < len(D0):
			arr.append(D0[negI])
			negI += 1
			percent = negI * ratio
			while posI < percent:
				arr.append(D1[posI])
				posI += 1
		return arr, D0, D1
if __name__ == "__main__":
	data = continuousScale(5, 4)
	print(*data)