#! /usr/bin/python3
import os
from numpy import random
from sys import argv
from tkinter.constants import *
from tkinter import *
from PIL import ImageTk, Image, ImageDraw
from time import time
class Rating:
	"""A class for displaying an image and recording the ratins and timings of the reader.
	Label parameter is the label where the image is going to be drawn.
	User must assign the instantiated rating class a canvas for drawing a bar where the reader clicked."""
	def __init__(self, posDir: str, negDir: str, n: int, outputFile: str, offset: str, label: Label):
		self.decision: int = -1
		self.posDir: str = posDir
		self.negDir: str = negDir
		self.counter: int = 0
		self.percent: int = -1
		self.offset: int = int(offset)
		self.n: int = int(n)
		self.f = open(outputFile + str(time()) + ".csv", "w")
		self.f.write(posDir + ' ' + negDir + '\n')
		# get pics
		posNames: list = [self.posDir + img for img in sorted(os.listdir(self.posDir))][self.offset:self.offset + (self.n // 2)]
		negNames: list = [self.negDir + img for img in sorted(os.listdir(self.negDir))][self.offset:self.offset + (self.n // 2 + self.n % 2)]
		self.n0: int = len(negNames)
		self.n1: int = len(posNames)
		names: list = negNames + posNames
		#print(len(names))
		self.images: list = [Image.open(name) for name in names]
		for img in self.images:
			img.thumbnail((IMGWIDTH, IMGHEIGHT))
		self.images: list = [ImageTk.PhotoImage(img) for img in self.images]
		for i, image in enumerate(self.images):
			setattr(image, "filename", names[i])
		random.shuffle(self.images)
		self.label = label
		print()

	def next(self, event: Event=None):
		if self.percent != -1:
			if hasattr(self, "frame"):
				self.canvas.delete("bar")
				self.f.write(f"{self.images[self.counter].filename} {self.percent} {time() - self.t1}\n")
				self.t1: float = time()
				self.percent: int = -1
			if self.counter < self.n - 1:
				self.counter += 1
				print(f"\r{int(100 * self.counter / self.n)}%", end='')
				self.label.configure(image=self.images[self.counter])
				self.label.image = self.images[self.counter]
			elif hasattr(self, "canvas"):
				self.f.close()
				self.canvas.master.destroy()
				print()

	def drawBar(self, event: Event):
		if not hasattr(self, "canvas"):
			raise Exception("Need to set Rating.canvas")
		self.percent: int = round(((IMGHEIGHT - event.y) / IMGHEIGHT) * 100)
		self.canvas.delete("bar")
		self.canvas.create_rectangle(0, event.y, 100, event.y + 2, fill="blue", tags=("bar"))

def hex2rgb(str_rgb):
	"""A function that takes hex values and converts them to rgb"""
	#https://github.com/vaab/colour/blob/master/colour.py
	try:
		rgb = str_rgb[1:]
		if len(rgb) == 6:
			r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
		elif len(rgb) == 3:
			r, g, b = rgb[0] * 2, rgb[1] * 2, rgb[2] * 2
		else:
			raise ValueError()
	except:
		raise ValueError("Invalid value %r provided for rgb color."% str_rgb)
	return tuple(int(v, 16) for v in (r, g, b))
class GradientFrame(Canvas):
	"""A canvas with a gradient drawn on it."""
	#https://stackoverflow.com/questions/11892521/tkinter-custom-window
	def __init__(self, master, from_color, to_color, width=None, height=None, orient=HORIZONTAL, steps=None, **kwargs):
		Canvas.__init__(self, master, **kwargs)
		if steps is None:
			if orient == HORIZONTAL:
				steps = height
			else:
				steps = width
		if isinstance(from_color, str):
			from_color = hex2rgb(from_color)
		if isinstance(to_color, str):
			to_color = hex2rgb(to_color)
		r,g,b = from_color
		dr = float(to_color[0] - r)/steps
		dg = float(to_color[1] - g)/steps
		db = float(to_color[2] - b)/steps
		if orient == HORIZONTAL:
			if height is None:
				raise ValueError("height can not be None")
			self.configure(height=height)
			if width is not None:
				self.configure(width=width)
			img_height = height
			img_width = self.winfo_screenwidth()
			image = Image.new("RGB", (img_width, img_height), "#FFFFFF")
			draw = ImageDraw.Draw(image)
			for i in range(steps):
				r,g,b = r+dr, g+dg, b+db
				y0 = int(float(img_height * i)/steps)
				y1 = int(float(img_height * (i+1))/steps)
				draw.rectangle((0, y0, img_width, y1), fill=(int(r),int(g),int(b)))
		else:
			if width is None:
				raise ValueError("width can not be None")
			self.configure(width=width)
			if height is not None:
				self.configure(height=height)
			img_height = self.winfo_screenheight()
			img_width = width
			image = Image.new("RGB", (img_width, img_height), "#FFFFFF")
			draw = ImageDraw.Draw(image)
			for i in range(steps):
				r,g,b = r+dr, g+dg, b+db
				x0 = int(float(img_width * i)/steps)
				x1 = int(float(img_width * (i+1))/steps)
				draw.rectangle((x0, 0, x1, img_height), fill=(int(r),int(g),int(b)))
		self._gradient_photoimage = ImageTk.PhotoImage(image)
		self.create_image(0, 0, anchor=NW, image=self._gradient_photoimage)

if __name__ == "__main__":
	if len(argv) == 2:
		import ROC1
		import matplotlib.pyplot as plt
		import numpy as np
		x0 = list()
		x1 = list()
		times = list()
		with open(argv[1]) as f:
			posDir, negDir = f.readline().strip().split()
			for line in f:
				line = line.strip().split()
				score = int(line[1])
				if negDir in line[0]:
					x0.append(score)
				else:
					x1.append(score)
				time = float(line[2])
				if time < 30:
					times.append(time)
		x1, x0 = np.array(x1), np.transpose(x0)
		print(len(x1), len(x0))
		roc = ROC1.rocxy(x1, x0)
		sm = ROC1.successmatrix(x1, x0)
		AUC = np.mean(sm)
		VAR = ROC1.unbiasedMeanMatrixVar(sm, 1)
		fig, (ax1, ax2) = plt.subplots(ncols=2)
		ax1.set_xlim(left=-0.01, right=1.01)
		ax1.set_ylim(top=1.01, bottom=-0.01)
		ax1.plot(roc['x'], roc['y'])
		ax1.set_title(f"N:{len(x1) + len(x0)} AUC:{AUC} VAR:{VAR:0.05f}")
		ax2.plot(times)
		ax2.set_title(f"Avg time {np.mean(times):0.2f}s")
		plt.show()
	elif len(argv) != 5 and len(argv) != 6:
		print(f"Usage: \n\tpython3 {__file__} [signal present directory] [signal absent directory] [n] [output file] [offset (optional)]\n\tpython3 {__file__} [results file]")
	else:
		IMGWIDTH: int = 600
		IMGHEIGHT: int = 600
		root = Tk()
		#root.geometry("800x800")
		title = Label(root, text="Choose the percent chance of there being a signal")
		title.grid(row=0, column=0)
		label = Label(root)
		if len(argv) < 6:
			argv.append("0")
		try:
			rating = Rating(*argv[1:], label)
			label.configure(image=rating.images[0])
			label.grid(row=1, column=0)
			ticks = Frame(root)
			ticks.grid(row=1, column=1)
			for tickmark in range(0, 110, 10):
				text = Label(ticks, text=str(tickmark))
				row: int = 10-(tickmark//10)
				text.grid(row=row, column=0, sticky=E, pady=18)
			gradient = GradientFrame(root, from_color="#000000", to_color="#FFFFFF", height=IMGHEIGHT, width=100, orient=HORIZONTAL)
			gradient.grid(row=1, column=2)
			rating.canvas = gradient
			gradient.bind("<Button-1>", rating.drawBar)
			button = Button(text="next", command=rating.next)
			button.grid(row=2, column=2)
			root.bind("<Return>", rating.next)
			rating.t1 = time()
			root.mainloop()

		except ValueError:
			print("Invalid Arguments")
			root.destroy()