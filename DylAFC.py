#!/usr/bin/python3.6
import os
import socket
import time
import numpy as np
from sys import argv
from tkinter import *
from PIL import ImageTk, Image
class AFC:
	"""A Class for doing AFC studies. This one only does 2AFC. It is used for integrating with a tkinter interface as seen in DylAFC."""
	def __init__(self, posDir: str, negDir: str, ansDir: str, ip: str, port: str, n0: int, n1: int, logFile: str):
		"""Creates a new AFC class with the given parameters. Opens the file name provided for logging. Does not connect yet."""
		self.decision: int = -1
		self.ready: bool = True
		self.img1: int = -1
		self.img2: int = -1
		self.mode: str = 'training'
		self.posDir: str = posDir
		self.negDir: str = negDir
		self.ansDir: str = ansDir
		self.counter: int = 0
		self.imgIndex: int = 0
		self.ip: str = ip
		self.port: str = port
		self.n0: int = int(n0)
		self.n1: int = int(n1)
		self.exit = self.__exit__
		self.connected: bool = False
		self.f = open(logFile, 'w')
	def connect(self):
		"""Connects to the comparator. Waits until connection is established, therefore this method is blocking."""
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		print("waiting for connection")
		self.title.configure(text="Waiting for Connection...")
		root.update()
		while True:
			try:
				s.connect((self.ip, int(self.port)))
				break
			except ConnectionRefusedError:
				time.sleep(0.1)
		data: bytes = s.recv(10)
		if data == b"I'm ready!":
			s.send(b'\x02' + self.n0.to_bytes(4, 'little') + self.n1.to_bytes(4, 'little') + b'\x03')
		print("connection established")
		self.title.configure(text="Choose Most Likely to have Signal")
		self.s = s
		self.connected: bool = True
	def __enter__(self):
		# get pics, use sorted() to ensure it's the same ones every time
		offset: int = 0
		posNames: list = [self.posDir + img for img in sorted(os.listdir(self.posDir))][offset:offset + self.n1]
		ansNames: list = [self.ansDir + img for img in sorted(os.listdir(self.ansDir))][offset:offset + self.n1]
		negNames: list = [self.negDir + img for img in sorted(os.listdir(self.negDir))][offset:offset + self.n0]
		self.n0: int = len(negNames)
		self.n1: int = len(posNames)
		#print(len(posNames), len(negNames), len(ansNames))
		names: list = negNames + posNames
		with open("names.txt", "w") as f:
			for name in names:
				f.write(name + ' ')
		self.images: list = [Image.open(name) for name in names]
		for img in self.images:
			img.thumbnail((IMGWIDTH, IMGHEIGHT))
		self.images: list = [ImageTk.PhotoImage(img) for img in self.images]
		for i, image in enumerate(self.images):
			setattr(image, "filename", names[i])
		ansImages: list = [Image.open(name) for name in ansNames]
		for img in ansImages:
			img.thumbnail((IMGWIDTH, IMGHEIGHT))
		ansImages: list = [ImageTk.PhotoImage(img) for img in ansImages]
		self.ansPairs: list = list(zip(self.images[self.n0:], ansImages))
		self.font: tuple = ('Arial', 72)
		self.correct = ImageTk.PhotoImage(Image.open("correct.png"))
		self.wrong = ImageTk.PhotoImage(Image.open("wrong.png"))
		return self
	def showPics(self, pic1: int, pic2: int=None):
		""" show the pictures identified by the image number arguments.
		If pic2 is not provided, pic1 is a boolean value for displaying either correct or incorrect."""
		if self.mode.get() == 'answers':
			self.img1.configure(image=self.ansPairs[pic1][0])
			self.img2.configure(image=self.ansPairs[pic1][1])
		elif self.mode.get() == 'training' and not isinstance(pic2, int):
			# pic1 is a boolean
				if pic1: #correct
					self.img1.configure(image=self.correct)
					self.img2.configure(image=self.correct)
				else:
					self.img1.configure(image=self.wrong)
					self.img2.configure(image=self.wrong)
		else:
			self.img1.configure(image=self.images[pic1])
			self.img2.configure(image=self.images[pic2])
		root.update()
	def switchModes(self):
		"""Call this when switching modes to make sure the class is ready to start a mode"""
		self.ready: bool = True
		self.decision: int = -1
		self.imgIndex: int = 0
	def run(self):
		"""Finite state machine based on the mode, the connection status, and whether the reader has decided an image or not.
		This will call itself on its own, so just make sure to call this method once with <root>.after()"""
		root.update()
		if self.mode.get() == 'training':
			if self.counter <= 0 and self.decision == -1: #put up new pictures
				if self.ready:
					if np.random.random() > 0.5:
						self.imgID1: int = np.random.randint(0, self.n0 - 1)
						self.imgID2: int = np.random.randint(self.n0, self.n1 + self.n0 - 1)
						self.showPics(self.imgID1, self.imgID2)
						self.answer: str = 'right'
					else:
						self.imgID1: int = np.random.randint(self.n0, self.n1 + self.n0 - 1)
						self.imgID2: int =  np.random.randint(0, self.n0 - 1)
						self.showPics(self.imgID1, self.imgID2)
						self.answer: str = 'left'
					self.ready: bool = False
					self.decision: int = -1
				else: # default place, pseudo main loop
					self.showPics(self.imgID1, self.imgID2)
			elif not self.ready and (self.decision == 113 or self.decision == 114):
				self.counter: int = 1000
				self.showPics((self.decision == 114 and self.answer == 'right') or (self.decision == 113 and self.answer == 'left'))
				self.decision: int = -1
				self.ready: bool = True
			elif self.counter > 0:
				self.counter -= 16
				self.ready: bool = self.counter <= 0
		elif self.mode.get() == 'study':
			if not self.connected:
				self.connect()
			if self.ready: #ready for another comparison
				try:
					#print("requesting pics")
					self.s.send(b"send pics!")
					data: bytes = self.s.recv(10)
					#print(data)
					if data:
						if data == b"I'm going!":
							print("Client exited succesfully")
							root.destroy()
						elif data[0] == 2 and data[9] == 3: #valid frame
							self.imgID1: int = int.from_bytes(data[1:5], 'little')
							self.imgID2: int = int.from_bytes(data[5:9], 'little')
							self.data: bytes = data
							#print(img1, img2)
							self.showPics(self.imgID1, self.imgID2)
							self.t1: float = time.time()
							root.update()
							self.ready: bool = False
				except ConnectionResetError:
					print("Client Disconnected")
					self.exit()
				except BrokenPipeError:
					print("Client Disconnected")
					self.exit()
			if self.decision == 113 or self.decision == 114:
				if self.decision == 114: #right key
					payload: bytes = (1).to_bytes(4, 'little') + self.data[5:9]
					des: int = self.imgID2
				elif self.decision == 113: #left key
					payload: bytes = (0).to_bytes(4, 'little') + self.data[1:5]
					des: int = self.imgID2
				try:
					self.f.write(f"{self.images[self.imgID1].filename} {self.images[self.imgID2].filename} {self.images[des].filename} {time.time() - self.t1}\n")
					self.s.send(b'\x02' + payload + b'\x03')
				except ConnectionResetError:
					print("Client Disconnected")
					self.exit()
				except BrokenPipeError:
					print("Client Disconnected")
					self.exit()
				self.decision: int = -1
				self.ready: bool = True
		elif self.mode.get() == "answers":
			if self.ready: #put up new pictures
				self.showPics(self.imgIndex)
				self.ready: bool = False
				root.update()
			elif self.decision == 114:
				#print("saw the event")
				self.imgIndex += 1 if self.imgIndex < self.n1 - 1 else 0
				self.decision: int = -1
				self.ready: bool = True
			elif self.decision == 113:
				self.imgIndex -= 1if self.imgIndex > 0 else 0
				self.decision: int = -1
				self.ready: bool = True
		root.after(16, self.run)
	def pressed(self, event: Event):
		"""Call this when the user has pressed a keyboard button"""
		#print("event!")
		self.counter: int = 0
		if event.keycode == 113 or event.keycode == 114:
			self.decision: int = event.keycode
	def clicked(self, event: Event):
		"""Call this when the user has clicked the mouse"""
		self.counter: int = 0
		if event.y > 100:
			if event.x > WIDTH - IMGWIDTH:
				self.decision: int = 114
			elif event.x < IMGWIDTH:
				self.decision: int = 113
	def __exit__(self, *args):
		self.f.close()
		if self.connected:
			self.s.close()
		try:
			root.destroy()
		except Exception: #root already destroyed
			self.counter: int = 0 # to sate the linters
if __name__ == "__main__":
	if len(argv) < 9:	
		print(f"Usage: {__file__} [target present directory] [target absent directory] [answers directory] [merge ip] [merger port] [n0] [n1] [log file]")
	else:
		HEIGHT: int = 700
		WIDTH: int = 1300
		IMGWIDTH: int = 600
		IMGHEIGHT: int = 600
		root = Tk()
		#canvas = Canvas(root, width=WIDTH, height=HEIGHT)
		#canvas.pack()
		#img = Image.open("repository-pic.png")
		#img.thumbnail((IMGWIDTH, IMGHEIGHT))
		#img = ImageTk.PhotoImage(img)
		label: Label = Label(root, text="Choose Most Likely to have Signal", font=('Arial', 56))
		label.grid(row=0, column=0)
		frame: Frame = Frame(root)
		img1: Label = Label(frame)#, image=img)
		img2: Label = Label(frame)#, image=img)
		img1.grid(row=0, column=0, sticky=E)
		img2.grid(row=0, column=2, sticky=W)
		modes: list = [
			("Answers", "answers"),
			("AFC Training", "training"),
			("Study", "study")
		]
		frame.grid_columnconfigure(1, weight=1)
		with AFC(*argv[1:]) as afc:
			afc.title: Label = label
			afc.mode = StringVar()
			afc.mode.set("none")
			afc.img1: Label = img1
			afc.img2: Label = img2
			buttons: Frame = Frame(frame)
			for i, (text, mode) in enumerate(modes):
				b: Radiobutton = Radiobutton(buttons, text=text, value=mode, variable=afc.mode, indicatoron=0, command=afc.switchModes)
				b.grid(row=i, column=0)
			buttons.grid(row=0, column=1)
			root.bind("<Key>", afc.pressed)
			root.bind("<Button-1>", afc.clicked)
			root.protocol("WM_DELETE_WINDOW", afc.exit)
			root.after(1, afc.run)
			frame.grid(row=1, column=0)
			mainloop()
