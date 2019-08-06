from sys import argv
from tkinter import *
from PIL import ImageTk, Image
import os
import socket
from numpy.random import random, randint
from time import sleep, time
class AFC:
	def __init__(self, posDir, negDir, ansDir, ip, port, n0, n1, f):
		self.decision = -1
		self.ready = True
		self.img1 = -1
		self.img2 = -1
		self.mode = 'training'
		self.posDir = posDir
		self.negDir = negDir
		self.ansDir = ansDir
		self.counter = 0
		self.imgIndex = 0
		self.ip = ip
		self.port = port
		self.n0 = int(n0)
		self.n1 = int(n1)
		self.exit = self.__exit__
		self.connected = False
		self.f = open(f, 'w')
	def connect(self):
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		print("waiting for connection")
		self.title.configure(text="Waiting for Connection...")
		root.update()
		while True:
			try:
				s.connect((self.ip, int(self.port)))
				break
			except ConnectionRefusedError:
				sleep(0.1)
		data = s.recv(10)
		if data == b"I'm ready!":
			s.send(b'\x02' + self.n0.to_bytes(4, 'little') + self.n1.to_bytes(4, 'little') + b'\x03')
		print("connection established")
		self.title.configure(text="Choose Most Likely to have Signal")
		self.s = s
		self.connected = True
	def __enter__(self):
		# get pics, use sorted() to ensure it's the same ones every time
		offset = 0
		posNames = [self.posDir + img for img in sorted(os.listdir(self.posDir))][offset:offset + self.n1]
		ansNames = [self.ansDir + img for img in sorted(os.listdir(self.ansDir))][offset:offset + self.n1]
		negNames = [self.negDir + img for img in sorted(os.listdir(self.negDir))][offset:offset + self.n0]
		self.n0 = len(negNames)
		self.n1 = len(posNames)
		print(len(posNames), len(negNames), len(ansNames))
		names = negNames + posNames
		self.images = [Image.open(name) for name in names]
		for img in self.images:
			img.thumbnail((IMGWIDTH, IMGHEIGHT))
		self.images = [ImageTk.PhotoImage(img) for img in self.images]
		for i, image in enumerate(self.images):
			setattr(image, "filename", names[i])
		ansImages = [Image.open(name) for name in ansNames]
		for img in ansImages:
			img.thumbnail((IMGWIDTH, IMGHEIGHT))
		ansImages = [ImageTk.PhotoImage(img) for img in ansImages]
		self.ansPairs = list(zip(self.images[self.n0:], ansImages))
		self.font = ('Arial', 72)
		self.correct = ImageTk.PhotoImage(Image.open("correct.png"))
		self.wrong = ImageTk.PhotoImage(Image.open("wrong.png"))
		return self
	def showPics(self, pic1, pic2=None):
		if self.mode.get() == 'answers':
			self.img1.configure(image=self.ansPairs[pic1][0])
			self.img2.configure(image=self.ansPairs[pic1][1])
		elif self.mode.get() == 'training':
			if isinstance(pic2, int):
				self.img1.configure(image=self.images[pic1])
				self.img2.configure(image=self.images[pic2])
			else: #boolean
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
		self.ready = True
		self.decision = -1
		self.imgIndex = 0
	def run(self):
		root.update()
		if self.mode.get() == 'training':
			if self.counter <= 0 and self.decision == -1: #put up new pictures
				if self.ready:
					if random() > 0.5:
						self.imgID1, self.imgID2 = randint(0, self.n0 - 1), randint(self.n0, self.n1 + self.n0 - 1)
						self.showPics(self.imgID1, self.imgID2)
						self.answer = 'right'
					else:
						self.imgID1, self.imgID2 = randint(self.n0, self.n1 + self.n0 - 1), randint(0, self.n0 - 1)
						self.showPics(self.imgID1, self.imgID2)
						self.answer = 'left'
						self.ready = False
						self.decision = -1
				else: # default place, pseudo main loop
					self.showPics(self.imgID1, self.imgID2)
			elif not self.ready and (self.decision == 113 or self.decision == 114):
				self.counter = 1000
				self.showPics((self.decision == 114 and self.answer == 'right') or (self.decision == 113 and self.answer == 'left'))
				self.decision = -1
				self.ready = True
			elif self.counter > 0:
				self.counter -= 16
				self.ready = self.counter <= 0
		elif self.mode.get() == 'study':
			if not self.connected:
				self.connect()
			if self.ready: #ready for another comparison
				try:
					#print("requesting pics")
					self.s.send(b"send pics!")
					data = self.s.recv(10)
					#print(data)
					if data:
						if data == b"I'm going!":
							print("Client exited succesfully")
							root.destroy()
						elif data[0] == 2 and data[9] == 3: #valid frame
							self.imgID1, self.imgID2 = img1, img2 = int.from_bytes(data[1:5], 'little'), int.from_bytes(data[5:9], 'little')
							self.data = data
							#print(img1, img2)
							self.showPics(img1, img2)
							self.t1 = time()
							root.update()
							self.ready = False
				except ConnectionResetError:
					print("Client Disconnected")
					self.exit()
				except BrokenPipeError:
					print("Client Disconnected")
					self.exit()
			if self.decision == 113 or self.decision == 114:
				if self.decision == 114: #right key
					payload = (1).to_bytes(4, 'little') + self.data[5:9]
					des = self.imgID2
				elif self.decision == 113: #left key
					payload = (0).to_bytes(4, 'little') + self.data[1:5]
					des = self.imgID2
				try:
					self.f.write(f"{self.images[self.imgID1].filename} {self.images[self.imgID2].filename} {self.images[des].filename} {time() - self.t1}\n")
					self.s.send(b'\x02' + payload + b'\x03')
				except ConnectionResetError:
					print("Client Disconnected")
					self.exit()
				except BrokenPipeError:
					print("Client Disconnected")
					self.exit()
				self.decision = -1
				self.ready = True
		elif self.mode.get() == "answers":
			if self.ready: #put up new pictures
				self.showPics(self.imgIndex)
				self.ready = False
				root.update()
			elif self.decision == 114:
				#print("saw the event")
				self.imgIndex += 1 if self.imgIndex < self.n1 - 1 else 0
				self.decision = -1
				self.ready = True
			elif self.decision == 113:
				self.imgIndex -= 1if self.imgIndex > 0 else 0
				self.decision = -1
				self.ready = True
		root.after(16, self.run)
	def pressed(self, event):
		#print("event!")
		self.counter = 0
		if event.keycode == 113 or event.keycode == 114:
			self.decision = event.keycode
	def clicked(self, event):
		self.counter = 0
		if event.y > 100:
			if event.x > WIDTH - IMGWIDTH:
				self.decision = 114
			elif event.x < IMGWIDTH:
				self.decision = 113
	def __exit__(self, *args):
		self.f.close()
		if self.connected:
			self.s.close()
		try:
			root.destroy()
		except Exception: #root already destroyed
			self.counter = 0 # to sate the linters
HEIGHT = 700
WIDTH = 1300
IMGWIDTH = 600
IMGHEIGHT = 600
root = Tk()
#canvas = Canvas(root, width=WIDTH, height=HEIGHT)
#canvas.pack()
#img = Image.open("repository-pic.png")
#img.thumbnail((IMGWIDTH, IMGHEIGHT))
#img = ImageTk.PhotoImage(img)
label = Label(root, text="Choose Most Likely to have Signal", font=('Arial', 56))
label.grid(row=0, column=0)
frame = Frame(root)
img1 = Label(frame)#, image=img)
img2 = Label(frame)#, image=img)
img1.grid(row=0, column=0, sticky=E)
img2.grid(row=0, column=2, sticky=W)
if len(argv) == 1:
	argv.append('/nashome/images/testImages1/targetPresentImages/')
	argv.append('/nashome/images/testImages1/targetAbsentImages/')
	argv.append('/nashome/images/testImages1/targetPresentAnswerImages/')
	argv.append('127.0.0.1')
	argv.append('6000')
	argv.append('64')
	argv.append('64')
	argv.append('log.csv')
elif len(argv) < 9:
	print(f"Usage: {__file__} [target present directory] [target absent directory] [answers directory] [merge ip] [merger port] [n0] [n1] [log file]")
	exit(-1)
modes = [
	("Answers", "answers"),
	("AFC Training", "training"),
	("Study", "study")
]
frame.grid_columnconfigure(1, weight=1)
with AFC(*argv[1:]) as afc:
	afc.title = label
	afc.mode = StringVar()
	afc.mode.set("none")
	afc.img1 = img1
	afc.img2 = img2
	buttons = Frame(frame)
	for i, (text, mode) in enumerate(modes):
		b = Radiobutton(buttons, text=text, value=mode, variable=afc.mode, indicatoron=0, command=afc.switchModes)
		b.grid(row=i, column=0)
	buttons.grid(row=0, column=1)
	root.bind("<Key>", afc.pressed)
	root.bind("<Button-1>", afc.clicked)
	root.protocol("WM_DELETE_WINDOW", afc.exit)
	root.after(1, afc.run)
	frame.grid(row=1, column=0)
	mainloop()
