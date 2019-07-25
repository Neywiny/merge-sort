from tkinter import *
from PIL import ImageTk, Image
import os
import socket
from random import random, randint


class AFC:
    def __init__(self, posDir, negDir):
        self.decision = -1
        self.ready = True
        self.img1 = -1
        self.img2 = -1
        self.mode = 'training'
        self.posDir = posDir
        self.negDir = negDir
        self.counter = 0   
        self.exit = self.__exit__
    def __enter__(self):
        # get pics
        posNames = [self.posDir + img for img in os.listdir(self.posDir)]
        negNames = [self.negDir + img for img in os.listdir(self.negDir)]
        self.n0 = len(negNames)
        self.n1 = len(posNames)
        print(len(posNames), len(negNames))
        self.images = [Image.open(name) for name in posNames + negNames]
        for img in self.images:
            img.thumbnail((IMGWIDTH, IMGHEIGHT))
        self.images = [ImageTk.PhotoImage(img) for img in self.images]

        TCP_IP = '127.0.0.1'
        TCP_PORT = 6000
        BUFFER_SIZE = 10  # Normally 1024, but we want fast response
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1) # also attempt to pull the most recent connection
        print("waiting for connection")
        conn, addr = s.accept()
        print('Connection address:', addr)
        established = False
        data = conn.recv(10)
        if data == b"I'm ready!":
            conn.send(b'\x02' + self.n0.to_bytes(4, 'little') + self.n1.to_bytes(4, 'little') + b'\x03')
        self.conn = conn
        self.s = s
        self.font = ('Arial', 72)

        return self
    def showPics(self, pic1, pic2):
        canvas.delete('image')
        canvas.create_image(0, HEIGHT, anchor=SW, image=self.images[pic1], tags=('image'))
        canvas.create_image(WIDTH, HEIGHT, anchor=SE, image=self.images[pic2], tags=('image'))
    def run(self):
        if self.mode == 'training':
            if self.ready and self.counter <= 0: #put up new pictures
                if random() > 0.5:
                    self.showPics(randint(0, self.n1 - 1), randint(self.n1, self.n1 + self.n0 - 1))
                    self.answer = 'left'
                else:
                    self.showPics(randint(self.n1, self.n1 + self.n0 - 1), randint(0, self.n1 - 1))
                    self.answer = 'right'
                self.ready = False
                root.update()
            elif self.decision == 113 or self.decision == 114:
                    if (self.decision == 114 and self.answer == 'right') or (self.decision == 113 and self.answer == 'left'): #correct
                        canvas.create_text(WIDTH/2, HEIGHT/2, fill="green2", text="CORRECT", font=self.font, tags=('image'))
                    else: #incorrect
                        canvas.create_text(WIDTH/2, HEIGHT/2, fill="red", text="WRONG", font=self.font, tags=('image'))
                    self.counter = 1000
                    self.decision = -1
                    self.ready = True
            elif self.counter > 0:
                self.counter -= 16
        else:
            if self.ready: #ready for another comparison
                try:
                    #print("requesting pics")
                    self.conn.send(b"send pics!")
                    data = self.conn.recv(10)
                    #print(data)
                    if data:
                        if data == b"I'm going!":
                            print("Client exited succesfully")
                            root.destroy()
                        elif data[0] == 2 and data[9] == 3: #valid frame
                            img1, img2 = int.from_bytes(data[1:5], 'little'), int.from_bytes(data[5:9], 'little')
                            self.data = data
                            #print(img1, img2)
                            self.showPics(img1, img2)
                            #canvas.delete('image')
                            #canvas.create_text(WIDTH / 4, HEIGHT / 2, text=str(img1), font=self.font, tags=('image', 'text'))
                            #canvas.create_text(3* WIDTH / 4, HEIGHT / 2, text=str(img2), font=self.font, tags=('image', 'text'))
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
                elif self.decision == 113: #left key
                    payload = (0).to_bytes(4, 'little') + self.data[1:5]
                try:
                    self.conn.send(b'\x02' + payload + b'\x03')
                except ConnectionResetError:
                    print("Client Disconnected")
                    self.exit()
                except BrokenPipeError:
                    print("Client Disconnected")
                    self.exit()
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
            if event.x > IMGWIDTH and event.x < IMGWIDTH + 100 and event.y > HEIGHT/2 - 50 and event.y < HEIGHT/2 + 50:
                canvas.delete('button')
                self.mode = 'actual'
                self.ready = True
    def __exit__(self, *args):
        self.conn.close()
        self.s.close()
        try:
            root.destroy()
        except: #root already destroyed
            pass


HEIGHT = 700
WIDTH = 1300
IMGWIDTH = 600
IMGHEIGHT = 600
root = Tk()
canvas = Canvas(root, width=WIDTH, height=HEIGHT)
canvas.pack()
img = Image.open("repository-pic.png")
img.thumbnail((IMGWIDTH, IMGHEIGHT))
img = ImageTk.PhotoImage(img)
canvas.create_text(WIDTH / 2, 50, text="Choose Most Likely to have Signal", font=('Arial', 56), tags=('text'))
#canvas.create_image(0, HEIGHT, anchor=SW, image=img)
#canvas.create_image(WIDTH, HEIGHT, anchor=SE, image=img)
with AFC('/nashome/images/targetPresentImages/', '/nashome/images/targetAbsentImages/') as afc:
    if afc.mode == 'training':
        canvas.create_rectangle(WIDTH / 2 - 50, HEIGHT / 2 - 50, WIDTH / 2 + 49, HEIGHT / 2 + 50, tags=('button'))
        canvas.create_text(WIDTH / 2, HEIGHT / 2, text="click when ready to proceed", width=100, tags=('button'))
    root.bind("<Key>", afc.pressed)
    root.bind("<Button-1>", afc.clicked)
    root.protocol("WM_DELETE_WINDOW", afc.__exit__)
    root.after(1, afc.run)
    mainloop()