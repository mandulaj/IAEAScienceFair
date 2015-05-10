#!/usr/bin/env python

import Tkinter as tk
import numpy as np
from PIL import Image, ImageDraw ,ImageFilter, ImageOps

from pybrain.tools.xml.networkreader import NetworkReader
from matplotlib import pyplot as plt
from argparse import ArgumentParser

import imgtools

def processImage(img):
    return img.filter(ImageFilter.BLUR).resize((28,28)).convert('L')


parser = ArgumentParser()

# Add more options if you like
parser.add_argument("-n", metavar="NET", type=str, dest="network",
                    default="./networks/good.xml",
                    help="pre trained network")
parser.add_argument("-i", metavar="IMG", type=str, dest="img",
                    default="img.png",
                    help="test image")

args = parser.parse_args()


print  "Loading Network..."
net = NetworkReader.readFrom(args.network)
print "Done"
print ""
print "Starting GUI..."


class ImageGenerator:
    def __init__(self,parent,posx,posy,*kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 230
        self.sizey = 230
        self.b1 = "up"
        self.xold = None
        self.yold = None
        self.drawing_area=tk.Canvas(self.parent,width=self.sizex,height=self.sizey)
        self.drawing_area.place(x=self.posx,y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button=tk.Button(self.parent,text="Done!",width=10,bg='white',command=self.detect)
        self.button.place(x=10,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Clear!",width=10,bg='white',command=self.clear)
        self.button1.place(x=130,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Save!",width=10,bg='white',command=self.saveImg)
        self.button1.place(x=10 ,y=self.sizey+50)

        self.image=Image.new("RGB",(self.sizex,self.sizey),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)

    def detect(self):
        im = processImage(self.image)
        data = np.array(ImageOps.invert(im))
        probs = net.activate(data.flatten())
        num = np.argmax(probs)
        print "Guessed: %d" % num + " with certainty %.5f" % probs[num]
        #plt.imshow(data)
        #plt.show()
        imgtools.imgCenter(data)

    def saveImg(self):
        processImage(self.image).save(args.img)

    def clear(self):
        self.drawing_area.delete("all")
        self.image=Image.new("RGB",(self.sizex,self.sizey),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self,event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold,self.yold,event.x,event.y,smooth='true',width=5,fill='black')
                self.draw.line(((self.xold,self.yold),(event.x,event.y)),(0,0,0),width=17)

        self.xold = event.x
        self.yold = event.y

if __name__ == "__main__":
    root=tk.Tk()
    root.configure(background='black')
    root.wm_geometry("%dx%d+%d+%d" % (250, 330, 10, 10))
    root.config(bg='white')
    ImageGenerator(root,10,10)
    root.mainloop()
