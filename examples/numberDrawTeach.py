import Tkinter as tk
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw ,ImageFilter, ImageOps


from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader


from argparse import ArgumentParser




parser = ArgumentParser()

# Add more options if you like
parser.add_argument("-n", metavar="NET", type=str, dest="network",
                    default="newNetwork.xml",
                    help="pre trained network")
parser.add_argument("-c", metavar="CLASSES", type=int, dest="classes",
                    default=10,
                    help="number of output classes")
parser.add_argument("-if", metavar="INPUT FEATURES", type=int, dest="inpFeatures",
                    default=28,
                    help="defaut resolution")
parser.add_argument("-H", metavar="H", type=int, dest="hidden_neurons",
                    default=200,
                    help="number of neurons in the hidden layer")
parser.add_argument("-e", metavar="EPOCHS", type=int,
                    dest="epochs", default=100,
                    help="number of epochs to learn")
parser.add_argument("-d", metavar="W", type=float, dest="weightdecay",
                    default=0.01,
                    help="weightdecay")
parser.add_argument("-m", metavar="M", type=float, dest="momentum",
                    default=0.1,
                    help="momentum")
parser.add_argument("-l", metavar="ETA", type=float, dest="learning_rate",
                    default=0.001,
                    help="learning rate")
parser.add_argument("-ld", metavar="ALPHA", type=float, dest="lrdecay",
                    default=1,
                    help="learning rate decay")
parser.add_argument("-sp", metavar="SPLIT", type=float, dest="split",
                    default=0.7,
                    help="percentage used for training")

args = parser.parse_args()


def processImage(img):
    return img.filter(ImageFilter.BLUR).resize((args.inpFeatures,args.inpFeatures)).convert('L')


class popupWindow(object):
    def __init__(self,master):
        top=self.top=tk.Toplevel(master)
        self.l=tk.Label(top,text="Sample Value")
        self.l.pack()
        self.e=tk.Entry(top)
        self.e.pack()
        self.b=tk.Button(top,text='Ok',command=self.cleanup)
        self.b.pack()
    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()

class ImageGenerator(object):
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
        self.button=tk.Button(self.parent,text="Add",width=10,bg='white',command=self.add)
        self.button.place(x=10,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Train",width=10,bg='white',command=self.train)
        self.button1.place(x=130,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Save",width=10,bg='white',command=self.saveNetwork)
        self.button1.place(x=10 ,y=self.sizey+50)

        self.image=Image.new("RGB",(self.sizex,self.sizey),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)

        INPUT_FEATURES = args.inpFeatures * args.inpFeatures
        print("Input features: %i" % INPUT_FEATURES)
        self.alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=args.classes)
        print "in",self.alldata.indim
        print "out",self.alldata.outdim
        self.net = buildNetwork(self.alldata.indim, args.hidden_neurons, args.classes, outclass=SoftmaxLayer)

    def add(self):
        im = processImage(self.image)
        data = np.array(ImageOps.invert(im))
        #open new window
        w=popupWindow(self.parent)
        self.parent.wait_window(w.top)
        value = 0
        try:
            value = int(w.value)
            if value >= args.classes:
                print "Input has to be smaller then the number of classes (%d)" % args.classes
                return
        except ValueError:
            print "Input has to be a number"
            return
        self.alldata.addSample(np.ravel(data), [value])
        print "Number of samples %d" % len(self.alldata)
        self.clear()

    def train(self):
        trndata, tstdata = self.alldata.splitWithProportion(args.split)
        print "Train dataset: %4d" % len(trndata)
        print "Test dataset: %4d" % len(tstdata)

        trndata._convertToOneOfMany()
        tstdata._convertToOneOfMany()
        print "In: ", trndata.indim
        print "Out: ", trndata.outdim
        trainer = BackpropTrainer(self.net, dataset=trndata, momentum=args.momentum,
                                  verbose=True, weightdecay=args.weightdecay,
                                  learningrate=args.learning_rate,
                                  lrdecay=args.lrdecay)
        for i in xrange(args.epochs):
            trainer.trainEpochs(1)
            trnresult = percentError(trainer.testOnClassData(),
                                     trndata['class'])
            tstresult = percentError(trainer.testOnClassData(
                                     dataset=tstdata), tstdata['class'])
            print "Epoch %4d" % trainer.totalepochs + ",  Train error: %5.2f%%" % trnresult + ",  Test error: %5.2f%%" % tstresult
        print "Done!"

    def saveNetwork(self):
        NetworkWriter.writeToFile(self.net, args.network)
        print "Saved!"

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
