#!/usr/bin/env python

import numpy as np
import idx2numpy as i2np
from matplotlib import pyplot as plt

from pybrain.tools.xml.networkreader import NetworkReader
from argparse import ArgumentParser

parser = ArgumentParser()

# Add more options if you like
parser.add_argument("-n", metavar="NET", type=str, dest="filename",
                    default="./networks/good.xml",
                    help="trained network")
parser.add_argument("-di", metavar="IMG DATA", type=str, dest="imgData",
                    default="./testimg/t10k-images-idx3-ubyte",
                    help="image data to test")
parser.add_argument("-dl", metavar="LABEL DATA", type=str, dest="labelData",
                    default="./testimg/t10k-labels-idx1-ubyte",
                    help="labels for data to test")
parser.add_argument("-v", metavar="VISUAL", type=bool, dest="visual",
                    default=False,
                    help="display wrong numbers")

args = parser.parse_args()

print "Loading Image data..."
imgs = i2np.convert_from_file(args.imgData)
print ""
print "Loading labels.."
labels = i2np.convert_from_file(args.labelData)
print ""
print "Loading Network..."
net = NetworkReader.readFrom(args.filename)
print "Ready!"

count = 0

for i in xrange(len(imgs)):
  probs = net.activate(np.ravel(imgs[i]))
  guess = np.argmax(probs)
  if guess != labels[i]:
    count += 1
    print "Network guess: %d" % guess, "Actual value: %d" % labels[i]
    if args.visual:
      print probs
      plt.imshow(imgs[i])
      plt.show()

print ""
print count, "of %d" % len(imgs), "wrong = ", str(count / float(len(imgs)) * 100), "%"
