#!/usr/bin/python
# (C) JACK CONCANON


import idx2numpy as idx2np
import numpy as np
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import cv2

def loadImage(path):
    path = "testimg/"+path
    im = cv2.imread(path)
    return flatten(im)


def loadImages(pathImg, pathName):
  npImg = idx2np.convert_from_file(pathImg)
  npNames = idx2np.convert_from_file(pathName)
  return (npImg, npNames)



    net = buildNetwork(len(t), len(t), 1)
    ds = SupervisedDataSet(len(t), 1)

    ds.addSample(loadImage('a.png'),(1,))
    ds.addSample(loadImage('b.png'),(2,))
    ds.addSample(loadImage('c.png'),(3,))
    ds.addSample(loadImage('d.png'),(4,))

    trainer = BackpropTrainer(net, ds)
    error = 10
    iteration = 0
    while error > 0.000000001:
        error = trainer.train()
        iteration += 1
        print "Iteration: {0} Error {1}".format(iteration, error)

    print "\nResult: 1", net.activate(loadImage('a.png'))
    print "\nResult: 2", net.activate(loadImage('b.png'))
    print "\nResult: 3", net.activate(loadImage('c.png'))
    print "\nResult: 4", net.activate(loadImage('d.png'))
    print "\nResult: 1", net.activate(loadImage('a.png'))
    print "\nResult: 1", net.activate(loadImage('a.png'))
    print "\nResult: 2", net.activate(loadImage('b.png'))

if __name__ == "__main__":

  (images, names) = loadImages("testimg/t10k-images-idx3-ubyte", "testimg/t10k-labels-idx1-ubyte")
  shape = images.shape
  size = shape[1] * shape[2]
  net = buildNetwork(size, size, 1)
  ds = SupervisedDataSet(size, 1)

  for i in range(0):
    ds.addSample(images[i].flatten(), (names[i],))

  print "ready to train"
  trainer = BackpropTrainer(net, ds)
  error = 10
  iteration = 0
  while error > 0.001:
    error = trainer.train()
    iteration += 1
    print "Iteration: {0} Error {1}".format(iteration, error)

  print "\nResult: {0} Actual {1} ".format(net.activate(images[500].flatten()), names[500])

