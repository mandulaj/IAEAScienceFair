#!/usr/bin/python
# (C) JACK CONCANON


import idx2numpy as idx2np
import numpy as np
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


def loadImages(pathImg, pathName):
  npImg = idx2np.convert_from_file(pathImg)
  npNames = idx2np.convert_from_file(pathName)
  return (npImg, npNames)


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
