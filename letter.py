#!/usr/bin/python
# (C) JACK CONCANON

from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import cv2

def loadImage(path):
    path = "testimg/"+path
    im = cv2.imread(path)
    return flatten(im)

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

if __name__ == "__main__":

    t = loadImage('a.png')

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
