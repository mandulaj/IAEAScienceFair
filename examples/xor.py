#!/usr/bin/env python
# -*- coding: utf8 -*-

import pybrain
import math
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

net = buildNetwork(2, 2, 1, bias=True, hiddenclass=TanhLayer)

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

trainer = BackpropTrainer(net, ds)

print 'Untrained:'
act = net.activate([1, 0])[0]
print [1, 0], act, "≈", math.floor(act + 0.5)
act = net.activate([0, 1])[0]
print [0, 1], act, "≈", math.floor(act + 0.5)
act = net.activate([0, 0])[0]
print [0, 0], act, "≈", math.floor(act + 0.5)
act = net.activate([1, 1])[0]
print [1, 1], act, "≈", math.floor(act + 0.5)

c = 0
while trainer.train() > .001:
  c += 1

print "Number of Epochs: %d" % c


print 'Trained:'
act = net.activate([1, 0])[0]
print [1, 0], act, "≈", math.floor(act + 0.5)
act = net.activate([0, 1])[0]
print [0, 1], act, "≈", math.floor(act + 0.5)
act = net.activate([0, 0])[0]
print [0, 0], act, "≈", math.floor(act + 0.5)
act = net.activate([1, 1])[0]
print [1, 1], act, "≈", math.floor(act + 0.5)
