#!/usr/bin/env python

from struct import unpack
import gzip
from numpy import zeros, uint8, ravel
import numpy as np
import math
from matplotlib import pyplot as plt
import csv

from pylab import imshow, show, cm

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

from argparse import ArgumentParser
import os.path
import cPickle as pickle
import idx2numpy as inp


def get_labeled_data(picklename, samples):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        images = inp.convert_from_file("testimg/t10k-images-idx3-ubyte")
        labels = inp.convert_from_file("testimg/t10k-labels-idx1-ubyte")
        data = {"images": images, "labels": labels}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return {"images": data["images"][:samples], "labels": data["labels"][:samples]}



def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()

def visualizeData(data):
  print "Visualize data..."
  X = data["images"]
  idxs = np.random.randint(data["images"].shape[0], size=100)
  fig, ax = plt.subplots(10, 10)
  img_size = math.sqrt(X.shape[1]*X.shape[2])
  for i in range(10):
      for j in range(10):
          Xi = X[idxs[i * 10 + j]]
          ax[i, j].set_axis_off()
          ax[i, j].imshow(Xi, aspect="auto", cmap="gray")
  plt.show()


def classify(data, split, HIDDEN_NEURONS, MOMENTUM, WEIGHTDECAY,
             LEARNING_RATE, LEARNING_RATE_DECAY, EPOCHS, ldnetwork, logFileName):
    INPUT_FEATURES = data['images'].shape[1] * data['images'].shape[2]
    print("Input features: %i" % INPUT_FEATURES)
    CLASSES = 10
    alldata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)


    for i in range(len(data["images"])):
        alldata.addSample(ravel(data['images'][i]), [data['labels'][i]])

    trndata, tstdata = alldata.splitWithProportion(split)
    print "Train dataset: %4d" % len(trndata)
    print "Test dataset: %4d" % len(tstdata)
    # This is necessary, but I don't know why
    # See http://stackoverflow.com/q/8154674/562769
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    if ldnetwork:
      net = ldnetwork
    else:
      net = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim, outclass=SoftmaxLayer)

    trainer = BackpropTrainer(net, dataset=trndata, momentum=MOMENTUM,
                              verbose=True, weightdecay=WEIGHTDECAY,
                              learningrate=LEARNING_RATE,
                              lrdecay=LEARNING_RATE_DECAY)
    log = []
    for i in range(EPOCHS):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),
                                 trndata['class'])
        tstresult = percentError(trainer.testOnClassData(
                                 dataset=tstdata), tstdata['class'])

        print "Epoch %4d" % trainer.totalepochs + ",  Train error: %5.2f%%" % trnresult + ",  Test error: %5.2f%%" % tstresult
        if logFileName:
          log.append([trnresult, tstresult])
          with open(logFileName, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(log)
    return net

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add more options if you like
    parser.add_argument("-H", metavar="H", type=int, dest="hidden_neurons",
                        default=200,
                        help="number of neurons in the hidden layer")
    parser.add_argument("-e", metavar="EPOCHS", type=int,
                        dest="epochs", default=200,
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
    parser.add_argument("-f", metavar="FILE", type=str, dest="file",
                        default="network.xml",
                        help="network export name")
    parser.add_argument("-s", metavar="SAMPLES", type=int, dest="samples",
                        default=10000,
                        help="number of samples to use")
    parser.add_argument("-sp", metavar="SPLIT", type=float, dest="split",
                        default=0.7,
                        help="percentage used for training")
    parser.add_argument("-ln", metavar="LOADNET", type=str, dest="ldnet",
                        default="",
                        help="load a pre-trained network from a file")
    parser.add_argument("-v", metavar="VISUALIZE", type=bool, dest="visualize",
                        default=False,
                        help="visualize a sample of 100 images")
    parser.add_argument("-log", metavar="LOG", type=str, dest="log",
                        default="",
                        help="save the progress into a log file")
    args = parser.parse_args()

    print ""
    print "Starting Neural network with the following options:"
    print ""
    print "# Samples: %4d" % args.samples
    print "Training split: %.4f" % args.split
    print "Hidden Neurons: %4d" % args.hidden_neurons
    print "Epochs: %4d" % args.epochs
    print "Weight-decay: %.4f" % args.weightdecay
    print "Momentum: %.4f" % args.momentum
    print "Learning rate: %.4f" % args.learning_rate
    print "Learning rate decay: %.4f" % args.lrdecay
    print "Network Output: %s" % args.file
    print ""

    if args.ldnet:
      ldnetwork = NetworkReader.readFrom(args.ldnet)
    else:
      ldnetwork = False

    print("Getting dataset")
    data = get_labeled_data('data', args.samples)
    if args.visualize:
      visualizeData(data)
    print("Got %i datasets." % len(data['images']))
    net = classify(data, args.split, args.hidden_neurons, args.momentum,
             args.weightdecay, args.learning_rate, args.lrdecay, args.epochs, ldnetwork, args.log)

    NetworkWriter.writeToFile(net, args.file)
