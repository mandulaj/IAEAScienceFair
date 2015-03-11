#!/usr/bin/env python

from struct import unpack
import gzip
from numpy import zeros, uint8, ravel

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
        images = inp.convert_from_file("testimg/t10k-images-idx3-ubyte")[:samples]
        labels = inp.convert_from_file("testimg/t10k-labels-idx1-ubyte")[:samples]
        data = {"images": images, "labels": labels}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data



def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()


def classify(data, split, HIDDEN_NEURONS, MOMENTUM, WEIGHTDECAY,
             LEARNING_RATE, LEARNING_RATE_DECAY, EPOCHS):
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

    net = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim,
                       outclass=SoftmaxLayer)

    trainer = BackpropTrainer(net, dataset=trndata, momentum=MOMENTUM,
                              verbose=True, weightdecay=WEIGHTDECAY,
                              learningrate=LEARNING_RATE,
                              lrdecay=LEARNING_RATE_DECAY)
    for i in range(EPOCHS):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),
                                 trndata['class'])
        tstresult = percentError(trainer.testOnClassData(
                                 dataset=tstdata), tstdata['class'])

        print "Epoch %4d" % trainer.totalepochs + ",  Train error: %5.2f%%" % trnresult + ",  Test error: %5.2f%%" % tstresult
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

    print("Getting dataset")
    data = get_labeled_data('data', args.samples)
    print("Got %i datasets." % len(data['images']))
    net = classify(data, args.split, args.hidden_neurons, args.momentum,
             args.weightdecay, args.learning_rate, args.lrdecay, args.epochs)

    NetworkWriter.writeToFile(net, args.file)



