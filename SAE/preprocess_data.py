import pickle
import bz2
import _pickle as cPickle
import numpy as np


# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)


# Get all classes
classes = {}
with open('classes.txt') as f_classes:
    lines = f_classes.readlines()
    for l in lines:
        classes[l.strip().split("\t")[1]] = l.strip().split("\t")[0]


# Get training classes
def get_training_classes():
    train_classes = []
    with open("trainclasses.txt") as f_tclasses:
        lines = f_tclasses.readlines()
        for l in lines:
            classname = l.strip()
            train_classes.append(int(classes[classname]))
    train_classes = np.array(train_classes)

    return train_classes


# Get test classes
def get_test_classes():
    test_classes = []
    with open("testclasses.txt") as f:
        lines = f.readlines()
        for l in lines:
            classname = l.strip()
            test_classes.append(int(classes[classname]))
    test_classes = np.array(test_classes)

    return test_classes


# MAIN
X = np.loadtxt('/home/cristianopatricio/Documents/Datasets/Animals_with_Attributes2/Features/VGG19/AwA2-features-vgg19'
               '.txt')
compressed_pickle('AwA2-features-vgg19', X)
