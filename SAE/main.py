"""
################################### AwA DEMO #####################################
Following code shows a demo for AwA dataset to reproduce the result of the paper:

Semantic Autoencoder for Zero-shot Learning.
Elyor Kodirov, Tao Xiang, and Shaogang Gong
To appear in CVPR 2017.

You are supposed to get following:
[1] AwA ZSL accuracy [V >>> S]: 84.7%
[2] AwA ZSL accuracy [S >>> V]: 84.0%

However, the AwA (version 1) dataset is not available anymore and the AwA dataset
used in this demo is the AwA version 2 which has more 7000 instances.
The results, considering extracted features by VGG19 (4096-dim) are the following:
[1] AwA ZSL accuracy [V >>> S]: 75.6%
[2] AwA ZSL accuracy [S >>> V]: 70.4%

Author: Cristiano Patrício
E-mail: cristiano.patricio@ubi.pt
University of Beira Interior, Portugal
"""

import numpy as np
from scipy.linalg import solve_sylvester
from sklearn import preprocessing
from scipy.spatial import distance
import time
import pickle
from preprocess_data import get_test_classes, get_training_classes
import bz2
import _pickle as cPickle


##################################################################################
#                               AUXILIARY FUNCTIONS
##################################################################################


def decompress_pickle(file):
    """
    Load any compressed pickle file
    :param file: filename
    :return: data inside the compressed pickle file
    """
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)

    return data


def normalizeFeatures(X):
    """
    Normalize features L2-norm
    :param X: feature matrix of shape d x N
    :return: X_normalized - normalized features
    """

    X_normalized = preprocessing.normalize(X, norm='l2')

    return X_normalized


def SAE(X, S, lamb):
    """
    SAE - Semantic Autoencoder
    :param X: d x N data matrix
    :param S: k x N semantic matrix
    :param lamb: regularization parameter
    :return: W -> k x d projection function
    """

    A = np.dot(S, S.T)
    B = lamb * np.dot(X, X.T)
    C = (1 + lamb) * np.dot(S, X.T)
    W = solve_sylvester(A, B, C)

    return W


def zsl_accuracy(S_preds, S_test, y_test):
    """
    Calculates zero-shot classification accuracy
    :param S_preds: estimated semantic labels
    :param S_test: ground truth semantic labels
    :param y_test: test labels
    :return: acc - zero-shot classification accuracy
    """

    n_samples = S_test.shape[0]

    # Calculate distance between the estimated representation and the projected prototypes
    dist = distance.cdist(S_preds, normalizeFeatures(S_test.T).T, metric='cosine')
    # Get the index of min distances
    idx_min = np.argmin(dist, axis=1)
    # Get the labels of predictions
    preds = y_test[[i for i in idx_min]]

    # Calculate Top-1 accuracy
    diff = y_test - preds
    n_incorrect = len(np.nonzero(diff)[0])
    mean_accuracy = (n_samples - n_incorrect) / n_samples

    return preds, mean_accuracy


##################################################################################
#                                     DATA
##################################################################################

# Loading the AwA dataset
print('[INFO]: Loading dataset...')
tic = time.time()
X = decompress_pickle('AwA2-features-vgg19.pbz2')
X = np.array(X)
toc = time.time()
print("Time elapsed: %.2f" % (toc-tic))
S = np.loadtxt('/home/cristianopatricio/Documents/Datasets/AwA2-base/Animals_with_Attributes2/predicate-matrix'
               '-continuous.txt')
labels = np.loadtxt(
    '/home/cristianopatricio/Documents/Datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt')

train_classes = get_training_classes()
test_classes = get_test_classes()


##################################################################################
#                                 TRAINING PHASE
##################################################################################


"""
#X_train = np.random.randn(20, 50)
#X_train = normalizeFeatures(X_train.T).T
#S_train = np.random.randint(2, size=(20, 10))
"""

X_train = X[np.where([labels == i for i in train_classes])[1]]
X_train = normalizeFeatures(X_train.T).T
y_train = labels[np.where([labels == i for i in train_classes])[1]]
S_train = S[[int(i - 1) for i in y_train]]

# Training setup
lamb = 500000

print("######## Training INFO ########")
print("X_train shape -- ", X_train.T.shape)
print("S_train shape -- ", S_train.T.shape)
print("Lambda --------- ", lamb)
print("\n")

# SAE
print("[INFO]: Calculating W ...")
tic = time.time()
W = SAE(X_train.T, S_train.T, lamb)
toc = time.time()
print("W shape -------- ", W.shape)
print("[INFO]: Training time: %.2f seconds." % (toc - tic))
print("\n")

##################################################################################
#                                    TEST PHASE
##################################################################################

"""
X_test = np.random.randn(2, 50)
X_test = normalizeFeatures(X_test.T).T
S_test = np.random.randint(2, size=(2, 10))
"""

X_test = X[np.where([labels == i for i in test_classes])[1]]
X_test = normalizeFeatures(X_test.T).T
y_test = labels[np.where([labels == i for i in test_classes])[1]]
S_test = S[[int(i - 1) for i in y_test]]

print("######## Test INFO ########")
print("X_test shape -- ", X_test.shape)
print("S_test shape -- ", S_test.shape)
print("\n")

# Test [V >>> S]
S_pred = np.dot(X_test, normalizeFeatures(W).T)
print("S_pred shape -- ", S_pred.shape)
preds, acc = zsl_accuracy(S_pred, S_test, y_test)
print("[1] AwA ZSL accuracy [V >>> S]: %.1f%%" % (acc * 100))
print("\n")

# Test [S >>> V]
X_test_pred = np.dot(normalizeFeatures(S_test.T).T, normalizeFeatures(W))
print("X_test_pred shape -- ", X_test_pred.shape)
preds, acc = zsl_accuracy(X_test, X_test_pred, y_test)
print("[2] AwA ZSL accuracy [S >>> V]: %.1f%%" % (acc * 100))
