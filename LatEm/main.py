"""
Python code for testing the performance of the latent embedding model described in
Y. Xian, Z. Akata, G. Sharma, Q. Nguyen, M. Hein, B. Schiele.
Latent Embeddings for Zero-shot Classification. IEEE CVPR 2016.

Author: Cristiano Patrício
E-mail: cristiano.patricio@ubi.pt
University of Beira Interior, Portugal
"""

import os
import numpy as np
from scipy import stats
from sklearn import preprocessing
import time
import pickle

#######################################################################
#   AUXILIARY FUNCTIONS
#######################################################################

def l2_normalization(X):

    norm = np.sqrt(np.sum(X**2, axis=1))
    l2norm = X / norm[:,None]

    return l2norm


def zscore_normalization(X):
    """
    Compute the z-score over image features X
    :param X: image embedding matrix, each row is an instance
    :return: z-score
    """

    z_score = stats.zscore(X, axis=1)

    return z_score


def w_init(X, Y, K):
    """
    Initialization of matrix W
    :param X: images embedding matrix, each row is an image instance
    :param Y: class embedding matrix, each row is a class
    :param K: number of embeddings to learn
    :return: a matrix with K embeddings
    """

    dim_X = X.shape[1]
    dim_Y = Y.shape[0]

    W = []
    for i in range(K):
        W.append(np.random.randn(dim_X, dim_Y) * 1.0 / np.sqrt(dim_X))

    W = np.array(W)

    return W


def argmax_over_matrices(x, y, W):
    """
    Calculates the maximum score over matrices
    :param x: an image embedding instance
    :param y: a class embedding
    :param W: a cell array of embeddings
    :return best_score: best bilinear score among all the embeddings
    :return best_idx: index of the embedding with the best score
    """

    K = len(W)
    best_score = -1e12
    best_idx = -1
    score = np.zeros((K, 1))

    for i in range(K):
        projected_x = np.dot(x, W[i])
        projected_x = projected_x.astype(float)
        y = y.astype(float)
        score[i] = np.dot(projected_x, y)
        if score[i] > best_score:
            best_score = score[i]
            best_idx = i

    return best_score, best_idx


def latem_train(X, labels, Y, learning_rate, n_epochs, K):
    """
    SGD optimization for LatEm
    :param X: images embedding matrix, each row is an image instance
    :param labels: ground truth labels of all image instances
    :param Y: class embedding matrix, each row is for a class
    :param learning_rate: learning rate for SGD algorithm
    :param n_epochs: number of epochs for SGD algorithm
    :param K: number of embeddings to learn
    :return W: a cell array with K embeddings
    """

    n_train = X.shape[0]
    n_class = len(np.unique(labels))

    W = w_init(X, Y, K)

    tic = time.time()
    for i in range(n_epochs):
        print("[INFO]: Epoch %d / %d" % (i+1, n_epochs))
        perm = np.random.permutation(n_train)
        for j in range(n_train):
            n_i = perm[j]  # Choose a training instance
            best_j = -1
            picked_y = labels[n_i]  # Correspondent class label for the chosen training instance
            while picked_y == labels[
                n_i]:  # Enforces to randomly select an y that is different from yn (the correspondent to xn)
                picked_y = np.random.randint(n_class)
            max_score, best_j = argmax_over_matrices(X[n_i, :], Y[:, picked_y],
                                                     W)  # Get max score over W_i matrices given an x and the random y
            best_score_yi, best_j_yi = argmax_over_matrices(X[n_i, :], Y[:, labels[n_i] - 1],
                                                            W)  # Get max score over W_i matrices given an x and the correspondent y
            if max_score + 1 > best_score_yi:
                if best_j == best_j_yi:
                    # print(W[best_j].shape)
                    X = X.astype(float)
                    Y = Y.astype(float)
                    W[best_j] = W[best_j] - np.dot(np.multiply(learning_rate, X[n_i, :].reshape((X.shape[1], 1))),
                                                   (Y[:, picked_y] - Y[:, labels[n_i] - 1]).reshape((1, Y.shape[0])))
                else:
                    X = X.astype(float)
                    Y = Y.astype(float)
                    W[best_j] = W[best_j] - np.dot(np.multiply(learning_rate, X[n_i, :].reshape((X.shape[1], 1))),
                                                   Y[:, picked_y].reshape((1, Y.shape[0])))
                    W[best_j_yi] = W[best_j_yi] + np.dot(np.multiply(learning_rate, X[n_i, :].reshape((X.shape[1], 1))),
                                                         Y[:, labels[n_i] - 1].reshape((1, Y.shape[0])))
    toc = time.time()

    training_time = (toc-tic)/60.0

    return W, training_time


def latem_test(W, X, Y, labels):
    """
    Perform classification task and returns mean class accuracy
    :param W: latent embeddings
    :param X: images embedding matrix, each row is an image instance
    :param Y: class embedding matrix, each row is for a class
    :param labels: ground truth labels of all image instances
    :return: the classification accuracy averaged over all classes
    """

    n_samples = X.shape[0]
    preds = []

    K = len(W)
    scores = {}
    max_scores = {}
    idx = {}

    X = X.astype(float)
    Y = Y.astype(float)

    print("[INFO]: Testing...")

    for i in range(K):
        projected_X = np.dot(X, W[i])
        scores[i] = np.dot(projected_X, Y)
        max_scores[i], idx[i] = np.sum(scores[i], axis=1), np.argmax(scores[i], axis=1)

    # Convert dict into matrix
    dataMatrix = np.array([max_scores[i] for i in range(K)])
    # Get list with maximum_scores
    maximum_scores = np.amax(dataMatrix, axis=0)
    # Get index of chosen latent embedding
    idxs = np.argwhere(dataMatrix == maximum_scores)
    final_idx = idxs[np.argsort(idxs[:, 1]), 0]

    for i, index in enumerate(final_idx):
        # Get value of preds
        preds.append(idx[index][i])
    preds = np.array(preds)

    diff = preds - labels
    n_incorrect = len(np.nonzero(diff)[0])

    mean_accuracy = (n_samples - n_incorrect) / n_samples

    return preds, mean_accuracy


def get_emb_vectors(stage="train"):
    """
    Get embedding vectors of classes using GloVe
    :return: vectors
    """

    if stage == "train":
        classes = ['antelope', 'grizzly', 'killer', 'beaver', 'dalmatian', 'horse', 'shepherd', 'whale', 'siamese', 'skunk',
                   'mole', 'tiger', 'moose', 'monkey', 'elephant', 'gorilla', 'ox', 'fox', 'sheep',
                   'hamster', 'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'weasel',
                   'otter', 'buffalo', 'zebra', 'deer', 'bobcat', 'lion', 'mouse', 'bear', 'collie', 'walrus',
                   'cow', 'dolphin']
    else:
        classes = ['chimpanzee', 'panda', 'leopard', 'cat', 'pig', 'hippopotamus', 'whale', 'raccoon', 'rat', 'seal']

    vectors = []
    f = open("glove.6B.300d.txt")
    for i in f:
        word = i.split()[0]
        if word in classes:
            vectors.append(i.split()[1:])

    vectors = np.array(vectors)
    f.close()

    return vectors


#######################################################################
#   PREPROCESSING DATA
#######################################################################

# Loading the AwA dataset
print('[INFO]: Loading dataset...')

labels = np.loadtxt(
    '../Datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt')

if not os.path.exists('../Datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-features.pkl'):
    X = np.loadtxt(
        '../Datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt')
    pickle.dump(X, open('../Datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-features.pkl', "wb"))
else:
    X = pickle.load(open('../Datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-features.pkl', "rb"))

# Get all classes
classes = {}
with open('../Datasets/Animals_with_Attributes2/classes.txt') as f_classes:
    lines = f_classes.readlines()
    for l in lines:
        classes[l.strip().split("\t")[1]] = l.strip().split("\t")[0]

# Get training classes
train_classes = []
with open("Data/trainclasses.txt") as f_tclasses:
    lines = f_tclasses.readlines()
    for l in lines:
        classname = l.strip()
        train_classes.append(int(classes[classname]))
train_classes = np.array(train_classes)

# Get test classes
test_classes = []
with open("Data/testclasses.txt") as f:
    lines = f.readlines()
    for l in lines:
        classname = l.strip()
        test_classes.append(int(classes[classname]))
test_classes = np.array(test_classes)

# Split into train and test sets (40 classes for training and 10 classe for test)
lbl = preprocessing.LabelEncoder()
y_train = lbl.fit_transform(labels[np.where([labels == i for i in train_classes])[1]])
X_train = X[np.where([labels == i for i in train_classes])[1]]
X_train = zscore_normalization(X_train)

S = np.loadtxt('../Datasets/Animals_with_Attributes2/predicate-matrix-continuous.txt')
# l2-normalize the samples (rows).
S_normalized = preprocessing.normalize(S, norm='l2', axis=1)
#S_normalized = l2_normalization(S)
#Y = get_emb_vectors(stage="train").T
Y = S_normalized[[(i-1) for i in train_classes]].T

#######################################################################
#   TRAINING
#######################################################################

# X = np.random.randn(20, 1024)
# labels = np.random.randint(3, size=20) + 1
# Y = np.random.randint(3, size=(85, 3))
learning_rate = 1e-3
n_epochs = 10
K = 6

print("########### TRAINING INFO ###########")
print("X shape --- ", X_train.shape)
print("Y shape --- ", Y.shape)
print("Labels --- ", y_train.shape)
print("Learning rate --- ", learning_rate)
print("No. epochs --- ", n_epochs)
print("No. latent embeddings --- ", K)

W, training_time = latem_train(X_train, y_train, Y, learning_rate, n_epochs, K)
print("W shape --- ", W.shape)
print("[INFO]: Training time: %.2f minutes" % (training_time))

#######################################################################
#   TEST
#######################################################################

lbl = preprocessing.LabelEncoder()
y_test = lbl.fit_transform(labels[np.where([labels == i for i in test_classes])[1]])
X_test = X[np.where([labels == i for i in test_classes])[1]]
X_test = zscore_normalization(X_test)
#Y = get_emb_vectors(stage="test").T
Y = S_normalized[[(i-1) for i in test_classes]].T

#X = np.random.randn(10, 1024)
#X = normalization(X)
#labels = np.random.randint(3, size=10) + 1
#Y = np.random.randint(3, size=(85, 3))

print("########## TEST INFO ##########")
print("X shape --- ", X_test.shape)
print("Y shape --- ", Y.shape)
print("Labels --- ", y_test)

preds, acc = latem_test(W, X_test, Y, y_test)
print("Preds --- ", preds)
print("Accuracy --- %.2f %%" % (acc * 100))
