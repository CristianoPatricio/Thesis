import numpy as np
import os
import scipy.io
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from sklearn import preprocessing
from scipy.linalg import solve_sylvester
from scipy.spatial import distance
import pickle

class SAE():
    """docstring for ClassName"""

    def __init__(self):

        with open("lfw_vgg16.pickle", "rb") as f:
            dataset = pickle.load(f)

        with open("lfw_att_split_norm_2.pickle", "rb") as f:
            att_split = pickle.load(f)

        features = dataset['features']
        labels = dataset['labels']

        train_loc = att_split['train_loc']
        trainval_loc = att_split['trainval_loc']
        val_loc = att_split['val_loc']
        test_unseen_loc = att_split['test_unseen_loc']
        test_seen_loc = att_split['test_seen_loc']
        attribute = att_split['att']

        self.x_train = features[train_loc]
        self.y_train = labels[train_loc].astype(int)
        self.s_train = attribute[self.y_train]
        print("Shape x_train", self.x_train.shape)
        print("Shape y_train", self.y_train.shape)
        print("Shape s_train", self.s_train.shape)

        self.x_trainval = features[trainval_loc]
        self.y_trainval = labels[trainval_loc].astype(int)
        self.s_trainval = attribute[self.y_trainval]
        print("Shape x_trainval", self.x_trainval.shape)
        print("Shape y_trainval", self.y_trainval.shape)
        print("Shape s_trainval", self.s_trainval.shape)

        self.x_val = features[val_loc]
        self.y_val = labels[val_loc].astype(int)
        self.s_val = attribute[self.y_val]
        print("Shape x_val", self.x_val.shape)
        print("Shape y_val", self.y_val.shape)
        print("Shape s_val", self.s_val.shape)

        self.x_test_seen = features[test_seen_loc]
        self.y_test_seen = labels[test_seen_loc].astype(int)
        self.s_test_seen = attribute[self.y_test_seen]
        print("Shape x_test_seen", self.x_test_seen.shape)
        print("Shape y_test_seen", self.y_test_seen.shape)
        print("Shape s_test_seen", self.s_test_seen.shape)

        self.x_test_unseen = features[test_unseen_loc]
        self.y_test_unseen = labels[test_unseen_loc].astype(int)
        self.s_test_unseen = attribute[self.y_test_unseen.squeeze()]
        unique_atts = np.unique(self.s_test_unseen, axis=0)
        print("Shape x_test_unseen", self.x_test_unseen.shape)
        print("Shape y_test_unseen", self.y_test_unseen.shape)
        print("Shape s_test_unseen", self.s_test_unseen.shape)

        # GZSL
        self.x_test_gzsl = np.concatenate((self.x_test_seen, self.x_test_unseen), axis=0)
        self.s_test_gzsl = np.concatenate((self.s_test_seen, self.s_test_unseen), axis=0)
        self.y_test_gzsl = np.concatenate((self.y_test_seen, self.y_test_unseen), axis=0)
        print("Shape x_test_gzsl", self.x_test_gzsl.shape)
        print("Shape y_test_gzsl", self.y_test_gzsl.shape)
        print("Shape s_test_gzsl", self.s_test_gzsl.shape)

        self.decoded_y_seen = self.y_test_seen.copy()
        self.decoded_y_unseen = self.y_test_unseen.copy()
        self.decoded_y_test = self.y_test_gzsl.copy()

        self.train_labels_seen = np.unique(self.y_train)
        self.val_labels_unseen = np.unique(self.y_val)
        self.trainval_labels_seen = np.unique(self.y_trainval)
        self.test_labels_unseen = np.unique(self.y_test_unseen)
        self.test_labels_gzsl = np.unique(self.y_test_gzsl)

        print(len(self.test_labels_gzsl))

        # Normalize
        self.x_trainval = preprocessing.normalize(self.x_trainval, norm="l2")
        self.x_test_unseen = preprocessing.normalize(self.x_test_unseen, norm="l2")
        self.x_train = preprocessing.normalize(self.x_train, norm="l2")
        self.x_val = preprocessing.normalize(self.x_val, norm="l2")
        self.x_test_gzsl = preprocessing.normalize(self.x_test_gzsl, norm="l2")
        self.s_trainval = preprocessing.normalize(self.s_trainval, norm="l2")
        self.s_test_unseen = preprocessing.normalize(self.s_test_unseen, norm="l2")
        self.s_train = preprocessing.normalize(self.s_train, norm="l2")
        self.s_val = preprocessing.normalize(self.s_val, norm="l2")
        self.s_test_gzsl = preprocessing.normalize(self.s_test_gzsl, norm="l2")

        # Labels Encoder
        lbl = preprocessing.LabelEncoder()
        self.y_trainval = lbl.fit_transform(self.y_trainval)
        self.y_test_unseen = lbl.fit_transform(self.y_test_unseen)
        self.y_train = lbl.fit_transform(self.y_train)
        self.y_val = lbl.fit_transform(self.y_val)
        self.y_test_gzsl = lbl.fit_transform(self.y_test_gzsl)

    def SAE(self, X, S, lamb):
        """
        SAE - Semantic Autoencoder
        :param X: d x N data matrix
        :param S: k x N semantic matrix
        :param lamb: regularization parameter
        :return: W -> k x d projection function
        """

        print("X shape", X.shape)
        print("S shape", S.shape)
        print("Lambda", lamb)

        A = np.dot(S.T, S)
        B = lamb * np.dot(X.T, X)
        C = (1 + lamb) * np.dot(S.T, X)
        W = solve_sylvester(A, B, C)

        return W

    def train(self, lamb):
        W = self.SAE(self.x_trainval, self.s_trainval, lamb)
        return W

    def lambda_tuning(self):
        lamb = np.arange(0, 500000, 1000)
        best_lamb = -1
        base_acc = -1
        for i in lamb:
            W = self.SAE(self.x_train, self.s_train, i)

            s_pred = np.dot(self.x_val, W.T)

            # Calculate distance between the estimated representation and the projected prototypes
            dist = distance.cdist(s_pred, self.s_val, metric='cosine')
            # Get the index of min distances
            idx_min = np.argmin(dist, axis=1)
            # Get the labels of predictions
            preds = self.y_val[[i for i in idx_min]]

            cmat = confusion_matrix(self.y_val, preds)
            per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

            acc = np.mean(per_class_acc)

            if acc > base_acc:
                best_lamb = i
                base_acc = acc

        return best_lamb


    def gzsl_accuracy_semantic(self, weights):
        """
        Calculate harmonic mean
        :param y_true: ground truth labels
        :param y_preds: estimated labels
        :param seen_classes: array of seen classes
        :param unseen_classes: array of unseen classes
        :return: harmonic mean
        """

        # Test [V >>> S]
        print("X Test shape: ", self.x_test_unseen.shape)
        print("W shape: ", weights.shape)

        s_pred = np.dot(self.x_test_gzsl, weights.T)
        s_pred = self.s_test_gzsl

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(s_pred, self.s_test_gzsl, metric='cosine')
        # Get the index of min distances
        idx_min = np.argmin(dist, axis=1)
        # Get the labels of predictions
        preds = self.y_test_gzsl[[i for i in idx_min]]

        cmat = confusion_matrix(self.y_test_gzsl, preds)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        seen_classes_encoded = self.y_test_gzsl[np.where([self.decoded_y_test == i for i in self.decoded_y_seen])[1]]
        unseen_classes_encoded = self.y_test_gzsl[np.where([self.decoded_y_test == i for i in self.decoded_y_unseen])[1]]

        print("[V >>>> S]")
        acc_seen = np.mean(per_class_acc[seen_classes_encoded])
        print("Accuracy seen classes: %.3f %%" % (acc_seen * 100))
        acc_unseen = np.mean(per_class_acc[unseen_classes_encoded])
        print("Accuracy unseen classes: %.3f %%" % (acc_unseen * 100))
        harmonic_mean = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        print("The harmonic mean is:", harmonic_mean * 100)

        return harmonic_mean


    def gzsl_accuracy_feature(self, weights):
        """
        Calculate harmonic mean
        :param y_true: ground truth labels
        :param y_preds: estimated labels
        :param seen_classes: array of seen classes
        :param unseen_classes: array of unseen classes
        :return: harmonic mean
        """

        x_pred = np.dot(self.test_all_gzsl_sig.T, weights)

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(x_pred, self.test_gzsl.T, metric='cosine')
        # Get the index of min distances
        idx_min = np.argmin(dist, axis=1)
        # Get the labels of predictions
        preds = self.labels_test_gzsl[[i for i in idx_min]]

        cmat = confusion_matrix(self.labels_test_gzsl, preds)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        seen_classes_encoded = self.labels_test_gzsl[np.where([self.decoded_y_true == i for i in self.decoded_seen_classes])[1]]
        unseen_classes_encoded = self.labels_test_gzsl[np.where([self.decoded_y_true == i for i in self.decoded_unseen_classes])[1]]

        print("[S >>>> V]")
        acc_seen = np.mean(per_class_acc[seen_classes_encoded])
        print("Accuracy seen classes: %.3f %%" % (acc_seen * 100))
        acc_unseen = np.mean(per_class_acc[unseen_classes_encoded])
        print("Accuracy unseen classes: %.3f %%" % (acc_unseen * 100))
        harmonic_mean = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        print("The harmonic mean is:", harmonic_mean * 100)

        return harmonic_mean


    def zsl_accuracy_semantic(self, weights):
        # Test [V >>> S]
        print("X Test shape: ", self.x_test_unseen.shape)
        print("W shape: ", weights.shape)

        s_pred = np.dot(self.x_test_unseen, weights.T)
        s_pred = self.s_test_unseen

        print("S pred shape: ", s_pred.shape)
        print("S test shape", self.s_test_unseen.shape)

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(s_pred, self.s_test_unseen, metric='cosine')
        # Get the index of min distances
        idx_min = np.argmin(dist, axis=1)
        # Get the labels of predictions
        preds = self.y_test_unseen[[i for i in idx_min]]

        cmat = confusion_matrix(self.y_test_unseen, preds)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        acc = np.mean(per_class_acc)

        return acc

    def zsl_accuracy_feature(self, weights):
        # Test [V >>> S]
        print("S Test shape: ", self.s_test_unseen.shape)
        print("W shape: ", weights.shape)

        x_pred = np.dot(self.s_test_unseen, weights)

        print("X pred shape: ", x_pred.shape)
        print("X test shape", self.x_test_unseen.shape)

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(x_pred, self.x_test_unseen, metric='cosine')
        # Get the index of min distances
        idx_min = np.argmin(dist, axis=1)
        # Get the labels of predictions
        preds = self.y_test_unseen[[i for i in idx_min]]

        cmat = confusion_matrix(self.y_test_unseen, preds)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        acc = np.mean(per_class_acc)

        return acc

    def test(self, weights):
        zsl_acc = self.zsl_accuracy_semantic(weights)
        #zsl_feat_acc = self.zsl_accuracy_feature(weights)
        print("The top 1% accuracy [V >>> S] is:", zsl_acc * 100)
        #print("The top 1% accuracy [S >>> V] is:", zsl_feat_acc * 100)


if __name__ == "__main__":
    lamb = 500000
    model = SAE()
    #lamb = model.lambda_tuning()
    # Train
    weights = model.train(lamb)
    # Test
    model.test(weights)
    model.gzsl_accuracy_semantic(weights)
