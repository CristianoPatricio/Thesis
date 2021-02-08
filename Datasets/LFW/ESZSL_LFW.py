import numpy as np
import os
import scipy.io
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from sklearn import preprocessing
import pickle

class EsZSL():
    """docstring for ClassName"""

    def __init__(self):
        with open("lfw_vgg16.pickle", "rb") as f:
            dataset = pickle.load(f)

        with open("lfw_att_auto_split_50_norm.pickle", "rb") as f:
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

        self.s_trainval = attribute[self.trainval_labels_seen]
        self.s_test_unseen = attribute[self.test_labels_unseen]
        self.s_train = attribute[self.train_labels_seen]
        self.s_val = attribute[self.val_labels_unseen]

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

        # params for train and val set
        m_train = self.x_train.shape[0]
        z_train = len(self.train_labels_seen)

        # params for trainval and test set
        m_trainval = self.x_trainval.shape[0]
        z_trainval = len(self.trainval_labels_seen)

        print("INFO")
        print(self.x_trainval.shape)
        print(self.trainval_labels_seen.shape)

        # ground truth for train and val set
        self.gt_train = 0 * np.ones((m_train, z_train))
        self.gt_train[np.arange(m_train), np.squeeze(self.y_train)] = 1

        # grountruth for trainval and test set
        self.gt_trainval = 0 * np.ones((m_trainval, z_trainval))
        self.gt_trainval[np.arange(m_trainval), np.squeeze(self.y_trainval)] = 1

    def find_hyperparams(self):
        # train set
        d_train = self.x_train.shape[1]
        a_train = self.s_train.shape[1]

        accu = 0.10
        alph1 = 4
        gamm1 = 1

        # Weights
        V = np.zeros((d_train, a_train))
        for alpha in range(-3, 4):
            for gamma in range(-3, 4):
                # One line solution
                part_1 = np.linalg.pinv(
                    np.matmul(self.x_train.transpose(), self.x_train) + (10 ** alpha) * np.eye(d_train))
                part_0 = np.matmul(np.matmul(self.x_train.transpose(), self.gt_train), self.s_train)
                part_2 = np.linalg.pinv(
                    np.matmul(self.s_train.transpose(), self.s_train) + (10 ** gamma) * np.eye(a_train))

                V = np.matmul(np.matmul(part_1, part_0), part_2)
                # print(V)

                # predictions
                outputs = np.matmul(np.matmul(self.x_val, V), self.s_val.transpose())
                preds = np.array([np.argmax(output) for output in outputs])

                # print(accuracy_score(labels_val,preds))
                cmat = confusion_matrix(self.y_val, preds)
                per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

                avg = np.mean(per_class_acc)

                if avg > accu:
                    accu = avg
                    alph1 = alpha
                    gamm1 = gamma
                    print(alph1, gamm1, avg)
        print("Alpha and gamma:", alph1, gamm1)
        return alph1, gamm1

    def train(self, alpha, gamma):
        print("X_trainval shape", self.x_trainval.shape)
        print("Gt trainval shape", self.gt_trainval.shape)
        print("S trainval shape", self.s_trainval.shape)
        # trainval set
        d_trainval = self.x_trainval.shape[1]
        a_trainval = self.s_trainval.shape[1]
        W = np.zeros((d_trainval, a_trainval))
        part_1_test = np.linalg.pinv(
            np.matmul(self.x_trainval.transpose(), self.x_trainval) + (10 ** alpha) * np.eye(d_trainval))
        print("Part 1 Shape", part_1_test.shape)
        #print(self.trainval_sig.transpose().shape)
        part_0_test = np.matmul(np.matmul(self.x_trainval.transpose(), self.gt_trainval), self.s_trainval)
        print("Part 0 Shape", part_0_test.shape)
        part_2_test = np.linalg.pinv(
            np.matmul(self.s_trainval.transpose(), self.s_trainval) + (10 ** gamma) * np.eye(a_trainval))
        print("Part 2 Shape", part_2_test.shape)
        W = np.matmul(np.matmul(part_1_test, part_0_test), part_2_test)
        print("W shape", W.shape)
        return W

    def gzsl_accuracy(self, weights):
        """
        Calculate harmonic mean
        :param y_true: ground truth labels
        :param y_preds: estimated labels
        :param seen_classes: array of seen classes
        :param unseen_classes: array of unseen classes
        :return: harmonic mean
        """
        #print("Unseen classes: ", np.unique(self.y_test_gzsl))
        #print("Seen classes: ", np.unique(self.y_test_seen))
        #print("All classes: ", np.unique(self.y_test_unseen))

        outputs_1 = np.matmul(np.matmul(self.x_test_gzsl, weights), self.s_test_gzsl.transpose())
        preds_1 = np.argmax(outputs_1, axis=1)
        cmat = confusion_matrix(self.y_test_gzsl, preds_1)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        #print("Initial seen classes: ", np.unique(self.decoded_y_seen))
        #print("Initial unseen classes: ", np.unique(self.decoded_y_unseen))
        #print("Initial gzsl classes: ", np.unique(self.decoded_y_test))

        seen_classes_encoded = self.y_test_gzsl[np.where([self.decoded_y_test == i for i in self.decoded_y_seen])[1]]
        unseen_classes_encoded = self.y_test_gzsl[np.where([self.decoded_y_test == i for i in self.decoded_y_unseen])[1]]

        acc_seen = np.mean(per_class_acc[seen_classes_encoded])
        print("Accuracy seen classes: %.3f %%" % (acc_seen * 100))
        acc_unseen = np.mean(per_class_acc[unseen_classes_encoded])
        print("Accuracy unseen classes: %.3f %%" % (acc_unseen * 100))
        harmonic_mean = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        print("The harmonic mean is:", harmonic_mean * 100)

        return harmonic_mean

    def zsl_accuracy(self, y_pred):
        cmat = confusion_matrix(self.y_test_unseen, y_pred)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        acc = np.mean(per_class_acc)

        return acc

    def test(self, weights):
        # predictions
        #print("......")
        print("x_test: ", self.x_test_unseen.shape)
        print("s_test: ", self.s_test_unseen.shape)

        outputs_1 = np.matmul(np.matmul(self.x_test_unseen, weights), self.s_test_unseen.transpose())
        preds_1 = np.argmax(outputs_1, axis=1)
        zsl_acc = self.zsl_accuracy(preds_1)
        print("The top 1% accuracy is:", zsl_acc * 100)
        return zsl_acc


if __name__ == "__main__":
    model = EsZSL()
    alpha = 0
    gamma = 0
    #alpha, gamma = model.find_hyperparams()
    weights = model.train(alpha, gamma)
    model.test(weights)
    model.gzsl_accuracy(weights)
