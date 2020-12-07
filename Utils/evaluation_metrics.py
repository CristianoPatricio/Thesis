import numpy as np
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

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
    dist = distance.cdist(S_preds, S_test, metric='cosine')
    # Get the index of min distances
    idx_min = np.argmin(dist, axis=1)
    # Get the labels of predictions
    preds = y_test[[i for i in idx_min]]

    # Calculate Top-1 accuracy
    diff = y_test - preds
    n_incorrect = len(np.nonzero(diff)[0])
    mean_accuracy = (n_samples - n_incorrect) / n_samples

    return preds, mean_accuracy


def gzsl_accuracy(y_true, y_preds, seen_classes, unseen_classes, decoded_y_true):
    """
    Calculate harmonic mean
    :param y_true: ground truth labels
    :param y_preds: estimated labels
    :param seen_classes: array of seen classes
    :param unseen_classes: array of unseen classes
    :return: harmonic mean
    """

    cmat = confusion_matrix(y_true, y_preds)
    per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

    seen_classes_encoded = y_true[np.where([decoded_y_true == i for i in seen_classes])[1]]
    unseen_classes_encoded = y_true[np.where([decoded_y_true == i for i in unseen_classes])[1]]

    acc_seen = np.mean(per_class_acc[seen_classes_encoded])
    print("Accuracy seen classes: %.3f %%" % (acc_seen * 100))
    acc_unseen = np.mean(per_class_acc[unseen_classes_encoded])
    print("Accuracy unseen classes: %.3f %%" % (acc_unseen * 100))

    harmonic_mean = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)

    return harmonic_mean