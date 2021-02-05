import numpy as np
import pickle

with open("pubfig_vgg16.pickle", "rb") as f:
    dataset = pickle.load(f)

#print(dataset)

with open("pubfig_att_split_norm.pickle", "rb") as f:
    att_split = pickle.load(f)

#print(att_split)

features = dataset['features']
labels = dataset['labels']

print("Shape features: ", features.shape)
print("Shape labels", labels.shape)

train_loc = att_split['train_loc']
trainval_loc = att_split['trainval_loc']
val_loc = att_split['val_loc']
test_unseen_loc = att_split['test_unseen_loc']
test_seen_loc = att_split['test_seen_loc']
attribute = att_split['att']

x_train = features[train_loc]
y_train = labels[train_loc].astype(int)
s_train = attribute[y_train]
print("Shape x_train", x_train.shape)
print("Shape y_train", y_train.shape)
print("Shape s_train", s_train.shape)

x_trainval = features[trainval_loc]
y_trainval = labels[trainval_loc].astype(int)
s_trainval = attribute[y_trainval]
print("Shape x_trainval", x_trainval.shape)
print("Shape y_trainval", y_trainval.shape)
print("Shape s_trainval", s_trainval.shape)

x_val = features[val_loc]
y_val = labels[val_loc].astype(int)
s_val = attribute[y_val]
print("Shape x_val", x_val.shape)
print("Shape y_val", y_val.shape)
print("Shape s_val", s_val.shape)

x_test_seen = features[test_seen_loc]
y_test_seen = labels[test_seen_loc].astype(int)
s_test_seen = attribute[y_test_seen]
print("Shape x_test_seen", x_test_seen.shape)
print("Shape y_test_seen", y_test_seen.shape)
print("Shape s_test_seen", s_test_seen.shape)

x_test_unseen = features[test_unseen_loc]
y_test_unseen = labels[test_unseen_loc].astype(int)
s_test_unseen = attribute[y_test_unseen]
print("Shape x_test_unseen", x_test_unseen.shape)
print("Shape y_test_unseen", y_test_unseen.shape)
print("Shape s_test_unseen", s_test_unseen.shape)

