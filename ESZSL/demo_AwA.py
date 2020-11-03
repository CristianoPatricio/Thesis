from sklearn import datasets, linear_model, preprocessing, decomposition, manifold, svm
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
import matplotlib.pyplot as plt
import time

#######################################################################
#   PREPROCESSING DATA
#######################################################################

# Loading the AwA dataset
X = np.loadtxt('/home/cristianopatricio/Documents/Datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt')
y = np.loadtxt('/home/cristianopatricio/Documents/Datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt')

print('The shape of X is: ' + str(X.shape))
print('The shape of Y is: ' + str(y.shape))
print('Number of classes: ' + str(len(np.unique(y))))

# Split into train and test sets (40 classes for training and 10 classe for test)
lbl = preprocessing.LabelEncoder()
y_train = lbl.fit_transform(y[np.where((y > 0) & (y < 41))])
X_train = X[np.where((y > 0) & (y < 41))]

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#print(np.unique(y_train))

#######################################################################
#   TRAIN
#######################################################################

#Create a svm Classifier
model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=5000)

tic = time.time()
#Train the model using the training sets
model.fit(X_train, y_train)
toc = time.time()

# We have Weight matrix, W z x d
W = model.coef_.T
#W = np.loadtxt('weights.txt')

# Saving the W in a text file 
np.savetxt("weights.txt", W) 

print("W shape: ", W.shape)

# Signatures matrix for training data
S = np.loadtxt('/home/cristianopatricio/Documents/Datasets/AwA2-base/Animals_with_Attributes2/predicate-matrix-binary.txt')
S_train = S[0:40].T

print("S_train shape: ", S_train.shape)
#print("S shape: ", S.shape)

# From W and S calculate V, d x a
V = np.linalg.lstsq(S_train.T, W.T, rcond=None)[0].T
print("V shape", V.shape)

W_new = np.dot(S_train.T, V.T).T
print("W_new shape: ", W_new.shape)

print("%f" % (np.sum(np.sqrt((W_new-W)**2))))

# Predictions that happens in the training phase of zero shot learning
#for ys, x in zip(y_train, X_train):
#	print(np.argmax(np.dot(x.T, W_new)), ys)

#################################################################
# INFERENCE
#################################################################
lbl = preprocessing.LabelEncoder()
y_test = lbl.fit_transform(y[np.where((y > 40) & (y < 51))])
X_test = X[np.where((y > 40) & (y < 51))]

S_test = S[40:].T

print("S_test shape: ", S_test.shape)

# Calculate the new Weight/Coefficient matrix		
W_new = np.dot(S_test.T, V.T).T
print("W_new shape: ", W_new.shape)

# Check performance
correct = 0
i = 0
for i, (ys, x) in enumerate(zip(y_test, X_test)):
	if np.argmax(np.dot(x.T, W_new)) == ys:
		correct += 1 


print("Results: ", correct, i, correct / float(i))
print("Training time: %.2f min." % ((toc-tic)/60.0))

###########################################################
# SAVE RESULTS TXT
###########################################################

file = open("results_eszsl_AwA2.txt","a")
file.write("No. samples: " + str(i) + "\n" + "No. correct samples: " + str(correct) + "\n" + "Percentage: " + str(correct / float(i)) + "\n" + "Exec. time: " + str((toc-tic)/60.0) + "min.")
file.close()
