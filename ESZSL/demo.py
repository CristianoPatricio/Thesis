from sklearn import datasets, linear_model, preprocessing, decomposition, manifold
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
import matplotlib.pyplot as plt

# Loading the MNIST dataset
X, y = datasets.load_digits().data, datasets.load_digits().target

# There are d dimensions, d=64
# z classes, z=6, [digit0, digit1, digit2, digit7, digit8, digit9]
lbl = preprocessing.LabelEncoder()
y_train = lbl.fit_transform(y[np.where((y == 0) | (y == 1) | (y == 2) | (y == 7) | (y == 8) | (y == 9))])
X_train = X[np.where((y == 0) | (y == 1) | (y == 2) | (y == 7) | (y == 8) | (y == 9))]

# Shape of X_train
print("X_train (n_instances, n_features) -> ", X_train.shape)
# Shape of y_train
print("y_train (n_instances,) -> ", y_train.shape)


# Logistic Regression
model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=2000)
model.fit(X_train, y_train)

#print(model.coef_.shape) # returns a matriz of weights (coefficients)
# We have Weight matrix, W z x d
W = model.coef_
print("W shape: ", W.shape)

# We have a attributes, a=4 [pca_d1, pca_d2, lle_d1, lle_d2]
# We have Signature matrix, S a x z
pca = decomposition.PCA(n_components=2)
lle = manifold.LocallyLinearEmbedding(n_components=2, random_state=1)
X_pca = pca.fit_transform(X_train)
X_lle = lle.fit_transform(X_train)

S = []
for i, ys in enumerate(np.unique(y_train)):
	if i == 0:
		S = np.r_[ np.mean(X_pca[y_train == ys], axis=0), np.mean(X_lle[y_train == ys], axis=0) ]
	else:
		S = np.c_[S, np.r_[ np.mean(X_pca[y_train == ys], axis=0), np.mean(X_lle[y_train == ys], axis=0) ]]
print("S shape: ", S.shape)

# From W and S calculate V, d x a
V = np.linalg.lstsq(S.T, W, rcond=None)[0].T
print("V shape", V.shape)


W_new = np.dot(S.T, V.T)
print("W_new shape: ", W_new.shape)

print("%f" % (np.sum(np.sqrt((W_new-W)**2))))

# Predictions that happens in the training phase of zero shot learning
#for ys, x in zip(y_train, X_train):
#	print(np.argmax(np.dot(x.T, W_new)), ys)

# INFERENCE
lbl = preprocessing.LabelEncoder()
y_test = lbl.fit_transform(y[np.where((y == 3) | (y == 4) | (y == 5) | (y == 6))])
X_test = X[np.where((y == 3) | (y == 4) | (y == 5) | (y == 6))]

# create S' (the Signature matrix for the new classes, using the old transformers)
X_test, X_sig, y_test, y_sig = train_test_split(X_test, y_test, test_size=4, random_state=1, stratify=y_test)

X_pca = pca.transform(X_sig)
X_lle = lle.transform(X_sig)

for i, ys in enumerate(np.unique(y_sig)):
	if i == 0:
		S = np.r_[ np.mean(X_pca[y_sig == ys], axis=0), np.mean(X_lle[y_sig == ys], axis=0) ]
	else:
		S = np.c_[S, np.r_[ np.mean(X_pca[y_sig == ys], axis=0), np.mean(X_lle[y_sig == ys], axis=0) ]]

# Calculate the new Weight/Coefficient matrix		
W_new = np.dot(S.T, V.T).T

# Check performance
correct = 0
i = 0
for i, (ys, x) in enumerate(zip(y_test, X_test)):
		if np.argmax(np.dot(x.T, W_new)) == ys:
			correct += 1 
		
print(correct, i, correct / float(i))
