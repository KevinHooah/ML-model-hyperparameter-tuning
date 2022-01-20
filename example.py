from tuning import svmTuning
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles
from sklearn.svm import SVC

X, y = make_gaussian_quantiles(n_features=20, n_classes=3, n_samples = 1000) # make a toy dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # split train and test

# define classifers
clf_lsvm = SVC(kernel = 'linear')
clf_rsvm = SVC(kernel = 'rbf')

# tune and print result
(tuned_lsvm, tuned_rsvm, lsvm_acc, rsvm_acc) = svmTuning(X_train, y_train, X_test, y_test, clf_lsvm, clf_rsvm, 3)
print(lsvm_acc, rsvm_acc)