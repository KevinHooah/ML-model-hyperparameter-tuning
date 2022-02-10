# Common machine learning classification models' hyperparameter tuning

This repo is for a collection of hyper-parameter tuning for "common" machine learning classification models, including:
* Linear SVM (Grid Search),
* RBF-Kernel SVM (Grid Search),
* Radom Forest (Bayesian Optimization),
* XG Boost(Bayesian Optimization),
* Logistic Regression (Grid Search),
* k-Nearest Neighbors (Grid Search),
* Extra Trees (Bayesian Optimization),
* Gaussian Process Classifier (Grid Search),

All hyper-parameters' searching space are set by empirical knowledge. You may play with it on your own.
For the use of the tuning code, you can refer to the toy example in ***example.py***.

If you find this tool is usefull, we will be glad if you can cite us in your paper :-)
> AutoQual: task-oriented structural vibration sensing quality assessment leveraging co-located mobile sensing context  (https://link.springer.com/article/10.1007/s42486-021-00073-3)

Recommended Packages:
* Python                    3.6+
* Numpy                     1.19.5
* scikit-learn              1.0.1
* xgboost                   1.5.1

If you are using an Intel chip, you may need this to accelerate the computing:
* scikit-learn-intelex      2021.2.2

If you want to use the Bayesian Optimization, you need install this package:
* hyperopt                  0.2.7
