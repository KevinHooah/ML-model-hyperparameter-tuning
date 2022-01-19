'''
All the model tuning codes are here.
'''
import warnings
warnings.filterwarnings("ignore")
from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def perform_metric(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    # print('TPR: ',TPR)
    return TPR

def svmTuning(xTr, yTr, xTe, yTe, clflsvm, clfrsvm, cvMethod):
    '''
    Input:
        Tuning the SVM classifiers with given train and test data.
        Tr: train, Te: test
        clf- classifier
        lsvm: linear svm
        rsvm: rbf-kernal svm
        cvMethod: The defined cross-validation object, it can be either a number (of folds) or other sklearn methods.

    Output:
        (tuned linear-svm model, tuned rbf-svm model, tuned linear-svm test acccuracy, tuned rbf-svm test accuracy)

    '''
    gp_svm = {'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10, 'scale']}
    gd_sr_lsvm = GridSearchCV(estimator=clflsvm,
                        param_grid=gp_svm,
                        scoring='accuracy',
                        cv=cvMethod,
                        n_jobs=-1,
                        refit=True,
                        verbose=0)
    gd_sr_rsvm = GridSearchCV(estimator=clfrsvm,
                        param_grid=gp_svm,
                        scoring='accuracy',
                        cv=cvMethod,
                        n_jobs=-1,
                        refit=True,
                        verbose=0)
    gd_sr_lsvm.fit(xTr, yTr)
    gd_sr_rsvm.fit(xTr, yTr)

    best_lsvm = gd_sr_lsvm.best_estimator_
    best_rsvm = gd_sr_rsvm.best_estimator_

    lsvm_pred = best_lsvm.predict(xTe)
    rsvm_pred = best_rsvm.predict(xTe)
    return (best_rsvm, best_rsvm, best_lsvm.score(xTe,yTe), best_rsvm.score(xTe,yTe))
    


def rfTuning(xTr, yTr, xTe, yTe, clfrf, cvMethod):
    '''
    Tuning random forest with bayes optimization.
    Input:
        Almost the same as svm tuning.
    Output:
        tuned random forest model, tuned random forest test accuracy
    '''
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 3, 4]
    bootstrap = [True, False]
    criterion = ["gini", "entropy"]

    param_space = {
    'max_depth': hp.choice('max_depth', max_depth),
    'max_features': hp.choice('max_features', max_features),
    'n_estimators': hp.choice('n_estimators', n_estimators),
    'criterion': hp.choice('criterion', criterion),
    'min_samples_split': hp.choice('min_samples_split', min_samples_split),
    'min_samples_leaf': hp.choice('min_samples_leaf', min_samples_leaf),
    'bootstrap': hp.choice('bootstrap', bootstrap)}

    def acc_model(params):
        acc_score = cross_val_score(clfrf, xTr, yTr, cv=cvMethod)
        return acc_score.mean()
    
    def f(params):
        acc = acc_model(params)
        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(f, param_space, algo=tpe.suggest, max_evals=100, trials=trials) # you may change the interation number
    #Get the index from space
    bstp = int(best['bootstrap'])
    cri = int(best['criterion'])
    m_dep = int(best['max_depth'])
    m_fea = int(best['max_features'])
    m_lef = int(best['min_samples_leaf'])
    m_sp = int(best['min_samples_split'])
    n_est = int(best['n_estimators'])

    best_rf = RandomForestClassifier(n_estimators=n_estimators[n_est],bootstrap=bootstrap[bstp], criterion=criterion[cri],  
                                    max_depth=max_depth[m_dep], max_features=max_features[m_fea], min_samples_leaf=min_samples_leaf[m_lef], 
                                    min_samples_split=min_samples_split[m_sp])

    best_rf.fit(xTr, yTr)
    return (best_rf, best_rf.score(xTe,yTe))

def lrTuning(xTr, yTr, xTe, yTe, clflr, cvMethod):
    '''
    This is for multi-class logistic regression.
    '''
    gp_lr={"C":[0.001,0.01,0.1,1,10,100], "multi_class":["auto", "ovr", "multinomial"], "max_iter":[100, 200, 300, 400, 500]}
    gd_sr_lr = GridSearchCV(estimator=clflr,
	                    param_grid=gp_lr,
	                    scoring='accuracy',
	                    cv=cvMethod,
	                    n_jobs=-1,
	                    refit=True,
	                    verbose=0)
    gd_sr_lr.fit(xTr, yTr)
    best_lr = gd_sr_lr.best_estimator_
    return (best_lr, best_lr.score(xTe,yTe))


def xgTuning(xTr, yTr, xTe, yTe, clfxgb, cvMethod):
    '''
    Tuning xg-boost with bayes optimization.
    Input:
        Almost the same as svm tuning.
    Output:
        tuned xg-boost model, tuned xg-boost test accuracy
    '''
    max_depth = [int(x) for x in range (2, 10, 1)]
    n_estimators = [int(x) for x in range(20, 200, 10)]
    learning_rate = [0.1, 0.01, 0.05]
    min_child_weight = [int(x) for x in range (1, 10, 1)]
    gamma = [0.5, 1, 1.5, 2, 5]
    colsample_bytree = [0.1, 0.5, 0.8, 1]

    gp_xg = { 'max_depth': hp.choice('max_depth', max_depth),
            'n_estimators': hp.choice('n_estimators', n_estimators),
            'learning_rate': hp.choice('learning_rate', learning_rate),
            'min_child_weight': hp.choice('min_child_weight', min_child_weight),
            'gamma': hp.choice('gamma', gamma),
            'colsample_bytree': hp.choice('colsample_bytree', colsample_bytree)}

    def acc_model(params):
        acc_score = cross_val_score(clfxgb, xTr, yTr, cv=cvMethod)
        return acc_score.mean()

    def f(params):
        acc = acc_model(params)
        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(f, gp_xg, algo=tpe.suggest, max_evals=100, trials=trials) # you may change the iteration number
    #Get the index from space
    n_est = int(best['n_estimators'])
    m_dep = int(best['max_depth'])
    lr = int(best['learning_rate'])
    mcw = int(best['min_child_weight'])
    gm = int(best['gamma'])
    cbt = int(best['colsample_bytree'])

    best_xg = XGBClassifier(n_estimators=n_estimators[n_est], max_depth=max_depth[m_dep], learning_rate=learning_rate[lr],  
                                    min_child_weight=min_child_weight[mcw], gamma=gamma[gm], colsample_bytree=colsample_bytree[cbt],
                                    objective='multi:softmax')
    best_xg.fit(xTr, yTr)
    return (best_xg, best_xg.score(xTe,yTe))

def knnTuning(xTr, yTr, xTe, yTe, clfknn, cvMethod):
    '''
    Tuning knn model.
    Input:
        Almost the same as svm tuning.
    Output:
        tuned knn model, tuned knn test accuracy
    '''
    gp_knn={'n_neighbors':[3, 4, 5, 6, 7, 8, 9, 10, 11], # you may change the space
            'weights':['uniform', 'distance'],
            'algorithm':['auto', 'ball_tree','kd_tree']}

    gd_sr_knn = GridSearchCV(estimator=clfknn,
                        param_grid=gp_knn,
                        scoring='accuracy',
                        cv=cvMethod,
                        n_jobs=-1,
                        refit=True,
                        verbose=0)

    gd_sr_knn.fit(xTr, yTr)
    best_knn = gd_sr_knn.best_estimator_
    return (best_knn, best_knn.score(xTe,yTe))

def etTuning(xTr, yTr, xTe, yTe, clfet, cvMethod):
    '''
    Tuning extra trees with bayes optimization. The setup is almost the same with random forest.
    '''
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 3, 4]
    bootstrap = [True, False]
    criterion = ["gini", "entropy"]

    param_space = {
    'max_depth': hp.choice('max_depth', max_depth),
    'max_features': hp.choice('max_features', max_features),
    'n_estimators': hp.choice('n_estimators', n_estimators),
    'criterion': hp.choice('criterion', criterion),
    'min_samples_split': hp.choice('min_samples_split', min_samples_split),
    'min_samples_leaf': hp.choice('min_samples_leaf', min_samples_leaf),
    'bootstrap': hp.choice('bootstrap', bootstrap)}

    def acc_model(params):
        acc_score = cross_val_score(clfet, xTr, yTr, cv=cvMethod)
        return acc_score.mean()
    
    def f(params):
        acc = acc_model(params)
        return {'loss': -acc, 'status': STATUS_OK}

    
    trials = Trials()
    best = fmin(f, param_space, algo=tpe.suggest, max_evals=60, trials=trials) #Get the index from space

    # indexes
    bstp = int(best['bootstrap'])
    cri = int(best['criterion'])
    m_dep = int(best['max_depth'])
    m_fea = int(best['max_features'])
    m_lef = int(best['min_samples_leaf'])
    m_sp = int(best['min_samples_split'])
    n_est = int(best['n_estimators'])

    best_et = RandomForestClassifier(n_estimators=n_estimators[n_est],bootstrap=bootstrap[bstp], criterion=criterion[cri],  
                                    max_depth=max_depth[m_dep], max_features=max_features[m_fea], min_samples_leaf=min_samples_leaf[m_lef], 
                                    min_samples_split=min_samples_split[m_sp])

    best_et.fit(xTr, yTr)
    return (best_et, best_et.score(xTe,yTe))
