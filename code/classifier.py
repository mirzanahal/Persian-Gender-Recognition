import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate

import utils

from config import MODEL_SCORE_METRICS




def create_random_forest_model(**kwargs):
    if 'n_estimators' in kwargs.keys():
        n_estimators = kwargs['n_estimators']
    else:
        n_estimators = 100
    if 'max_depth' in kwargs.keys():
        max_depth = kwargs['max_depth']
    else:
        max_depth = 3
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)


def create_svc_model(**kwargs):
    if 'kernel' in kwargs.keys():
        kernel = kwargs['kernel']
    else:
        kernel = 'rbf'
    if 'gamma' in kwargs.keys():
        gamma = kwargs['gamma']
    else:
        gamma = 'auto'
    return SVC(kernel=kernel, gamma=gamma)


def create_naive_bayes_model(**kwargs):
    if 'var_smoothing' in kwargs.keys():
        var_smoothing = kwargs['var_smoothing']
    else:
        var_smoothing = 1e-9
    return GaussianNB(var_smoothing=var_smoothing)


def create_adaboost_model(**kwargs):
    if 'n_estimators' in kwargs.keys():
        n_estimators = kwargs['n_estimators']
    else:
        n_estimators = 50
    if 'learning_rate' in kwargs.keys():
        learning_rate = kwargs['learning_rate']
    else:
        learning_rate = 1
    if 'random_state' in kwargs.keys():
        random_state = kwargs['random_state']
    else:
        random_state = 0
    return AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)


def create_mlp_model(**kwargs):
    if 'activation' in kwargs.keys():
        activation = kwargs['activation']
    else:
        activation = 'relu'
    if 'solver' in kwargs.keys():
        solver = kwargs['solver']
    else:
        solver = 'adam'
    if 'random_state' in kwargs.keys():
        random_state = kwargs['random_state']
    else:
        random_state = 1
    if 'max_iter' in kwargs.keys():
        max_iter = kwargs['max_iter']
    else:
        max_iter = 100
    if 'hidden_layer_sizes' in kwargs.keys():
        hidden_layer_sizes = kwargs['hidden_layer_sizes']
    else:
        hidden_layer_sizes = (100,)
    return MLPClassifier(activation=activation, solver=solver, random_state=random_state, max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)


class Classifier:
    def __init__(self, model, **kwargs):
        if model == 'Random Forest':
            self.model = create_random_forest_model(**kwargs)
        elif model == 'SVM':
            self.model = create_svc_model(**kwargs)
        elif model == 'Naive Bayes':
            self.model = create_naive_bayes_model(**kwargs)
        elif model == 'Ada Boost':
            self.model = create_adaboost_model(**kwargs)
        elif model == 'MLP':
            self.model = create_mlp_model(**kwargs)
        else:
            self.model = None
            print('Model should be selected from: Random Forest, SVM, Naive Bayes, Ada Boost, MLP')

    def fit(self, X, y):
        self.model.fit(X, y)
        return self.test(X)

    def test(self, X):
        return self.model.predict_proba(X)

    def train(self, X, y, n_folds=10):
        cv = ShuffleSplit(n_splits=n_folds, test_size=0.1, random_state=0)
        scores = cross_validate(self.model, X, y, scoring=MODEL_SCORE_METRICS, cv=cv)
        utils.report(scores)
        
        


