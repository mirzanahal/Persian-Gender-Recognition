import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

import utils

from config import MODEL_SCORE_METRICS




def create_random_forest_model(**kwargs):
    return RandomForestClassifier(**kwargs)

def create_svc_model(**kwargs):
    return SVC(**kwargs)

def create_naive_bayes_model(**kwargs):
    return GaussianNB(**kwargs)

def create_adaboost_model(**kwargs):
    return AdaBoostClassifier(**kwargs)

def create_mlp_model(**kwargs):
    return MLPClassifier(**kwargs)

def create_lda_model(**kwargs):
    return LinearDiscriminantAnalysis(**kwargs)

def create_qda_model(**kwargs):
    return QuadraticDiscriminantAnalysis(**kwargs)

def create_knn_model(**kwargs):
    return KNeighborsClassifier(**kwargs)


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
        elif model == 'LDA':
            self.model = create_lda_model(**kwargs)
        elif model == 'QDA':
            self.model = create_qda_model(**kwargs)
        elif model == 'KNN':
            self.model = create_knn_model(**kwargs)
        else:
            self.model = None
            print('Model should be selected from: Random Forest, SVM, Naive Bayes, AdaBoost, MLP, LDA, QDA, KNN')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)

    def train(self, X, y, n_folds=10):
        cv = ShuffleSplit(n_splits=n_folds, test_size=0.1, random_state=0)
        scores = cross_validate(self.model, X, y, scoring=MODEL_SCORE_METRICS, cv=cv)
        utils.report(scores)
        self.fit(X, y)

    def predict_label(self, X):
        return self.model.predict(X)
        
        


