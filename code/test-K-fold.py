import pickle
import os
import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import data_path

from classifier import Classifier
from feature_generator import generate_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test based on k-fold cross validation')
    parser.add_argument('--method', default='fasttext', const='fasttext', nargs='?', choices=['fasttext', 'gensim'], help='method for embedding words fasttext or gensim (default: %(default)s)')
    parser.add_argument('--k', type=str, default=10, help='number of folds')
    parser.add_argument('--model', type=str, default='../model/model.pkl', help='model address')
    parser.add_argument('--output', type=str, default='train', help='output name')
    args = parser.parse_args()
    
    k = args.k
    model_path = args.model
    method = args.method
    
    
    path = 'train'
    
    # generate features
    X , y = generate_features(path, method)
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)  
    
    with open(os.path.join(data_path.DATA_PATH, 'selected_features.npy'), "rb") as fp:
        selected_featuers = pickle.load(fp)    
    
    X = X[selected_featuers]
    
    # load the model
    classifier = pickle.load(open(model_path, 'rb'))
    classifier.train(X, y, n_folds=k)
    