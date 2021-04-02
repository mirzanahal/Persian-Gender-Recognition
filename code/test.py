import pickle
import os
import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from os.path import isfile
from sklearn.metrics import accuracy_score

import data_path
from feature_generator import generate_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test report')
    parser.add_argument('--method', default='fasttext', const='fasttext', nargs='?', choices=['fasttext', 'gensim'], help='method for embedding words fasttext or gensim (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model', help='model name')
    parser.add_argument('--threshold', type=float, default=0.5, help='model probability threshold')
    parser.add_argument('--output', type=str, default='', help='output name')
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    model_path = os.path.join(data_path.MODEL_PATH, args.model+'.pkl')
    output_path = os.path.join(data_path.DATA_PATH, args.output)
    threshold = args.threshold  
    method = args.method
    verbose = args.verbose

    if verbose:
        print('Generating/Loading Features ...')

    # generate features
    X , y = generate_features(output_path, method)
    
    if verbose:
        print('Normalizing Features ...')

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)  
    
    selected_feature_path = os.path.join(data_path.DATA_PATH, 'selected_features.npy')
    with open(selected_feature_path, "rb") as fp:
        selected_featuers = pickle.load(fp) 
    X = X[selected_featuers]

    if verbose:
        print('{} features selected from {}.'.format(len(X.columns), selected_feature_path))

    # load the model
    classifier = pickle.load(open(model_path, 'rb'))
    test_predictions = classifier.predict(X)
    test_prediction_labels = np.array(test_predictions[:,1] > threshold, dtype=int)
    if verbose:
        print('Prediction on test set:')

    print(accuracy_score(y, test_prediction_labels))
    
