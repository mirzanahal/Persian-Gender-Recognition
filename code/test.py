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
    parser.add_argument('--output', type=str, default='test', help='output name')

    args = parser.parse_args()
    

    model_path = os.path.join(data_path.MODEL_PATH, args.model+'.pkl')
    path = args.output  
    threshold = args.threshold  
    method = args.method
    # generate features
    X , y = generate_features(path, method)
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)  
    
    with open(os.path.join(data_path.DATA_PATH, 'selected_features.npy'), "rb") as fp:
        selected_featuers = pickle.load(fp) 

    X = X[selected_featuers]
    # load the model
    classifier = pickle.load(open(model_path, 'rb'))
    test_predictions_svm = classifier.predict(X)
    test_prediction_labels_svm = np.array(test_predictions_svm[:,1] > threshold, dtype=int)
    print(accuracy_score(y, test_prediction_labels_svm))
    
