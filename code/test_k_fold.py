import pickle
import os
import argparse
import pandas as pd

from sklearn.preprocessing import StandardScaler

import data_path

from classifier import Classifier
from feature_generator import generate_features



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test based on k-fold cross validation')
    parser.add_argument('--method', default='fasttext', const='fasttext', nargs='?', choices=['fasttext', 'gensim'], help='method for embedding words fasttext or gensim (default: %(default)s)')
    parser.add_argument('--k', type=str, default=10, help='number of folds')
    parser.add_argument('--model', type=str, default='model.pkl', help='model')
    parser.add_argument('--output', type=str, default='', help='output name')
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    
    k = args.k
    model_path = os.path.join(data_path.MODEL_PATH, args.model)
    method = args.method
    verbose = args.verbose
    
    
    output_path = os.path.join(data_path.DATA_PATH, args.output)
    
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

    if verbose:
        print('{} features selected and saved in {}.'.format(len(X.columns), selected_feature_path))
    
    X = X[selected_featuers]
    
    if verbose:
        print('K fold results:')

    # load the model
    classifier = pickle.load(open(model_path, 'rb'))
    classifier.train(X, y, n_folds=k)
    