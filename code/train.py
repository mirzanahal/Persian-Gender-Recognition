import argparse
import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler

import data_path

from classifier import Classifier 
from feature_generator import generate_features
               

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Persian Gender Recognition')
    parser.add_argument('--method', default='fasttext', const='fasttext', nargs='?', choices=['fasttext', 'gensim'], help='method for embedding words fasttext or gensim (default: %(default)s)')
    parser.add_argument('--output', type=str, default='', help='output name')
    parser.add_argument('--feature-importance-threshold', type=float, default=0.005, help='Feature importance threshold from random forest model')
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    
    output_path = os.path.join(data_path.DATA_PATH, args.output)
    model_output_path = os.path.join(data_path.MODEL_PATH, 'model.pkl')
    threshold = args.feature_importance_threshold
    method = args.method
    verbose = args.verbose
    
    if verbose:
        print('Generating/Loading Features ...')

    # generate features
    X , y = generate_features(output_path, method)

    if verbose:
        print('Normalizing Features ...')    

    #normalize              
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)  
                
    if verbose:
        print('Import Feature Importance with random forest with threshold={}.'.format(threshold)) 

    #feature importance
    classifier = Classifier('Random Forest', n_estimator=300)
    classifier.fit(X, y)
    feature_importance = classifier.model.feature_importances_                     
    selected_features = feature_importance > threshold            
    X = X.T[selected_features].T 

    selected_feature_path = os.path.join(data_path.DATA_PATH, 'selected_features.npy')
    if verbose:
        print('{} features selected and saved in {}.'.format(len(X.columns), selected_feature_path))
    with open(selected_feature_path, "wb") as fp:
        pickle.dump(X.columns, fp)      

    if verbose:
        print('Fit the model ...')    
    #classification               
    classifier = Classifier('SVM' , kernel='rbf', gamma='scale')
    classifier.fit(X, y)     

    if verbose:
        print('Saving model in {}.'.format(model_output_path))          
    #save model
    pickle.dump(classifier, open(model_output_path, 'wb'))         
                
                
                