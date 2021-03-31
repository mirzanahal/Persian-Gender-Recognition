import argparse
import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from os.path import isfile

import config
import utils
import data_path

from classifier import Classifier              
               

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Persian Gender Recognition')
    parser.add_argument('--method', default='gensim', const='gensim', nargs='?', choices=['fasttext', 'gensim'], help='method for embedding words fasttext or gensim (default: %(default)s)')
    parser.add_argument('--output', type=str, default='', help='output name')
    parser.add_argument('--feature-importance-threshold', type=float, default=0.01, help='Feature importance threshold from random forest model')
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    
    embedding_method = args.method
    output = args.output


    #check for positive and negetive words paths
    if (not isfile(data_path.POSITIVE_WORDS_PATH)) or (not isfile(data_path.NEGATIVE_WORDS_PATH)):
            utils.import_positive_negative_words_from(
                path = data_path.EMOTION_LEXICON_WORDS_PATH,
                positive_words_path = data_path.POSITIVE_WORDS_PATH,
                negative_words_path = data_path.NEGATIVE_WORDS_PATH
            )

    #generate features
    #embedding
    if (not isfile(os.path.join(data_path.DATA_PATH, 'embedding_{}.csv'.embedding_method))):
        os.system('python generate_embedding.py --verbose')
    embedding_features = pd.read_csv(os.path.join(data_path.DATA_PATH, 'embedded_texts_gensim.csv'), index_col=0)

    #psychological_features               
    if (not isfile(os.path.join(data_path.DATA_PATH, 'psychological_features.csv'))):
        os.system('python generate_psychological_features.py --verbose')
    psychological_features = pd.read_csv(os.path.join(data_path.DATA_PATH, 'psychological_features.csv'), index_col=0)               

    #structural_features
    if (not isfile(os.path.join(data_path.DATA_PATH, 'structural_features.csv'))):
        os.system('python generate_structural_features.py --verbose')
    structural_features = pd.read_csv(os.path.join(data_path.DATA_PATH, 'structural_features.csv'), index_col=0)               
                
    #syntactics_features
    if (not isfile(os.path.join(data_path.DATA_PATH, 'syntactics_features.csv'))):
        os.system('python generate_syntactics_features.py --verbose')
    syntactics_features = pd.read_csv(os.path.join(data_path.DATA_PATH, 'syntactics_features.csv'), index_col=0)                   
                
    #text_dependent_features
    if (not isfile(os.path.join(data_path.DATA_PATH, 'text_dependent_features.csv'))):
        os.system('python generate_text_dependent_features.py --verbose')
    text_dependent_features = pd.read_csv(os.path.join(data_path.DATA_PATH, 'text_dependent_features.csv'), index_col=0)               
                
    #word_dependent_features
    if (not isfile(os.path.join(data_path.DATA_PATH, 'word_dependent_features.csv'))):
        os.system('python generate_word_dependent_features.py --verbose')
    word_dependent_features = pd.read_csv(os.path.join(data_path.DATA_PATH, 'word_dependent_features.csv'), index_col=0)                   
                
    features_list = [embedding_features, psychological_features, structural_features, syntactics_features, text_dependent_features, word_dependent_features]

    features = features_list.pop()
    for feature in features_list:
        features = pd.merge(features, feature, left_on=['label', 'number'], right_on=['label', 'number'])
                
                
    #split X , y
    y = features['label']
    X = features.drop(['label', 'number'], axis=1)
                
    #normalize              
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)  
                
                
    #feature importance
    classifier = Classifier('Random Forest', n_estimator=300)
    classifier.fit(X, y)
    feature_importance = classifier.model.feature_importances_              
                
    selected_features = feature_importance > 0.01               
    X = X.T[selected_features].T              
                
    #classification               
    classifier = Classifier('SVM' , kernel='rbf', gamma='scale')
    classifier.fit(X, y)               
                
    #save model
    pickle.dump(classifier, open('../model/model.pkl', 'wb'))         
                
                
                