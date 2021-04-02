import pandas as pd
from os import system
from os.path import exists, join

import data_path
import utils
import config



def generate_features(path, method):
    if (not exists(data_path.POSITIVE_WORDS_PATH)) or (not exists(data_path.NEGATIVE_WORDS_PATH)):
        utils.import_positive_negative_words_from(
            path = data_path.EMOTION_LEXICON_WORDS_PATH,
            positive_words_path = data_path.POSITIVE_WORDS_PATH,
            negative_words_path = data_path.NEGATIVE_WORDS_PATH
        )

    features_list = []

    #embedding_gensim
    if config.FEATURES['embedding']:
        if (not exists(join(path, 'embedding_{}.csv'.format(method)))):
            system('python generate_embedding.py --verbose --output {} --method {}'.format(path, method))
        embedding_features = pd.read_csv(join(path, 'embedding_{}.csv'.format(method)), index_col=0)
        features_list.append(embedding_features)

    #psychological_features               
    if config.FEATURES['psychological'] :
        if (not exists(join(path, 'psychological_features.csv'))):
            system('python generate_psychological_features.py --verbose --output {}'.format(path))
        psychological_features = pd.read_csv(join(path, 'psychological_features.csv'), index_col=0)     
        features_list.append(psychological_features)          

    #structural_features
    if config.FEATURES['structural']:
        if (not exists(join(path, 'structural_features.csv'))):
            system('python generate_structural_features.py --verbose --output {}'.format(path))
        structural_features = pd.read_csv(join(path, 'structural_features.csv'), index_col=0)
        features_list.append(structural_features)         
                
    #syntactics_features
    if config.FEATURES['syntactics']:
        if (not exists(join(path, 'syntactics_features.csv'))):
            system('python generate_syntactics_features.py --verbose --output {}'.format(path))
        syntactics_features = pd.read_csv(join(path, 'syntactics_features.csv'), index_col=0)
        features_list.append(syntactics_features)                   
                
    #text_dependent_features
    if config.FEATURES['text_dependent']:
        if (not exists(join(path, 'text_dependent_features.csv'))):
            system('python generate_text_dependent_features.py --verbose --output {}'.format(path))
        text_dependent_features = pd.read_csv(join(path, 'text_dependent_features.csv'), index_col=0)  
        features_list.append(text_dependent_features)             
                
    #word_dependent_features
    if config.FEATURES['word_dependent']:
        if (not exists(join(path, 'word_dependent_features.csv'))):
            system('python generate_word_dependent_features.py --verbose --output {}'.format(path))
        word_dependent_features = pd.read_csv(join(path, 'word_dependent_features.csv'), index_col=0)      
        features_list.append(word_dependent_features)             

    features = features_list.pop()
    for feature in features_list:
        features = pd.merge(features, feature, left_on=['label', 'number'], right_on=['label', 'number']) 
                
    #split X , y
    y = features['label']
    X = features.drop(['label', 'number'], axis=1)

    return X, y
