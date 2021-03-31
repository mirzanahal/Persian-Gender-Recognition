import pandas as pd
from os import system
from os.path import isfile, join

import data_path
import utils



def generate_features():
    if (not isfile(data_path.POSITIVE_WORDS_PATH)) or (not isfile(data_path.NEGATIVE_WORDS_PATH)):
        utils.import_positive_negative_words_from(
            path = data_path.EMOTION_LEXICON_WORDS_PATH,
            positive_words_path = data_path.POSITIVE_WORDS_PATH,
            negative_words_path = data_path.NEGATIVE_WORDS_PATH
        )

    #embedding_gensim
    if (not isfile(join(data_path.DATA_PATH, 'embedded_texts_gensim.csv'))):
        system('python generate_embedding_gensim.py --verbose')
    embedding_features = pd.read_csv(join(data_path.DATA_PATH, 'embedded_texts_gensim.csv'), index_col=0)

    #psychological_features               
    if (not isfile(join(data_path.DATA_PATH, 'psychological_features.csv'))):
        system('python generate_psychological_features.py --verbose')
    psychological_features = pd.read_csv(join(data_path.DATA_PATH, 'psychological_features.csv'), index_col=0)               

    #structural_features
    if (not isfile(join(data_path.DATA_PATH, 'structural_features.csv'))):
        system('python generate_structural_features.py --verbose')
    structural_features = pd.read_csv(join(data_path.DATA_PATH, 'structural_features.csv'), index_col=0)               
                
    #syntactics_features
    if (not isfile(join(data_path.DATA_PATH, 'syntactics_features.csv'))):
        system('python generate_syntactics_features.py --verbose')
    syntactics_features = pd.read_csv(join(data_path.DATA_PATH, 'syntactics_features.csv'), index_col=0)                   
                
    #text_dependent_features
    if (not isfile(join(data_path.DATA_PATH, 'text_dependent_features.csv'))):
        system('python generate_text_dependent_features.py --verbose')
    text_dependent_features = pd.read_csv(join(data_path.DATA_PATH, 'text_dependent_features.csv'), index_col=0)               
                
    #word_dependent_features
    if (not isfile(join(data_path.DATA_PATH, 'word_dependent_features.csv'))):
        system('python generate_word_dependent_features.py --verbose')
    word_dependent_features = pd.read_csv(join(data_path.DATA_PATH, 'word_dependent_features.csv'), index_col=0)                   
                
    features_list = [embedding_features, psychological_features, structural_features, syntactics_features, text_dependent_features, word_dependent_features]

    features = features_list.pop()
    for feature in features_list:
        features = pd.merge(features, feature, left_on=['label', 'number'], right_on=['label', 'number'])
                
                
    #split X , y
    y = features['label']
    X = features.drop(['label', 'number'], axis=1)

    return X, y
