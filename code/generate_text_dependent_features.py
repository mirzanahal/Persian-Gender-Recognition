import os
import argparse
import pandas as pd

from collections import Counter
from tqdm import tqdm

import data_loader
import data_path
import utils
import config



def get_text_dependent_features(dataset):
    features = []
    for key, text in tqdm(dataset.items()):
        feature = {}
        normalized_text = utils.normalize_text(text)
        words, _ = utils.tokenize_word_and_sentence(normalized_text)

        alphabets_in_texts = [i for i in Counter(normalized_text) if i in config.alphabet]
        C = len(normalized_text)
        N = len(words)

        feature['TD_F1'] = C
        # تعداد کل حروف الفبا / C
        feature['TD_F2'] = len(alphabets_in_texts)/C
        # # تعداد حروف الفبا
        feature['TD_F49'] = len(alphabets_in_texts)
        # تعداد کل اعداد
        feature['TD_F3'] = utils.count_chars(normalized_text, config.numbers)
        # تعداد نویسه فاصله
        feature['TD_F4'] = text.count(' ')/C
        # تعداد نویسه تب
        feature['TD_F5'] = text.count('\t')/C
        # تعداد نویسه ویژه
        feature['TD_F6'] = utils.count_chars(normalized_text , config.special_chars)/C

        feature['number'] = key

        features.append(feature)
    return pd.DataFrame(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Gensim Word Embedding Features')
    parser.add_argument('--output', type=str, default='', help='output path')
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    output_path = os.path.join(data_path.DATA_PATH, args.output, 'text_dependent_features.csv')
    verbose = args.verbose

    female_data_path = os.path.join(data_path.DATA_PATH, args.output, data_path.FEMALE_DATA_PATH)
    male_data_path = os.path.join(data_path.DATA_PATH, args.output, data_path.MALE_DATA_PATH)

    female_dataset = data_loader.load_dataset(female_data_path)
    male_dataset = data_loader.load_dataset(male_data_path)

    if verbose:
        print('Genrate female text dependent features ...')

    female_features_pdf = get_text_dependent_features(female_dataset)
    female_features_pdf['label'] = config.FEMALE_LABEL

    if verbose:
        print('Genrate male text dependent features ...')

    male_features_pdf = get_text_dependent_features(male_dataset)
    male_features_pdf['label'] = config.MALE_LABEL

    features_pdf = pd.concat([female_features_pdf, male_features_pdf], axis=0)

    if verbose:
        print('Text Dependent Features for {} sentences: {} male, {} female calculated.'.format(len(features_pdf), len(female_features_pdf), len(male_features_pdf)))

    features_pdf.to_csv(output_path)
    if verbose:
        print('Two dataset concatenate and saved in {} with shape {}.'.format(output_path, features_pdf.shape))
    