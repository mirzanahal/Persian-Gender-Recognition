import os
import argparse
import pandas as pd
import math

from collections import Counter
from tqdm import tqdm

import data_loader
import data_path
import utils
import config



def get_syntactics_features(dataset):
    features = []
    for key, text in tqdm(dataset.items()):
        feature = {}
        normalized_text = utils.normalize_text(text)
        words, _ = utils.tokenize_word_and_sentence(normalized_text)

        C = len(normalized_text)
        # # ویژگی های نحوی
        # C/تعداد کاما
        feature['SYN_F18'] = text.count('،')/C
        # C/تعداد نقطه
        feature['SYN_F19'] = text.count('.')/C
        # C/تعداد دو نقطه
        feature['SYN_F20'] = text.count(':')/C
        # C/تعداد سمیکلون
        feature['SYN_F21'] = text.count(';')/C
        # C/تعداد علامت سوال
        feature['SYN_F22'] = text.count('؟')/C
        # C/تعداد علامت تعجب
        feature['SYN_F23'] = text.count('!')/C
        # C/تعداد علامت تعجب سه تایی
        feature['SYN_F24'] = text.count('!!!')/C

        feature['number'] = key

        features.append(feature)
    return pd.DataFrame(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Syntactics Features')
    parser.add_argument('--output', type=str, default='', help='output path')
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    output_path = os.path.join(data_path.DATA_PATH, args.output, 'syntactics_features.csv')
    verbose = args.verbose

    female_dataset = data_loader.load_dataset(data_path.FEMALE_DATA_PATH)
    male_dataset = data_loader.load_dataset(data_path.MALE_DATA_PATH)

    if verbose:
        print('Genrate female Syntactics features ...')

    female_features_pdf = get_syntactics_features(female_dataset)
    female_features_pdf['label'] = config.FEMALE_LABEL

    if verbose:
        print('Genrate male Syntactics features ...')

    male_features_pdf = get_syntactics_features(male_dataset)
    male_features_pdf['label'] = config.MALE_LABEL

    features_pdf = pd.concat([female_features_pdf, male_features_pdf], axis=0)

    if verbose:
        print('Syntactics Features for {} sentences: {} male, {} female calculated.'.format(len(features_pdf), len(female_features_pdf), len(male_features_pdf)))

    features_pdf.to_csv(output_path)
    if verbose:
        print('Two dataset concatenate and saved in {} with shape {}.'.format(output_path, features_pdf.shape))
    