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



def get_structural_features(dataset):
    features = []
    for key, text in tqdm(dataset.items()):
        feature = {}
        normalized_text = utils.normalize_text(text)
        words, sentences = utils.tokenize_word_and_sentence(normalized_text)
        N = len(words)
        S = len(sentences)

        # # ویژگی های ساختاری
        # # تعداد کل خط
        feature['STR_F25'] = text.count('\n')
        # تعداد کل جملات = S
        feature['STR_F26'] = S
        # تعداد کلمه در هر جمله (میانگین)
        feature['STR_F27'] = N/S

        empty_lines = text.replace(" ", "").count('\n\n')
        # کل خطوط / تعداد خطوط خالی
        total_lines = text.replace(" " , "").count('\n') + 1
        feature['STR_F28'] = empty_lines/total_lines
        # میانگین خطوط غیر خالی
        feature['STR_F29'] = len(text)/(total_lines-empty_lines)

        feature['number'] = key

        features.append(feature)
    return pd.DataFrame(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Structural Features')
    parser.add_argument('--output', type=str, default='', help='output path')
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    output_path = os.path.join(args.output, 'structural_features.csv')
    verbose = args.verbose

    female_data_path = os.path.join(args.output, data_path.FEMALE_DATA_PATH)
    male_data_path = os.path.join(args.output, data_path.MALE_DATA_PATH)

    female_dataset = data_loader.load_dataset(female_data_path)
    male_dataset = data_loader.load_dataset(male_data_path)

    if verbose:
        print('Genrate female Structural features ...')

    female_features_pdf = get_structural_features(female_dataset)
    female_features_pdf['label'] = config.FEMALE_LABEL

    if verbose:
        print('Genrate male Structural features ...')

    male_features_pdf = get_structural_features(male_dataset)
    male_features_pdf['label'] = config.MALE_LABEL

    features_pdf = pd.concat([female_features_pdf, male_features_pdf], axis=0)

    if verbose:
        print('Structural Features for {} sentences: {} male, {} female calculated.'.format(len(features_pdf), len(female_features_pdf), len(male_features_pdf)))

    features_pdf.to_csv(output_path)
    if verbose:
        print('Two dataset concatenate and saved in {} with shape {}.'.format(output_path, features_pdf.shape))
    