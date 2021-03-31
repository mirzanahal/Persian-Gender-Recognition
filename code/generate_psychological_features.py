import os
import argparse
import pandas as pd

from tqdm import tqdm

import data_loader
import data_path
import utils
import config



def get_psychological_features(dataset):
    features = []
    for key, text in tqdm(dataset.items()):
        feature = {}
        normalized_text = utils.normalize_text(text)

        colors = [utils.normalize_text(color) for color in config.colors]
        rakiks = [utils.normalize_text(rakik) for rakik in config.rakik]
        doubt_phrases = [utils.normalize_text(doubt_phrase) for doubt_phrase in config.doubt_phrase]
        certain_phrases = [utils.normalize_text(certain_phrase) for certain_phrase in config.certain_phrase]

         # نشانه های زبانی - روانی
        pos_words, neg_words = data_loader.load_positive_negative_words(
            positive_words_path = data_path.POSITIVE_WORDS_PATH,
            negative_words_path = data_path.NEGATIVE_WORDS_PATH
        )
        # صفات مثبت
        feature['PSY_F36'] = utils.count_chars(normalized_text, pos_words)
        # صفات منفی 
        feature['PSY_F37'] = utils.count_chars(normalized_text, neg_words)
        # رنگ‌ها
        feature['PSY_F38'] = utils.count_chars(normalized_text, colors)
        # کلمات رکیک
        feature['PSY_F39'] = utils.count_chars(normalized_text, rakiks)
        # شک و تردید
        feature['PSY_F47'] = utils.count_chars(normalized_text, doubt_phrases)
        # قطعیت
        feature['PSY_F48'] = utils.count_chars(normalized_text, certain_phrases)

        feature['number'] = key

        features.append(feature)
    return pd.DataFrame(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Psychological Features')
    parser.add_argument('--data', type=str, default='../data/train/', help='path to dataset')
    parser.add_argument('--output', type=str, default='', help='output name')
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    data = args.data
    output_path = os.path.join(data_path.DATA_PATH, '{}psychological_features.csv'.format(args.output))
    verbose = args.verbose

    female_dataset = data_loader.load_dataset(data_path.FEMALE_DATA_PATH)
    male_dataset = data_loader.load_dataset(data_path.MALE_DATA_PATH)

    if verbose:
        print('Genrate female Psychological features ...')

    female_features_pdf = get_psychological_features(female_dataset)
    female_features_pdf['label'] = config.FEMALE_LABEL

    if verbose:
        print('Genrate male Psychological features ...')

    male_features_pdf = get_psychological_features(male_dataset)
    male_features_pdf['label'] = config.MALE_LABEL

    features_pdf = pd.concat([female_features_pdf, male_features_pdf], axis=0)

    if verbose:
        print('Psychological Features for {} sentences: {} male, {} female calculated.'.format(len(features_pdf), len(female_features_pdf), len(male_features_pdf)))

    features_pdf.to_csv(output_path)
    if verbose:
        print('Two dataset concatenate and saved in {} with shape {}.'.format(output_path, features_pdf.shape))
    