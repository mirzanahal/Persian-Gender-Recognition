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



def get_word_dependent_features(dataset):
    features = []
    for key, text in tqdm(dataset.items()):
        feature = {}
        normalized_text = utils.normalize_text(text)
        words, _ = utils.tokenize_word_and_sentence(normalized_text)

        C = len(normalized_text)
        N = len(words)

        # #ویژگی های مبتنی بر واژه 
        # تعداد کل کلمات
        feature['WD_F7'] = N
        #میانگین تعداد نویسه در هر کلمه
        feature['WD_F8'] = C/N
        # غنای واژگانی (کل کلمات یکتا تقسیم بر تعداد کل کلمه ها)
        V = set(words)
        feature['WD_F9'] = len(V)/N
        #N/ کلمات طولانی (بزرگ تر از 3 نویسه)
        long_words = [w for w in words if len(w)>=3]
        feature['WD_F10'] = len(long_words)/N
        #N/کلمات کوچک تر از 2 نویسه
        short_words = [w for w in words if len(w)<=2]
        feature['WD_F11'] = len(short_words)/N
        #N/کلمات 1 تکراره
        counts = Counter(words)
        unique_words = [w for w in words if counts[w]==1]
        feature['WD_F12'] = len(unique_words)
        #N/کلمات 2 تکراره
        double_words = [w for w in words if counts[w]==2]
        feature['WD_F13'] = len(double_words)
        # معیار k یول
        yules_k = 10000*(-1*(1.0/N) + sum(list([(len(list(w for w in V if counts[w]==i)))*((i/N)**2) for i in range(1,len(V)+1)])))
        feature['WD_F14'] =yules_k
        # معیار D سیمپسون
        simpsons_d = sum((len(list(w for w in V if counts[w]==i)))*(i/N)*((i-1)/(N-1)) for i in range(1,len(V)))
        feature['WD_F15'] = simpsons_d
        # معیار S سیشل 
        sichels_s = len(double_words)/len(V)
        feature['WD_F16'] = sichels_s
        # معیار R هونور
        delimiter = 1 - len(unique_words)/len(V)
        if delimiter == 0:
            delimiter = 0.0001
        honores_R = (100 * math.log(N))/(delimiter)
        feature['WD_F17'] = honores_R
        #معیار انتروپی
        entorpy = sum((len(list(w for w in V if counts[w]==i)))*(i/N)*(-1*math.log(i/N)) for i in range(1,len(V)))
        feature['WD_F50'] = entorpy

        feature['number'] = key

        features.append(feature)
    return pd.DataFrame(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Word Dependent Features')
    parser.add_argument('--output', type=str, default='', help='output path')
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    output_path = os.path.join(data_path.DATA_PATH, args.output, 'word_dependent_features.csv')
    verbose = args.verbose

    female_data_path = os.path.join(data_path.DATA_PATH, args.output, data_path.FEMALE_DATA_PATH)
    male_data_path = os.path.join(data_path.DATA_PATH, args.output, data_path.MALE_DATA_PATH)

    female_dataset = data_loader.load_dataset(female_data_path)
    male_dataset = data_loader.load_dataset(male_data_path)

    if verbose:
        print('Genrate female word dependent features ...')

    female_features_pdf = get_word_dependent_features(female_dataset)
    female_features_pdf['label'] = config.FEMALE_LABEL

    if verbose:
        print('Genrate male word dependent features ...')

    male_features_pdf = get_word_dependent_features(male_dataset)
    male_features_pdf['label'] = config.MALE_LABEL

    features_pdf = pd.concat([female_features_pdf, male_features_pdf], axis=0)

    if verbose:
        print('Word Dependent Features for {} sentences: {} male, {} female calculated.'.format(len(features_pdf), len(female_features_pdf), len(male_features_pdf)))

    features_pdf.to_csv(output_path)
    if verbose:
        print('Two dataset concatenate and saved in {} with shape {}.'.format(output_path, features_pdf.shape))
    