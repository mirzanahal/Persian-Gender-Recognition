import os
import argparse
import pandas as pd
import hazm

from tqdm import tqdm

import data_loader
import data_path
import utils
import config



def get_grammatical_features(dataset):
    features = []
    for key, text in tqdm(dataset.items()):
        feature = {}
        normalized_text = utils.normalize_text(text)
        tagger = hazm.POSTagger(model=data_path.POSTAGGER_MODEL_PATH)
        tags = tagger.tag(hazm.word_tokenize(normalized_text))
        tags_list = [i[1] for i in tags]

        sounds = [utils.normalize_text(sound) for sound in config.sounds]
        group_pros = [utils.normalize_text(group_pro) for group_pro in config.group_pro]
        conjunctions = [utils.normalize_text(conjunction) for conjunction in config.conjunctions]
        subjective_pronounces = [utils.normalize_text(subjective_pronounce) for subjective_pronounce in config.subjective_pronounce]

        feature['GRM_F30'] = utils.count_chars(text, subjective_pronounces)

        feature['GRM_F31'] = utils.count_chars(text, config.question)
 
        feature['GRM_F32'] = utils.count_chars(text, conjunctions)

        feature['GRM_F33'] = utils.count_chars(text, group_pros)

        feature['GRM_F34'] = utils.count_chars(text, sounds)

        feature['GRM_F35'] = tags_list.count('P') + tags_list.count('POSTP')

        feature['GRM_F40'] = tags_list.count('AJ')
        feature['GRM_F41'] = tags_list.count('ADV')
        feature['GRM_F42'] = tags_list.count('PRO')
        feature['GRM_F51'] = tags_list.count('NUM')

        feature['number'] = key

        features.append(feature)
    return pd.DataFrame(features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Grammatical Features')
    parser.add_argument('--data', type=str, default='../data/train/', help='path to dataset')
    parser.add_argument('--output', type=str, default='', help='output name')
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    data = args.data
    output_path = os.path.join(data_path.DATA_PATH, '{}grammatical_features.csv'.format(args.output))
    verbose = args.verbose

    female_dataset = data_loader.load_dataset(data_path.FEMALE_DATA_PATH)
    male_dataset = data_loader.load_dataset(data_path.MALE_DATA_PATH)

    if verbose:
        print('Genrate female Grammatical features ...')

    female_features_pdf = get_grammatical_features(female_dataset)
    female_features_pdf['label'] = config.FEMALE_LABEL
    female_features_pdf.to_csv('../data/female_grammatical_features.csv')

    if verbose:
        print('Genrate male Grammatical features ...')

    male_features_pdf = get_grammatical_features(male_dataset)
    male_features_pdf['label'] = config.MALE_LABEL
    male_features_pdf.to_csv('../data/male_grammatical_features.csv')

    features_pdf = pd.concat([female_features_pdf, male_features_pdf], axis=0)

    if verbose:
        print('Grammatical Features for {} sentences: {} male, {} female calculated.'.format(len(features_pdf), len(female_features_pdf), len(male_features_pdf)))

    features_pdf.to_csv(output_path)

    if verbose:
        print('Two dataset concatenate and saved in {} with shape {}.'.format(output_path, features_pdf.shape))
    