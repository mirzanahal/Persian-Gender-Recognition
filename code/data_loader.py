import os
import numpy as np

import utils


def load_dataset(data_path):
    dataset = {}
    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        text_file = open(filepath, 'r')
        text = text_file.read()
        dataset[utils.import_name_from(filename)] = text
        text_file.close()
    return dataset


def load_positive_negative_words(positive_words_path, negative_words_path):
    pos_data_arr = np.load(positive_words_path)
    neg_data_arr = np.load(negative_words_path)
    return pos_data_arr, neg_data_arr