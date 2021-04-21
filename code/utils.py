import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import hazm

from itertools import chain
from sklearn.metrics import accuracy_score

import config



def count_chars(s, li):
    counts = [s.count(c) for c in li]
    return sum(counts)


def import_name_from(name):
    return int(name[:-4])


def tokenize_word_by(word, delimiters):
    if len(delimiters) == 0 : return [word]
    delimiter = delimiters.pop()
    if delimiter in word:
        splitted_words = word.split(delimiter)
        result = []
        for splitted_word in splitted_words:
            if splitted_word == '':
                result.append(delimiter)
            else:
                result += tokenize_word_by(splitted_word, delimiters)
            result.append(delimiter)
        result.pop()
        return result
    return tokenize_word_by(word, delimiters)
          

def normalize_word(word):
    delimiters = config.numbers + config.special_chars
    delimiters_count = count_chars(word, delimiters)
    if delimiters_count != 0 and delimiters_count != len(word):
        return tokenize_word_by(word, delimiters)
    return [word]


def normalize_text(text):
    normalizer = hazm.Normalizer()
    normalized_text = normalizer.normalize(text)
    return normalized_text


def tokenize_word(text):
    words = []
    raw_words = hazm.word_tokenize(text)
    for raw_word in raw_words:
        words += normalize_word(raw_word)
    return words


def tokenize_sentence(word_list):
    sentences = []
    sentence = []
    for i in range(len(word_list)):
        word = word_list[i]
        sentence.append(word)
        if word in config.finished_chars:
            if word_list[min(i+1, len(word_list)-1)] in config.finished_chars: continue
            sentences.append(sentence)
            sentence = []
    if len(sentence) != 0:
        sentence.append('.')
        sentences.append(sentence)
    return sentences 


def tokenize_word_and_sentence(text, include_special_chars=False):
    primary_word_list = tokenize_word(text)
    sentences_list  = tokenize_sentence(primary_word_list)
    if include_special_chars:
        words_list = [word for word in list(chain(*sentences_list)) if word not in config.special_chars]
    else:
        words_list = [word for word in list(chain(*sentences_list))]
    return words_list, sentences_list


def import_positive_negative_words_from(path, positive_words_path, negative_words_path):
    data = pd.read_table(path)
    data.columns = ['word', 'sense', 'hasIt']
    pos_data = data[data.sense == 'positive']
    pos_data_arr = np.array(pos_data[pos_data.hasIt == 1].word.tolist())
    np.save(positive_words_path, pos_data_arr)

    neg_data = data[data.sense == 'negative']
    neg_data_arr = np.array(neg_data[neg_data.hasIt == 1].word.tolist())
    np.save(negative_words_path, neg_data_arr)


def report(scores):
    metrics = []
    for metric in config.MODEL_SCORE_METRICS:
        row = {}
        metric_scores = scores['test_' + metric]
        for i in range(len(metric_scores)):
            row['Fold{}'.format(i+1)] = metric_scores[i]
        row['index'] = metric
        metrics.append(row)
    columns = metrics[0].keys()
    metrics_pdf = pd.DataFrame(metrics, columns=columns).set_index('index', drop=True)
    metrics_pdf['mean'] = metrics_pdf.mean(axis=1)
    print(metrics_pdf.to_markdown())
    return metrics_pdf


def plot_feature_importance(feature_importance, X, figsize=(10, 30)):
    indices = np.argsort(feature_importance)
    y_labels = [X.columns[idx] for idx in indices]

    plt.figure(figsize=figsize)
    plt.title("Feature importances")
    plt.barh(range(X.shape[1]), feature_importance[indices], color="r", align="center")
    plt.yticks(range(X.shape[1]), y_labels)
    plt.ylim([-1, X.shape[1]])
    plt.show()


def calculate_accuracy(model, features, labels):
    predictions = model.predict_label(features)
    return accuracy_score(labels, predictions)


def plot_line_chart(x, y_list, xlabel, ylabel, title):
    for label, y in y_list.items():
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def create_dataframe_from_feature_list(feature_list):
    pdf = feature_list.pop()
    for feature in feature_list:
        pdf = pd.merge(pdf, feature, left_on=['label', 'number'], right_on=['label', 'number'])
    pdf = pdf.drop('number', axis=1)
    return pdf


def split_features_and_labels(dataset_pdf, label):
    labels = dataset_pdf[label]
    features = dataset_pdf.drop(label, axis=1)
    return features, labels