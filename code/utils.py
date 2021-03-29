import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

from tabulate import tabulate

import os
from config import MODEL_SCORE_METRICS



def count_chars(s, li):
    counts = [s.count(c) for c in li]
    return sum(counts)


def import_name_from(name):
    return int(name[:-4])


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
    for metric in MODEL_SCORE_METRICS:
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


def plot_feature_importance(feature_importance, X):
    indices = np.argsort(feature_importance)
    y_labels = [X.columns[idx] for idx in indices]

    plt.figure(figsize=(10, 15))
    plt.title("Feature importances")
    plt.barh(range(X.shape[1]), feature_importance[indices], color="r", align="center")
    plt.yticks(range(X.shape[1]), y_labels)
    plt.ylim([-1, X.shape[1]])
    plt.show()

