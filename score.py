import tqdm
import ast
import argparse
import copy
import os
import time

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from multiprocessing import Pool


def get_probs(model, vectorizer, train_texts, train_labels, eval_texts, tn=10, en=10):
    
    kfold = StratifiedKFold(n_splits=tn, shuffle=True)
    classes = list(set(train_labels))
    classes.sort()
    for _, train_ids in kfold.split(train_texts, train_labels):
        train_vectors = vectorizer.transform([train_texts[idx] for idx in train_ids]).toarray()
        model.partial_fit(train_vectors, np.array(train_labels)[train_ids], classes=classes)

    eval_len = len(eval_texts)
    eval_step = max(1, eval_len // en)
    probs = []
    for i in range(0, eval_len, eval_step):
        start = time.perf_counter()
        eval_vectors = vectorizer.transform(eval_texts[i:i+eval_step]).toarray()
        pred_probs = model.predict_proba(eval_vectors)
        probs.extend(pred_probs.tolist())

    return probs


def score_train(train_data, part, save_root):
    model_dict = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB(),
        'ComplementNB': ComplementNB()
    }

    ids = list(train_data['id'])
    texts = list(train_data[part])
    labels = list(train_data['gfi_label'])

    ids_a, ids_b, texts_a, texts_b, labels_a, labels_b = train_test_split(ids, texts, labels, test_size=0.5, stratify=labels)

    vectorizer_a = CountVectorizer(max_features=10000, min_df=5, stop_words='english').fit(texts_a)
    vectorizer_b = CountVectorizer(max_features=10000, min_df=5, stop_words='english').fit(texts_b)

    score_ids = list(ids_a) + list(ids_b)
    score_dict = {}
    for name in model_dict:
        probs_b = get_probs(copy.deepcopy(model_dict[name]), vectorizer_a, texts_a, labels_a, texts_b)
        probs_a = get_probs(copy.deepcopy(model_dict[name]), vectorizer_b, texts_b, labels_b, texts_a)
        score_dict[name] = probs_a + probs_b

    score = pd.DataFrame()
    score['id'] = ids
    for name in score_dict:
        probs = score_dict[name]            
        probs_dict = dict(zip(score_ids, probs))
        score[name + '_0'] = [probs_dict[idx][0] for idx in ids]
        score[name + '_1'] = [probs_dict[idx][1] for idx in ids]
    score.to_csv(os.path.join(save_root, 'train.csv'), index=False)


def score_test(train_data, test_data, part, save_root):
    model_dict = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB(),
        'ComplementNB': ComplementNB()
    }

    train_texts = list(train_data[part])
    train_labels = list(train_data['gfi_label'])
    vectorizer = CountVectorizer(max_features=10000, min_df=5, stop_words='english').fit(train_texts)

    test_ids = list(test_data['id'])
    test_texts = list(test_data[part])

    score_dict = {}
    for name in model_dict:
        score_dict[name] = get_probs(copy.deepcopy(model_dict[name]), vectorizer, train_texts, train_labels, test_texts)

    score = pd.DataFrame()
    score['id'] = test_ids
    for name in score_dict:
        probs = score_dict[name]
        score[name + '_0'] = [proba[0] for proba in probs]
        score[name + '_1'] = [proba[1] for proba in probs]
    score.to_csv(os.path.join(save_root, 'test.csv'), index=False)


if __name__ == "__main__":

    parts = ['title', 'body', 'comments']

    data_root = 'data'
    features_path = os.path.join(data_root, 'features.csv')
    features = pd.read_csv(features_path, usecols=['id'] + parts)
    features.set_index(keys='id', inplace=True)
    features.fillna('', inplace=True)

    data_sp_root = os.path.join(data_root, 'data_split')
    owners = os.listdir(data_sp_root)
    for owner in owners:
        owner_root = os.path.join(data_sp_root, owner)
        repos = os.listdir(owner_root)
        for repo in repos:
            repo_root = os.path.join(owner_root, repo)
            modes = os.listdir(repo_root)
            for mode in modes:
                print ((owner, repo, mode))
                start = time.perf_counter()

                mode_root = os.path.join(repo_root, mode)
                train_path = os.path.join(mode_root, 'train.csv')
                train = pd.read_csv(train_path)
                train.set_index(keys='id', drop=False, inplace=True)
                train_data = pd.merge(train, features, left_index=True, right_index=True)

                test_path = os.path.join(mode_root, 'test.csv')
                test = pd.read_csv(test_path)
                test.set_index(keys='id', drop=False, inplace=True)
                test_data = pd.merge(test, features, left_index=True, right_index=True)

                for part in parts:
                    save_root = os.path.join(mode_root, 'score', part)
                    if not os.path.exists(save_root):
                        os.makedirs(save_root)

                    score_train(train_data, part, save_root)
                    score_test(train_data, test_data, part, save_root)
                end = time.perf_counter()
                print (end - start)