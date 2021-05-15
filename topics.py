import nltk
import tqdm
import ast
import os
import time

import pandas as pd

from multiprocessing import Pool
from gensim.models import LdaMulticore, LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import MWETokenizer


def make_dictionary_global(text_dictionary_in):
    global text_dictionary
    text_dictionary = text_dictionary_in


def doc2bow_unit(text):
    return text_dictionary.doc2bow(text)


def make_model_global(lda_model_in):
    global lda_model
    lda_model = lda_model_in


def get_document_topics_unit(bow):
    topics = lda_model.get_document_topics(bow)
    return topics, len(topics)


def tokenize(text):
    return nltk.word_tokenize(text)


def lda_train(train_data, part, save_root):
    ids = list(train_data['id'])
    texts = list(train_data[part])

    with Pool() as pool:
        texts = list(tqdm.tqdm(pool.imap(tokenize, texts), total=len(texts), ncols=100))

    text_dictionary = Dictionary(texts)
    text_dictionary.save(os.path.join(save_root, 'dict'))

    with Pool(initializer=make_dictionary_global, initargs=(text_dictionary,)) as pool:
        texts = list(tqdm.tqdm(pool.imap(doc2bow_unit, texts), total=len(texts), ncols=100))

    lda_model = LdaMulticore(texts, workers=7)
    lda_model.save(os.path.join(save_root, 'model'))

    with Pool(initializer=make_model_global, initargs=(lda_model,)) as pool:
        rows = list(tqdm.tqdm(pool.imap(get_document_topics_unit, texts), total=len(texts), ncols=100))
    topics = pd.DataFrame(rows, columns=['topics', 'topic_num'])
    topics.insert(0, 'id', ids)
    topics.to_csv(os.path.join(save_root, 'train.csv'), index=False)

    return text_dictionary, lda_model


def lda_test(test_data, part, text_dictionary, lda_model, save_root):
    test_ids = list(test_data['id'])
    test_texts = list(test_data[part])

    with Pool() as pool:
        test_texts = list(tqdm.tqdm(pool.imap(tokenize, test_texts), total=len(test_texts), ncols=100))

    with Pool(initializer=make_dictionary_global, initargs=(text_dictionary,)) as pool:
        test_texts = list(tqdm.tqdm(pool.imap(doc2bow_unit, test_texts), total=len(test_texts), ncols=100))

    with Pool(initializer=make_model_global, initargs=(lda_model,)) as pool:
        test_rows = list(tqdm.tqdm(pool.imap(get_document_topics_unit, test_texts), total=len(test_texts), ncols=100))
    test_topics = pd.DataFrame(test_rows, columns=['topics', 'topic_num'])
    test_topics.insert(0, 'id', test_ids)
    test_topics.to_csv(os.path.join(save_root, 'test.csv'), index=False)


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
                    save_root = os.path.join(mode_root, 'topics', part)
                    if not os.path.exists(save_root):
                        os.makedirs(save_root)

                    text_dictionary, lda_model = lda_train(train_data, part, save_root)
                    lda_test(test_data, part, text_dictionary, lda_model, save_root)
                    