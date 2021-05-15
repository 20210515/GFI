import textstat
import tqdm
import ast
import os

import pandas as pd
import numpy as np

from multiprocessing import Pool


def readability_unit(args, preproc=False):
    _, row = args
    readability_list = []

    mid_fix = '' if preproc else 'full_'

    title = row['title_' + mid_fix + 'text']
    body = row['body_' + mid_fix + 'text']
    comments = [] if row['comments_' + mid_fix + 'text_list'] == '' else ast.literal_eval(row['comments_' + mid_fix + 'text_list'])

    readability_list.append(get_readability_func(title))
    readability_list.append(get_readability_func(body))
    readability_list.append(get_readability_func(' '.join(comments)))
    
    if len(comments) > 0:
        comments_readability_list = [get_readability_func(comment) for comment in comments]
        readability_list.append(np.mean(comments_readability_list))
    else:
        readability_list.append(get_readability_func(''))

    return readability_list


def make_global(get_readability_func_in):
    global get_readability_func
    get_readability_func = get_readability_func_in


if __name__ == "__main__":
    data_root = 'data'
    selected_data_path = os.path.join(data_root, 'selected_data.csv')
    body_data_path = os.path.join(data_root, 'body_data.csv')
    comments_data_path = os.path.join(data_root, 'comments_data.csv')

    selected_data = pd.read_csv(selected_data_path, usecols=['id', 'title_full_text', 'title_text'])
    selected_data.set_index(keys='id', drop=False, inplace=True)
    body_data = pd.read_csv(body_data_path, usecols=['id', 'body_full_text', 'body_text'])
    body_data.set_index(keys='id', inplace=True)
    comments_data = pd.read_csv(comments_data_path, usecols=['id', 'comments_full_text_list', 'comments_text_list'])
    comments_data.set_index(keys='id', inplace=True)
    
    data = pd.merge(selected_data, body_data, left_index=True, right_index=True)
    data = pd.merge(data, comments_data, left_index=True, right_index=True)
    data.fillna('', inplace=True)

    root = os.path.join(data_root, 'readability')
    if not os.path.exists(root):
        os.makedirs(root)

    form_func = {
        'fre': textstat.flesch_reading_ease,
        'smog': textstat.smog_index,
        'fkg': textstat.flesch_kincaid_grade,
        'cli': textstat.coleman_liau_index,
        'ari': textstat.automated_readability_index,
        'dcrs': textstat.dale_chall_readability_score,
        'dw': textstat.difficult_words,
        'lwf': textstat.linsear_write_formula,
        'gf': textstat.gunning_fog,
    }

    issue_ids = list(data['id'])
    parts = ['title', 'body', 'comments_tol', 'comments_avg']

    for form in form_func:
        get_readability_func = form_func[form]
        with Pool(initializer=make_global, initargs=(get_readability_func,)) as pool:
            readability_list_list = list(tqdm.tqdm(pool.imap(readability_unit, data.iterrows()), total=len(data), ncols=100))
                
        readability = pd.DataFrame(readability_list_list, columns=parts)
        readability.insert(0, 'id', issue_ids)

        readability.to_csv(os.path.join(root, form + '.csv'), index=False)