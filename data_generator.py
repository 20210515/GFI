import os
import json
import markdown
import datetime
import tqdm
import re
import copy
import ast
import pickle
import nltk
import jieba

import pandas as pd

from orderedset import OrderedSet
from multiprocessing import Pool
from pandas import DataFrame
from bs4 import BeautifulSoup
from urlextract import URLExtract
from collections import OrderedDict


def read_data(path):
    data = pd.read_csv(path)
    data.fillna('', inplace=True)
    data.set_index(keys='id', drop=False, inplace=True)
    return data


def get_owner_repo(onwer_repo_root):
    root = onwer_repo_root
    
    owner_repo_set = OrderedSet()
    owners = os.listdir(root)
    for owner in owners:
        owner_root = os.path.join(root, owner)
        repos = os.listdir(owner_root)
        for repo in repos:
            owner_repo_set.add((owner, repo))

    return owner_repo_set


def text_tokenize(text):
    text = ' '.join(nltk.word_tokenize(text))
    return text.lower()


def get_args_list(owner_repo_set, issues_root):
    args_list = []
    for owner, repo in owner_repo_set:
        issues_repo_root = os.path.join(issues_root, owner, repo)
        issue_dirs = os.listdir(issues_repo_root)
        for issue_dir in issue_dirs:
            args_list.append([owner, repo, issue_dir])
    return args_list


def issue_label_proc(text):
    text = re.sub('%20', ' ', text)
    words = re.split(r'[\W_]', text)
    words = list(filter(None, words))
    text_ = ' '.join(words)
    return text_.lower()


def base_data_global(issues_root_value, gfi_label_set_value):
    global issues_root
    issues_root = issues_root_value
    global gfi_label_set
    gfi_label_set = gfi_label_set_value


def base_data_extract(args):
    owner, repo, issue_dir = args
    issue_path = os.path.join(issues_root, owner, repo, issue_dir, 'issue')
    with open(issue_path, 'r') as fr:
        issue = json.load(fr)

    created_at = issue['created_at']
    closed_at = issue['closed_at']

    issue_id = issue['id']
    issue_num = issue['number']

    all_proc_labels = [issue_label_proc(issue_label['name']) for issue_label in issue['labels']]
    gfi_label = 1 if len(gfi_label_set.intersection(all_proc_labels)) > 0 else 0

    issue_title = issue['title']
    title_proc = re.sub(r'#\d+', ' BDRNUMBDR ', issue_title)

    title_proc = text_tokenize(title_proc)

    issue_body = issue['body']
    author_association = issue['author_association']
    user = issue['user']
    user_id = user['id']
    user_name = user['login']
    user_type = user['type']

    data_row = [
        issue_id, owner, repo, issue_num, 
        issue_title, title_proc, issue_body, author_association, 
        user_id, user_name, user_type, gfi_label, 
        created_at, closed_at, all_proc_labels
    ]

    return data_row


def generate_base_data(gfi_label_set, owner_repo_set, issues_root, save_path):

    args_list = get_args_list(owner_repo_set, issues_root)

    with Pool(initializer=base_data_global, initargs=(issues_root, gfi_label_set)) as pool:
        base_data_rows = list(tqdm.tqdm(pool.imap(base_data_extract, args_list), total=len(args_list), ncols=100))

    base_cols = [
        'id', 'owner', 'repo', 'number', 
        'title_full_text', 'title_text', 'body_md', 'author_association', 
        'user_id', 'user_name', 'user_type', 'gfi_label', 
        'created_at', 'closed_at', 'all_proc_labels'
    ]
    base_data = DataFrame(base_data_rows, columns=base_cols)
    base_data.to_csv(save_path, index=False)

    return base_data.set_index(keys='id', drop=False)


def labels_process_global(github_labels_set_value, gfi_label_set_value, event_root_value):
    global github_labels_set
    github_labels_set = github_labels_set_value
    global gfi_label_set
    gfi_label_set = gfi_label_set_value
    global event_root
    event_root = event_root_value


def get_pre_labels(gfi_events_path):
    with open(gfi_events_path, 'r') as fr:
        events = json.load(fr)

    gfi_time_dict = {}
    time_dict = {}
    for event in events:
        if event['event'] != 'labeled':
            continue

        label_name = issue_label_proc(event['label']['name'])
        
        if label_name in gfi_label_set:
            gfi_time_dict[event['created_at']] = datetime.datetime.strptime(event['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        elif label_name in github_labels_set:
            time_dict[label_name] = event['created_at']

    if len(gfi_time_dict) > 0:
        gfi_time, time_start = min(list(gfi_time_dict.items()), key=lambda x: x[1])
    else:
        gfi_time = ''
        time_start = None

    pre_labels = []
    if time_start is not None:
        for key in time_dict:
            label_time = datetime.datetime.strptime(time_dict[key], '%Y-%m-%dT%H:%M:%SZ')
            if label_time < time_start:
                pre_labels.append(key)

    if len(pre_labels) == 0:
        pre_labels = None
    
    return gfi_time, pre_labels


def labels_process(args):
    _, row = args
    if row['gfi_label'] == 0:
        gfi_time = ''
        all_proc_labels = ast.literal_eval(row['all_proc_labels'])
        pre_labels = list(github_labels_set.intersection(all_proc_labels))
        event_error = 0
    else:
        issue_id = row['id']
        issue_number = row['number']
        issue_dir = str(issue_id) + '_' + str(issue_number)
        owner = row['owner']
        repo = row['repo']
        gfi_events_path = os.path.join(event_root, owner, repo, issue_dir, 'events')
        if not os.path.exists(gfi_events_path):
            gfi_time = 'Error'
            pre_labels = 'Error'
            event_error = 1
        else:
            gfi_time, pre_labels = get_pre_labels(gfi_events_path)
            if gfi_time == '':
                gfi_time = 'Error'
                pre_labels = 'Error'
                event_error = 1
            else:
                event_error = 0
    return gfi_time, pre_labels, event_error


def generate_labels_data(base_data, github_labels_set, gfi_label_set, event_root, save_path):
    with Pool(initializer=labels_process_global, initargs=(github_labels_set, gfi_label_set, event_root)) as pool:
        labels_data_rows = list(tqdm.tqdm(pool.imap(labels_process, base_data.iterrows()), total=len(base_data), ncols=100))

    labels_data_cols = ['gfi_time', 'pre_labels', 'event_error']
    labels_data = DataFrame(labels_data_rows, columns=labels_data_cols)
    labels_data.insert(loc=0, column='id', value=list(base_data['id']))
    labels_data.to_csv(save_path, index=False)

    return labels_data.set_index(keys='id', drop=False)


def select_data_global(min_time_value, max_time_value):
    global min_time
    min_time = min_time_value
    global max_time
    max_time = max_time_value


def select_data(args):
    _, row = args
    issue_id = row['id']
    gfi_label = row['gfi_label']
    event_error = row['event_error']
    if gfi_label == 1:
        if event_error == 0:
            return issue_id
        else:
            return None
    else:
        create_time = datetime.datetime.strptime(row['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        if create_time >= max_time:
            return None
        
        if row['closed_at'] is None or row['closed_at'] == '':
            return issue_id
        else:
            close_time = datetime.datetime.strptime(row['closed_at'], '%Y-%m-%dT%H:%M:%SZ')
            if close_time <= min_time:
                return None
            else:
                return issue_id


def generate_selected_data(base_data, labels_data, save_path, min_issue=10):
    base_data = base_data.set_index(keys='id', drop=False)
    labels_data = labels_data.set_index(keys='id')
    merge_data = pd.merge(base_data, labels_data, left_index=True, right_index=True)
    owner_repo_set = OrderedSet(zip(merge_data['owner'], merge_data['repo']))

    selected_issue_ids = []
    for owner, repo in owner_repo_set:
        print ((owner, repo))
        owner_data = merge_data[merge_data['owner'] == owner]
        repo_data = owner_data[owner_data['repo'] == repo]
        if len(repo_data) < min_issue:
            continue
        pos_data = repo_data[repo_data['gfi_label'] == 1]
        valid_gfi_times = [datetime.datetime.strptime(gfi_time, '%Y-%m-%dT%H:%M:%SZ') for gfi_time, event_error in zip(list(pos_data['gfi_time']), list(pos_data['event_error'])) if event_error == 0]
        if len(valid_gfi_times) == 0:
            continue
        
        min_gfi_time = min(valid_gfi_times)
        max_gfi_time = max(valid_gfi_times)
        with Pool(initializer=select_data_global, initargs=(min_gfi_time, max_gfi_time)) as pool:
            issue_ids = list(tqdm.tqdm(pool.imap(select_data, repo_data.iterrows()), total=len(repo_data), ncols=100))
        issue_ids = list(filter(None, issue_ids))
        selected_issue_ids.extend(issue_ids)

    selected_issue_ids = list(OrderedSet(base_data['id']).intersection(selected_issue_ids))
    selected_data = base_data.loc[selected_issue_ids]
    selected_data.to_csv(save_path, index=False)

    return selected_data.set_index(keys='id', drop=False)


def body_process_global():
    global extractor
    extractor = URLExtract(extract_email=True)


def body_process(issue_body):
    if issue_body is None or issue_body == '':
        body_full_text = ''
        body_text = ''
        body_codes = None
        body_codes_num = 0
        text_href_list = None
        url_num = 0
    else:
        text = markdown.markdown(issue_body)
        soup = BeautifulSoup(text, 'lxml')
        body_full_text = soup.get_text()
        code_list = soup.findAll('code')
        body_codes = [code.get_text() for code in code_list]
        a_list = soup.findAll('a')
        text_href_list = extractor.find_urls(body_full_text)

        for code in code_list:
            new_tag = soup.new_tag('p')
            new_tag.string = ' BDRCODEBDR '
            code.replace_with(new_tag)
        body_text = soup.get_text()
        
        for text_href in text_href_list:
            temp_text = body_text.replace(text_href, ' BDRURLBDR ')
            body_text = temp_text
            del temp_text
        
        body_text = re.sub(r'#\d+', ' BDRNUMBDR ', body_text)
        body_text = text_tokenize(body_text)
        
        url_num = len(a_list) + len(text_href_list)
        body_codes_num = len(body_codes)
        body_codes = None if len(body_codes) == 0 else body_codes
        text_href_list = None if len(text_href_list) == 0 else text_href_list

    return body_full_text, body_text, body_codes, body_codes_num, text_href_list, url_num


def generate_body_data(selected_data, save_path):

    with Pool(initializer=body_process_global) as pool:
        body_data_rows = list(tqdm.tqdm(pool.imap(body_process, list(selected_data['body_md'])), total=len(selected_data), ncols=100))
    body_cols = ['body_full_text', 'body_text', 'body_codes', 'body_codes_num', 'text_href_list', 'url_num']
    body_data = DataFrame(body_data_rows, columns=body_cols)
    body_data.insert(loc=0, column='id', value=list(selected_data['id']))
    body_data.to_csv(save_path, index=False)

    return read_data(save_path)


def comments_process_global(comments_root_value):
    global comments_root
    comments_root = comments_root_value
    global extractor
    extractor = URLExtract(extract_email=True)


def comments_process(args):
    _, row = args
    issue_id = row['id']
    issue_number = row['number']
    issue_dir = str(issue_id) + '_' + str(issue_number)
    owner = row['owner']
    repo = row['repo']
    comments_path = os.path.join(comments_root, owner, repo, issue_dir, 'comments')

    gfi_time = row['gfi_time']

    time_start = None if gfi_time is None or gfi_time == '' else datetime.datetime.strptime(gfi_time, '%Y-%m-%dT%H:%M:%SZ')

    if os.path.exists(comments_path):
        with open(comments_path, 'r') as fr:
            comments = json.load(fr)
    else:
        comments = []

    comments_full_text_list = []
    comments_text_list = []
    comments_codes = {}
    comments_codes_num_list = []
    comments_hrefs = {}
    comments_url_num_list = []
    for i in range(len(comments)):
        comment = comments[i]
        if time_start is not None:
            comment_time = datetime.datetime.strptime(comment['created_at'], '%Y-%m-%dT%H:%M:%SZ')
            if comment_time >= time_start:
                continue
        
        body_full_text, body_text, body_codes, body_codes_num, text_href_list, url_num = body_process(comment['body'])
        comments_full_text_list.append(body_full_text)
        comments_text_list.append(body_text)
        if body_codes is not None:
            comments_codes[i] = body_codes
        comments_codes_num_list.append(body_codes_num)
        if text_href_list is not None:
            comments_hrefs[i] = text_href_list
        comments_url_num_list.append(url_num)

    comments_full_text = ' '.join(comments_full_text_list)
    comments_text = ' '.join(comments_text_list)
    comments_codes_num = sum(comments_codes_num_list)
    comments_url_num = sum(comments_url_num_list)

    comments_full_text_list = None if len(comments_full_text_list) == 0 else comments_full_text_list
    comments_text_list = None if len(comments_text_list) == 0 else comments_text_list
    comments_codes = None if len(comments_codes) == 0 else comments_codes
    comments_codes_num_list = None if len(comments_codes_num_list) == 0 else comments_codes_num_list
    comments_hrefs = None if len(comments_hrefs) == 0 else comments_hrefs
    comments_url_num_list = None if len(comments_url_num_list) == 0 else comments_url_num_list

    return comments_full_text_list, comments_full_text, comments_text_list, comments_text, comments_codes, comments_codes_num_list, comments_codes_num, comments_hrefs, comments_url_num_list, comments_url_num
    

def generate_comments_data(selected_data, labels_data, comments_root, save_path):
    selected_data = selected_data.set_index(keys='id', drop=False)
    labels_data = labels_data.set_index(keys='id')
    merge_data = pd.merge(selected_data, labels_data, left_index=True, right_index=True)
    merge_data = merge_data[['id', 'number', 'owner', 'repo', 'gfi_time']]
    with Pool(initializer=comments_process_global, initargs=(comments_root,)) as pool:
        comments_data_rows = list(tqdm.tqdm(pool.imap(comments_process, merge_data.iterrows()), total=len(merge_data), ncols=100))

    comments_data_cols = [
        'comments_full_text_list', 'comments_full_text', 'comments_text_list', 
        'comments_text', 'comments_codes', 'comments_codes_num_list', 'comments_codes_num', 
        'comments_hrefs', 'comments_url_num_list', 'comments_url_num'
    ]
    comments_data = DataFrame(comments_data_rows, columns=comments_data_cols)
    comments_data.insert(loc=0, column='id', value=list(selected_data['id']))
    comments_data.to_csv(save_path, index=False)

    return comments_data.set_index(keys='id', drop=False)


#--------------------------------------------------------------------------------
def pre_gfi_global(pos_data_value):
    global pos_data
    pos_data = pos_data_value


def pre_gfi_unit(args):

    _, row = args
    owner = row['owner']
    repo = row['repo']
    user_id = row['user_id']
    issue_time = datetime.datetime.strptime(row['created_at'], '%Y-%m-%dT%H:%M:%SZ')

    sub_data = pos_data[pos_data['user_id'] == user_id]
    sub_data = sub_data[sub_data['repo'] == repo]
    sub_data = sub_data[sub_data['owner'] == owner]

    pre_gfi_value = len([sub_row for _, sub_row in sub_data.iterrows() if datetime.datetime.strptime(sub_row['created_at'], '%Y-%m-%dT%H:%M:%SZ') < issue_time])

    return pre_gfi_value


def generate_pre_gfi(selected_data, save_path):
    pos_data = selected_data[selected_data['gfi_label'] == 1]
    with Pool(initializer=pre_gfi_global, initargs=(pos_data,)) as pool:
        pre_gfi_list = list(tqdm.tqdm(pool.imap(pre_gfi_unit, selected_data.iterrows()), total=len(selected_data), ncols=100))
    
    pre_gfi = pd.DataFrame()
    pre_gfi['id'] = list(selected_data['id'])
    pre_gfi['pre_gfi'] = pre_gfi_list

    pre_gfi.to_csv(save_path, index=False)

    return pre_gfi.set_index(keys='id', drop=False)


def run(data_save_root, owner_repo_set):

    gfi_labels_list = ['good first issue', 'easy', 'Easy', 'low hanging fruit', 'minor bug', 'Easy Pick', 'Easy to Fix', 'good first bug', 'beginner', 'good first contribution', 'Good first task', 'newbie', 'starter bug', 'beginner-task', 'easy-pick', 'minor feature', 'help wanted(easy)', 'up-for-grabs', 'good-first-bug']
    gfi_label_set = OrderedSet([issue_label_proc(label) for label in gfi_labels_list])
    
    print ('base_data')
    issues_root = 'github_issues'
    base_data_path = os.path.join(data_save_root, 'base_data.csv')
    base_data = generate_base_data(gfi_label_set, owner_repo_set, issues_root, base_data_path)
    base_data = read_data(base_data_path)

    print ('labels_data')
    github_labels_set = OrderedSet([
        'bug', 'documentation', 'duplicate', 'enhancement', 
        'help wanted', 'invalid', 'question', 'wontfix'
    ])
    event_root = 'github_gfi_events'
    labels_data_path =  os.path.join(data_save_root, 'labels_data.csv')
    labels_data = generate_labels_data(base_data, github_labels_set, gfi_label_set, event_root, labels_data_path)
    labels_data = read_data(labels_data_path)

    print ('selected_data')
    selected_data_path = os.path.join(data_save_root, 'selected_data.csv')
    selected_data = generate_selected_data(base_data, labels_data, selected_data_path)
    selected_data = read_data(selected_data_path)

    print ('body_data')
    body_data_path = os.path.join(data_save_root, 'body_data.csv')
    body_data = generate_body_data(selected_data, body_data_path)
    body_data = read_data(body_data_path)

    print ('comments_data')
    comments_root = 'github_comments'
    comments_data_path = os.path.join(data_save_root, 'comments_data.csv')
    comments_data = generate_comments_data(selected_data, labels_data, comments_root, comments_data_path)
    comments_data = read_data(comments_data_path)

    print ('pre_gfi')
    pre_gfi_path = os.path.join(data_save_root, 'pre_gfi.csv')
    pre_gfi = generate_pre_gfi(selected_data, pre_gfi_path)
    pre_gfi = read_data(pre_gfi_path)