import os
import sys
import math
import pickle
import textstat
import tldextract
import numpy as np
import pandas as pd
from collections import defaultdict

import gibberish_train
import markov_generater
import ngram_freq_generater

# Information Entropy
def cal_entropy(data):
    if not data:
        return 0
    valid_chars = set(data)

    entropy = 0
    for x in valid_chars:
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    
    return entropy

# Readability
def cal_readability(data):
    score = textstat.flesch_reading_ease(data)
    return score

# Ratio of vowel characters
def cal_vowels(data):
    vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']

    vowel_count = 0
    total_chars = 0

    for char in data:
        if char.isalpha():
            total_chars += 1
            if char in vowels:
                vowel_count += 1
    
    if total_chars > 0:
        vowel_ratio = vowel_count / total_chars
        return vowel_ratio
    else:
        return 0.0

# Gibberish probability
def load_gib_model(data_path):
    model_path = 'Outputs/Gibberish/gib_model.pickle'
    
    if not os.path.exists(model_path):
        gibberish_train.train(data_path)
    
    model = pickle.load(open(model_path, 'rb'))
    model_mat = model['mat']
    return model_mat

def cal_gibberish(data, model_mat):
    return gibberish_train.avg_transition_prob(data, model_mat)

# Ratio of numbers
def cal_numbers(data):
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    number_count = 0
    for char in data:
        if char in numbers:
            number_count += 1
    
    number_ratio = number_count / len(data)
    return number_ratio

# Ratio of characters
def cal_characters(data):
    character_count = 0
    for char in data:
        if char.isalpha():
            character_count += 1
    
    character_ratio = character_count / len(data)
    return character_ratio

# Ratio of repeated characters
def cal_repeated_characters(data):
    char_count = {}

    for char in data:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    repeated_char_count = sum(1 for count in char_count.values() if count > 1)
    repeated_char_ratio = repeated_char_count / len(data)
    return repeated_char_ratio

# Ratio of consecutive characters
def cal_consecutive_characters(data):
    consecutive_count = 0

    for i in range(len(data) - 1):
        if data[i] == data[i + 1] or data[i].lower() == data[i + 1].lower():
            consecutive_count += 1
    
    total_pairs = len(data) - 1
    if total_pairs > 0:
        consecutive_ratio = consecutive_count / total_pairs
        return consecutive_ratio
    else:
        return 0.0

# Ratio of consecutive consonants
def is_consonant(char):
    consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w','x', 'y', 'z']
    return char in consonants

def cal_consecutive_consonants(data):
    consecutive_consonants_count = 0

    for i in range(len(data) - 1):
        if is_consonant(data[i]) and is_consonant(data[i + 1]):
            consecutive_consonants_count += 1

    total_pairs = len(data) - 1
    if total_pairs > 0:
        consecutive_consonants_ratio = consecutive_consonants_count / total_pairs
        return consecutive_consonants_ratio
    else:
        return 0.0
    
# N-gram rank stats
def load_ngram_rank_file(data_path):
    ngram_rank_file_path = "Datasets/Words/N-gram/ngram-rank-freq.txt"
    if not os.path.exists(ngram_rank_file_path):
        ngram_freq_generater.train(data_path)

    ngram_rank_file = open(ngram_rank_file_path, 'r')
    ngram_rank_dict = dict()
    for i in ngram_rank_file:
        _, gram, _, rank = i.strip().split(',')
        ngram_rank_dict[gram] = int(rank)
    ngram_rank_file.close()
    return ngram_rank_dict

def ave(array_):
    if len(array_) > 0:
        return array_.mean()
    else:
        return 0
    
def std(array_):
    if len(array_) > 0:
        return array_.std()
    else:
        return 0

def cal_ngam_rank_stats(data, ngram_rank_dict):
    extract = tldextract.TLDExtract(include_psl_private_domains=True)
    ext = extract(data)
    
    main_domain = '$' + ext.domain + '$'
    unigram_rank_main = np.array([ngram_rank_dict[i] if i in ngram_rank_dict else 0 for i in main_domain[1:-1]])
    bigram_rank_main = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in ngram_freq_generater.bigrams(main_domain)])
    trigram_rank_main = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in ngram_freq_generater.trigrams(main_domain)])
    
    tld = ext.suffix
    tld_split = tld.split('.')
    if len(tld_split) > 1:
        tld = tld_split[-1]
        tld_domain = '$' + tld_split[-2] + '$'
        unigram_rank_tld = np.array([ngram_rank_dict[i] if i in ngram_rank_dict else 0 for i in tld_domain[1:-1]])
        bigram_rank_tld = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in ngram_freq_generater.bigrams(tld_domain)])
        trigram_rank_tld = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in ngram_freq_generater.rigrams(tld_domain)])
        
        unigram_rank = np.concatenate((unigram_rank_main, unigram_rank_tld))
        bigram_rank = np.concatenate((bigram_rank_main, bigram_rank_tld))
        trigram_rank = np.concatenate((trigram_rank_main, trigram_rank_tld))
    else:
        unigram_rank = unigram_rank_main
        bigram_rank = bigram_rank_main
        trigram_rank = trigram_rank_main

    subdomain = ext.subdomain
    subdomain_split = subdomain.split('.')
    if len(subdomain_split) > 0:
        for string in subdomain_split:
            string = '$' + string + '$'
            unigram_rank_sub = np.array([ngram_rank_dict[i] if i in ngram_rank_dict else 0 for i in string[1:-1]])
            bigram_rank_sub = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in ngram_freq_generater.bigrams(string)])
            trigram_rank_sub = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in ngram_freq_generater.trigrams(string)])

            unigram_rank = np.concatenate((unigram_rank, unigram_rank_sub))
            bigram_rank = np.concatenate((bigram_rank, bigram_rank_sub))
            trigram_rank = np.concatenate((trigram_rank, trigram_rank_sub))
    
    unigram_rank_ave = ave(unigram_rank)
    unigram_rank_std = std(unigram_rank)

    bigram_rank_ave = ave(bigram_rank)
    bigram_rank_std = std(bigram_rank)

    trigram_rank_ave = ave(trigram_rank)
    trigram_rank_std = std(trigram_rank)

    return unigram_rank_ave, unigram_rank_std, bigram_rank_ave, bigram_rank_std, trigram_rank_ave, trigram_rank_std

# Markov chain
def load_trans_matrix(data_path, n):
    trans_matrix_path = f"Outputs/Markov/trans_matrix_{n}.csv"
    if not os.path.exists(trans_matrix_path):
        markov_generater.train(data_path, n)
    
    transitions = defaultdict(lambda: defaultdict(float))
    f_trans = open(trans_matrix_path, 'r')
    for f in f_trans:
        key1, key2, value = f.strip().split('\t')
        value = float(value)
        transitions[key1][key2] = value
    f_trans.close()
    return transitions

def cal_markov_probs(data, transitions, n):
    if n == 2:
        ngram = [''.join((i, j)) for i, j in ngram_freq_generater.bigrams(data) if not i == None]
    elif n == 3:
        ngram = [''.join((i, j)) for i, j in ngram_freq_generater.trigrams(data) if not i == None]
    else:
        print("N-length should be 2 or 3.")
        sys.exit(0)
    
    prob = transitions[''][ngram[0]]
    for x in range(len(ngram) - 1):
        next_step = transitions[ngram[x]][ngram[x + 1]]
        prob *= next_step
    
    return prob

# Length of domain
def cal_length(data):
    return len(data)

# TLD rank
def load_tld_rank_file(data_path):
    default_path = "Outputs/TLD-Rank/tld_top_rank.csv"
    if os.path.exists(default_path):
        tld_top_rank_df = pd.read_csv(default_path)
    else:
        tld_top_rank_df = pd.DataFrame()
        tld_rank_df = pd.read_csv(data_path)
        for _, row in tld_rank_df.iterrows():
            if len(row[1].strip().split('.')) == 1:
                temp = pd.DataFrame(
                    {
                        'rank': row[0],
                        'tld': row[1]
                    },
                    index = [tld_top_rank_df.size]
                )
                tld_top_rank_df = pd.concat([tld_top_rank_df, temp], ignore_index=True)
        tld_top_rank_df.to_csv("Outputs/TLD-Rank/tld_top_rank.csv")
    
    return tld_top_rank_df

def cal_tld_rank(data, tld_top_rank_df):
    tld = data.split('.')[-1]
    for _, row in tld_top_rank_df.iterrows():
        if row['tld'] == tld:
            return row['rank']
        else:
            return -1
