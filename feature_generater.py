import os
import math
import pickle
import textstat
import tldextract
import numpy as np

import gibberish_train
from ngram_freq_generater import bigrams, trigrams

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
def cal_Readability(data):
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
def cal_gibberish(data):
    model_path = 'Outputs/Gibberish/gib_model.pickle'
    
    if not os.path.exists(model_path):
        command = "python gibberish_train.py"
        os.system(command)
    
    model = pickle.load(open(model_path, 'rb'))
    model_mat = model['mat']

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
def load_ngram_rank_file():
    ngram_rank_file_path = "Datasets/Words/N-gram/ngram-rank-freq.txt"
    if not os.path.exists(ngram_rank_file_path):
        command = "python ngram_freq_generater.py"
        os.system(command)

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
    bigram_rank_main = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in bigrams(main_domain)])
    trigram_rank_main = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in trigrams(main_domain)])
    
    tld = ext.suffix
    tld_split = tld.split('.')
    if len(tld_split) > 1:
        tld = tld_split[-1]
        tld_domain = '$' + tld_split[-2] + '$'
        unigram_rank_tld = np.array([ngram_rank_dict[i] if i in ngram_rank_dict else 0 for i in tld_domain[1:-1]])
        bigram_rank_tld = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in bigrams(tld_domain)])
        trigram_rank_tld = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in trigrams(tld_domain)])
        
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
            bigram_rank_sub = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in bigrams(string)])
            trigram_rank_sub = np.array([ngram_rank_dict[''.join(i)] if ''.join(i) in ngram_rank_dict else 0 for i in trigrams(string)])

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