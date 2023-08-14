import os
import math
import pickle
import textstat

import gibberish_train

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
        command = "python gibberish_train"
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
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
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
