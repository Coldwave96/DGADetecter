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

# Weights of vowel characters
def cal_vowel(data):
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
