import math
import textstat

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
