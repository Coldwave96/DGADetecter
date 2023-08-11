import math

# Information Entropy
def info_entropy(data):
    if not data:
        return 0
    valid_chars = set(data)

    entropy = 0
    for x in valid_chars:
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    
    return entropy
