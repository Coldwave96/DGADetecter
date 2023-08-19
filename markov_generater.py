'''
this script generate the n-gram markov chain model from umbrella top 1m
'''
import pandas as pd
from collections import defaultdict

def train(data_path, n):
    domain_df = pd.read_csv(data_path, header=None)
    words = [row[1].strip().lower() for _, row in domain_df.iterrows()]
    words = ["^" + w + "$" for w in words]

    # Construct a discrete-time markov chain of n-grams
    transitions = defaultdict(lambda: defaultdict(float))
    for word in words:
        if len(word) >= n:
            transitions[""][word[:n]] += 1.0
        for i in range(len(word) - n):
            gram = word[i : i + n]
            next = word[i + 1 : i + n + 1]
            transitions[gram][next] += 1.0
    
    # Normalize the probabilities
    for gram in transitions:
        total = sum([transitions[gram][next] for next in transitions[gram]])
        for next in transitions[gram]:
            transitions[gram][next] /= total
    
    fw = open(f"Outputs/Markov/trans_matrix_{n}.csv", 'w')
    for key1, dict in transitions.items():
        for key2, value in dict.items():
            fw.write('%s\t%s\t%f\n' % (key1, key2, value))
    fw.close()
