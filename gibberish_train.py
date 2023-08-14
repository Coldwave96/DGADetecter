import math
import pickle

accepted_chars = 'abcdefghijklmnopqrstuvwxyz '
pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

# Return only the subset of chars from acceptrd_chars
# Ignore punctuation, symbols, etc.
def normalize(line):
    return [c.lower() for c in line if c.lower() in accepted_chars]

# Return all n-grams fron normalized line
def ngram(n, line):
    normalized_line = normalize(line)
    for start in range(0, len(normalized_line) - n + 1):
        yield "".join(normalized_line[start, start + n])

# Return the average transition prob from line through log_prob_mat.
def avg_transition_prob(line, log_prob_mat):
    log_prob = 0.0
    transition_ct = 0
    for a, b in ngram(2, line):
        log_prob += log_prob_mat[pos[a]][pos[b]]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))

def train():
    k = len(accepted_chars)
    counts = [[10 for i in range(k)] for i in range(k)]

    # Count transitions from big text file
    for line in open('Datasets/Words/Gibberish/gibberish.txt'):
        for a, b in ngram(2, line):
            counts[pos[a]][pos[b]] += 1
    
    # Normalize the counts via log probabilities
    for _, row in enumerate(counts):
        s = float(sum(row))
        for j in range(len(row)):
            row[j] = math.log(row[j] / s)

    pickle.dump({'mat': counts}, open('Outputs/Gibberish/gib_model.pickle', 'wb'))

    if __name__ == '__main__':
        train()
