'''
this script generate the reference rank list of unigram, bigram, trigram from umbrella top 1m
'''
import tldextract
import pandas as pd
from collections import defaultdict

def bigrams(words):
    wprev = None
    for w in words:
        if not wprev == None:
            yield (wprev, w)
        wprev = w

def trigrams(words):
    wprev1 = None
    wprev2 = None
    for w in words:
        if not (wprev1 == None or wprev2 == None):
            yield (wprev1, wprev2, w)
        wprev1 = wprev2
        wprev2 = w

def train(data_path):
    unigram_rank = defaultdict(int)
    bigram_rank = defaultdict(int)
    trigram_rank = defaultdict(int)

    domain_df = pd.read_csv(data_path, header=None)
    extract = tldextract.TLDExtract(include_psl_private_domains=True)

    for _, row in domain_df.iterrows():
        rank = row[0]
        domain = row[1]
        ext = extract(domain)

        main_domain = '$' + ext.domain + '$'
        for i in main_domain[1:-1]:
            unigram_rank[i] += 1
        for i in bigrams(main_domain):
            bigram_rank[''.join(i)] += 1
        for i in trigrams(main_domain):
            trigram_rank[''.join(i)] += 1
        
        tld = ext.suffix
        tld_split = tld.split('.')
        if len(tld_split) > 1:
            tld = tld_split[-1]
            tld_domain = '$' + tld_split[-2] + '$'
            for i in tld_domain[1:-1]:
                unigram_rank[i] += 1
            for i in bigrams(tld_domain):
                bigram_rank[''.join(i)] += 1
            for i in trigrams(tld_domain):
                trigram_rank[''.join(i)] += 1

        subdomain = ext.subdomain
        subdomain_split = subdomain.split('.')
        if len(subdomain_split) > 0:
            for string in subdomain_split:
                string = '$' + string + '$'
                for i in string[1:-1]:
                    unigram_rank[i] += 1
                for i in bigrams(string):
                    bigram_rank[''.join(i)] += 1
                for i in trigrams(string):
                    trigram_rank[''.join(i)] += 1

        file = open("Datasets/Words/N-gram/ngram-rank-freq.txt", 'w')
        for rank, (i, freq) in enumerate(sorted(unigram_rank.items(), key = lambda x:x[1], reverse = True)):
            try:
                file.write('1,%s,%d,%d\n' % (i, freq, rank + 1))
            except UnicodeEncodeError:
                continue
        for rank, (i, freq) in enumerate(sorted(bigram_rank.items(), key = lambda x:x[1], reverse = True)):
            try:
                file.write('2,%s,%d,%d\n' % (i, freq, rank + 1))
            except UnicodeEncodeError:
                continue
        for rank,(i, freq) in enumerate(sorted(trigram_rank.items(), key = lambda x:x[1], reverse = True)):
            try:
                file.write('3,%s,%d,%d\n' % (i, freq, rank+1))
            except UnicodeEncodeError:
                continue
        file.close()
