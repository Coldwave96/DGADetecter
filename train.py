import os
import pandas as pd

import feature_generater

# Datasets file path
benign_domain_path = "Datasets/Benign/top-1m-umbrella.csv"
benign_tld_path = "Datasets/Benign/top-1m-TLD-umbrella.csv"
malicious_dgarchive_dir = "Datasets/Malicious/Existed/"
gibberish_dataset_path = "Datasets/Words/Gibberish/gibberish.txt"

# Load all dga families files
def list_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

# Load all the dga domains into a Dataframe
dga_domain_df = pd.DataFrame()
dga_families_dict = dict(int)
malicious_dgarchive_file_list = list_files_in_folder(malicious_dgarchive_dir)
for file in malicious_dgarchive_file_list:
    dgarchive_df = pd.read_csv(file)
    dga_family = ""
    for _, row in dgarchive_df.iterrows():
        dga_domain = row[0]
        if len(dga_family) == 0:
            dga_family = dga_family.join(row[-1].split('_')[0])
            if dga_family not in dga_families_dict:
                dga_families_dict[dga_family] = len(dga_families_dict) + 1
        temp_dga = pd.DataFrame(
            {
                'domain': dga_domain,
                'family': dga_family,
                'label': dga_families_dict[dga_family]
            },
            index = [dga_domain_df.size]
        )
        dga_domain_df = pd.concat([dga_domain_df, temp_dga], ignore_index=True)

# Load all the benign domain into a Dataframe
benign_domain_df = pd.DataFrame()
umbrella_df = pd.read_csv(benign_domain_df)
for _, row in umbrella_df.iterrows():
    benign_domain = row[1]
    temp_benign = pd.DataFrame(
        {
            'domain': benign_domain,
            'family': 'benign',
            'label': 0
        },
        index = [benign_domain_df.size]
    )
    benign_domain_df = pd.concat([benign_domain_df, temp_benign], ignore_index=True)

# Concat malicious and benign domains
dataset_df = pd.concat([benign_domain_df, dga_domain_df], ignore_index=True)

# Generate man-made features & Convert original domain string into a vector for extracting features
additional_features = []
labels = []
for _, row in dataset_df.iterrows():
    domain = row['domain']
    label = row['label']

    entropy = feature_generater.cal_entropy(domain)
    readability = feature_generater.cal_readability(domain)
    vowel_ratio = feature_generater.cal_vowels(domain)

    model_mat = feature_generater.load_gib_model(gibberish_dataset_path)
    gibberish = feature_generater.cal_gibberish(domain)

    number_ratio = feature_generater.cal_numbers(domain)
    character_ratio = feature_generater.cal_characters(domain)
    repeated_char_ratio = feature_generater.cal_repeated_characters(domain)
    consecutive_ratio = feature_generater.cal_consecutive_characters(domain)
    consecutive_consonants_ratio = feature_generater.cal_consecutive_consonants(domain)

    ngram_rank_dict = feature_generater.load_ngram_rank_file(benign_domain_path)
    unigram_rank_ave, unigram_rank_std, bigram_rank_ave, bigram_rank_std, trigram_rank_ave, trigram_rank_std = feature_generater.cal_ngam_rank_stats(domain, ngram_rank_dict)

    transitions_2 = feature_generater.load_trans_matrix(benign_domain_path, 2)
    transitions_3 = feature_generater.load_trans_matrix(benign_domain_path, 3)
    markov_prob_2 = feature_generater.cal_markov_probs(domain, transitions_2, 2)
    markov_prob_3 = feature_generater.cal_markov_probs(domain, transitions_3, 3)

    length_domain = feature_generater.cal_length(domain)

    tld_top_rank_df = feature_generater.load_tld_rank_file(benign_tld_path)
    tld_rank = feature_generater.cal_tld_rank(domain, tld_top_rank_df)

    additional_feature = [
        entropy,
        readability,
        vowel_ratio,
        gibberish,
        number_ratio,
        character_ratio,
        repeated_char_ratio,
        consecutive_ratio,
        consecutive_consonants_ratio,
        unigram_rank_ave,
        unigram_rank_std,
        bigram_rank_ave,
        bigram_rank_std,
        trigram_rank_ave,
        trigram_rank_std,
        markov_prob_2,
        markov_prob_3,
        length_domain,
        tld_rank
    ]
    additional_features.append(additional_feature)
    
    labels.append(label)

# Normalize all the features
