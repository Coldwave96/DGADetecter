import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
print("[*] Start loading DGA datasets...")
dga_domain_df = pd.DataFrame()
labels_dict = dict()
malicious_dgarchive_file_list = list_files_in_folder(malicious_dgarchive_dir)
num_dga_domains = 0
max_sample = 100000
print(f"Found {len(malicious_dgarchive_file_list)} DGA domain files in total.\n")
for file in malicious_dgarchive_file_list:
    file_name = file.strip().split('/')[-1]
    print(f"Loading {file_name}...")
    dgarchive_df = pd.read_csv(file, header=None)
    if dgarchive_df.shape[0] > max_sample:
        dgarchive_df = dgarchive_df.iloc[:max_sample]
    dga_family = ""
    for index, row in dgarchive_df.iterrows():
        dga_domain = row[0]
        if len(dga_family) == 0:
            dga_family = dga_family.join(row[len(row) - 1].split('_')[0])
            if dga_family not in labels_dict:
                labels_dict[dga_family] = len(labels_dict) + 1
        temp_dga = pd.DataFrame(
            {
                'domain': dga_domain,
                'family': dga_family,
                'label': labels_dict[dga_family]
            },
            index = [dga_domain_df.shape[0]]
        )
        dga_domain_df = pd.concat([dga_domain_df, temp_dga], ignore_index=True)

        if dgarchive_df.shape[0] > 10 and index % int(dgarchive_df.shape[0] / 10) == 0:
            print(f"Progress: {index} / {dgarchive_df.shape[0]}")

    num_dga_domains += dgarchive_df.size
    print(f"Done with {file_name}, loaded {dgarchive_df.size} {dga_family} DGA domains.\n")
print(f"[*] Done with all the DGA fmailes, {num_dga_domains} DGA damins in total.\n")

# Load all the benign domain into a Dataframe
print("[*] Start loading benign datasets...")
labels_dict['benign'] = 0
benign_domain_df = pd.DataFrame()
benign_df = pd.read_csv(benign_domain_path, header=None)
for index, row in benign_df.iterrows():
    benign_domain = row[1]
    temp_benign = pd.DataFrame(
        {
            'domain': benign_domain,
            'family': 'benign',
            'label': 0
        },
        index = [benign_domain_df.shape[0]]
    )
    benign_domain_df = pd.concat([benign_domain_df, temp_benign], ignore_index=True)

    if index % (benign_df.shape[0] / 10) == 0:
        print(f"Progress: {index} / {benign_df.shape[0]}")
print(f"[*] Done with benign dataset, {benign_df.shape[0]} benign domains in total.\n")

# Concat malicious and benign domains, then save to a local csv file
dataset_df = pd.concat([benign_domain_df, dga_domain_df], ignore_index=True)
dataset_df.to_csv("Outputs/Datasets/raw_domains_mix.csv")
print("[*] Mixed raw domains saved in Outputs/Datasets/raw_domains_mix.csv\n")

# Generate man-made features & Convert original domain string into a vector for extracting features
additional_features = []
labels = []
domains = []

print("[*] Loading config files for generating human designed features...")
model_mat = feature_generater.load_gib_model(gibberish_dataset_path)
ngram_rank_dict = feature_generater.load_ngram_rank_file(benign_domain_path)
transitions_2 = feature_generater.load_trans_matrix(benign_domain_path, 2)
transitions_3 = feature_generater.load_trans_matrix(benign_domain_path, 3)
tld_top_rank_df = feature_generater.load_tld_rank_file(benign_tld_path)

print("[*] Start generating human designed features...")
for index, row in dataset_df.iterrows():
    domain = row['domain']
    label = row['label']

    entropy = feature_generater.cal_entropy(domain)
    readability = feature_generater.cal_readability(domain)
    vowel_ratio = feature_generater.cal_vowels(domain)
    gibberish = feature_generater.cal_gibberish(domain, model_mat)
    number_ratio = feature_generater.cal_numbers(domain)
    character_ratio = feature_generater.cal_characters(domain)
    repeated_char_ratio = feature_generater.cal_repeated_characters(domain)
    consecutive_ratio = feature_generater.cal_consecutive_characters(domain)
    consecutive_consonants_ratio = feature_generater.cal_consecutive_consonants(domain)
    unigram_rank_ave, unigram_rank_std, bigram_rank_ave, bigram_rank_std, trigram_rank_ave, trigram_rank_std = feature_generater.cal_ngam_rank_stats(domain, ngram_rank_dict)
    markov_prob_2 = feature_generater.cal_markov_probs(domain, transitions_2, 2)
    markov_prob_3 = feature_generater.cal_markov_probs(domain, transitions_3, 3)
    length_domain = feature_generater.cal_length(domain)
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
    
    domains.append(domain)
    labels.append(label)

    if index % (dataset_df.shape[0] / 10) == 0:
        print(f"Progress: {index} / {dataset_df.shape[0]}")

# Save original features into a local csv file
additional_features_df = pd.DataFrame(additional_features)
additional_features_df.to_csv("Outputs/Datasets/features_original.csv")
print("[*] Original features saved in Outputs/Datasets/features_original.csv")

# Normalize all the features since there are no catagorical features
print("[*] Normalizing numerical human designed features...")
additional_features_array = np.array(additional_features)
mean = np.mean(additional_features_array, axis=0)
std = np.std(additional_features_array, axis=0)
additional_features_normalized = (additional_features_array - mean) / std

print("[*] Processing raw domains...")
# Create the vectorizer for raw domains
char_to_index = {char: idx + 1 for idx, char in enumerate(sorted(set(''.join([domain for domain in domains]))))}
vocab_size = len(char_to_index) + 1

# Convert raw domains into sequences
sequences = [[char_to_index[char] for char in domain] for domain in domains]

# Padding all sequences to the same length
max_length = 50
padded_sequences = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(seq) for seq in sequences], batch_first=True, padding_value=0)
padded_sequences = padded_sequences[:, :max_length]

# Split datasets for training and testing
x_train_seq, x_test_seq, x_train_features, x_test_features, y_train, y_test = train_test_split(padded_sequences, additional_features_normalized, labels, test_size=0.2, random_state=42)

print("[*] All done!\n\n [*] Start training...")

class DomainDataset(Dataset):
    def __init__(self, sequences, features, labels):
        self.sequences = sequences
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.features[idx], self.labels[idx]

train_dataset = DomainDataset(x_train_seq, x_train_features, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class FeatureExtrator(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, feature_size):
        super(FeatureExtrator, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.gru = torch.nn.GRU(embedding_size, feature_size, batch_first=True)
    
    def forward(self, x):
        embedded_x = self.embedding(x)
        gru_output, _ = self.gru(embedded_x)
        features = gru_output[:, -1, :]
        return features

# Construct the combined model
class CombinedModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, feature_size, additional_features_size, num_classes):
        super(CombinedModel, self).__init__()
        self.feature_extractor = FeatureExtrator(vocab_size, embedding_size, feature_size)
        self.fc = torch.nn.Linear(feature_size + additional_features_size, num_classes)

    def forward(self, x, additional_features):
        extracted_features = self.feature_extractor(x)
        combined_features = torch.cat((extracted_features, additional_features), dim=1)
        predictions = self.fc(combined_features)
        predictions = torch.nn.functional.log_softmax(predictions, dim=1)
        return predictions

# Create a model instant
embedding_size = 32
feature_size = 8
additional_features_size = len(additional_features_normalized[0])
num_classes = len(labels_dict) + 1
model = CombinedModel(vocab_size, embedding_size, feature_size, additional_features_size, num_classes)

# Train
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (seq, features, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(seq, features.float())
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

# Evaluation
print("[*] Evaluation")
model.eval()
with torch.no_grad():
    y_pred_probs = model(x_test_seq, torch.FloatTensor(x_test_features))
    predicted_classes = torch.argmax(y_pred_probs, dim=1)
    classification_report = classification_report(y_test, predicted_classes, target_names=labels_dict)
    print(classification_report)

# Save model
torch.save(model.state_dict(), 'Outputs/Models/combined_model.pth')
print("[*] Model saved!")
