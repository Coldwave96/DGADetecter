import json
import torch
import uvicorn
from joblib import load
from fastapi import FastAPI, Request
from sklearn.preprocessing import StandardScaler

import feature_generater
from utils import CombinedModel

benign_domain_path = "Datasets/Benign/top-1m-umbrella.csv"
benign_tld_path = "Datasets/Benign/top-1m-TLD-umbrella.csv"
gibberish_dataset_path = "Datasets/Words/Gibberish/gibberish.txt"

def load_config_files():
    global model_mat, ngram_rank_dict, transitions_2, transitions_3, tld_top_rank_df, labels_dict
    model_mat = feature_generater.load_gib_model(gibberish_dataset_path)
    ngram_rank_dict = feature_generater.load_ngram_rank_file(benign_domain_path)
    transitions_2 = feature_generater.load_trans_matrix(benign_domain_path, 2)
    transitions_3 = feature_generater.load_trans_matrix(benign_domain_path, 3)
    tld_top_rank_df = feature_generater.load_tld_rank_file(benign_tld_path)

    # with open("Outputs/Datasets/label_dict.json", 'r') as json_file:
    with open("Outputs/Datasets/label_dict_binary.json", 'r') as json_file: # Binary
        labels_dict = json.load(json_file)

def cal_designed_feature(domain):
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
    
    markov_domain = '^' + domain + '$'
    markov_prob_2 = feature_generater.cal_markov_probs(markov_domain, transitions_2, 2)
    markov_prob_3 = feature_generater.cal_markov_probs(markov_domain, transitions_3, 3)
    
    length_domain = feature_generater.cal_length(domain)
    tld_rank = feature_generater.cal_tld_rank(domain, tld_top_rank_df)

    designed_feature = [
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

    return designed_feature

app = FastAPI()

@app.post("/")
async def predict(request: Request):
    json_post_raw = await request.json()
    request_id = json_post_raw["id"]
    request_domains = json_post_raw["domains"]

    designed_features = []
    for domain in request_domains:
        designed_feature = cal_designed_feature(domain)
        designed_features.append(designed_feature)

    # scaler = load("Outputs/Datasets/Processed/scaler.joblib")
    scaler = load("Outputs/Datasets/Processed/scaler_binary.joblib") # Binary
    designed_features_standardized = scaler.transform(designed_features)
    designed_features_standardized = torch.FloatTensor(designed_features_standardized)
    
    char_to_index = {char: idx + 1 for idx, char in enumerate(sorted(set(''.join([domain for domain in request_domains]))))}
    sequences = [[char_to_index[char] for char in domain] for domain in request_domains]

    max_length = 50
    padded_sequences = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(seq) for seq in sequences], batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :max_length]

    vocab_size = 59
    embedding_size = 64
    feature_size = 32
    designed_features_size = designed_features_standardized.shape[1]
    num_classes = len(labels_dict)
    model = CombinedModel(vocab_size, embedding_size, feature_size, designed_features_size, num_classes)
    # model.load_state_dict(torch.load("Outputs/Models/combined_model_59_64_32_19_93.pth"))
    # model.load_state_dict(torch.load("Outputs/Models/combined_model_59_64_32_19_2.pth")) # Binary
    # model.load_state_dict(torch.load("Outputs/Models/combined_model.pth")) # Processed
    model.load_state_dict(torch.load("Outputs/Models/combined_model_vocabSize59_embeddingSize64_featureSize32_additionalFeaturesSize19_numClasses2_malicious71494_benign70000.pth")) # Binary

    model.eval()
    with torch.no_grad():
        y_pred_probs = model(padded_sequences.clone().detach(), designed_features_standardized.clone().detach())
        predicts = torch.max(torch.round(y_pred_probs * 1000) / 1000, dim=1)
        predicted_classes = predicts.indices.tolist()
        predicted_probs = predicts.values.tolist()
        
        labels_classes = [labels_dict[str(i)] for i in predicted_classes]
    
        response = {
            "request_id": request_id,
            "labels": labels_classes,
            "Probs": predicted_probs
        }
    
        return response

if __name__ == '__main__':
    load_config_files()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
