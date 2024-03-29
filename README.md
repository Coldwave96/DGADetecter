# DGADetecter
Traditional machine learning schemes for detecting DGA domain names are limited by feature engineering and classification algorithms. Manual construction of features is time-consuming and laborious and the detection false alarm rate is high. This repo tries to solve the above limitations through manual features plus deep network to extract features automatically.

## Feature Extractor
### Manual Features
By far, 19 features are selected, detailed description listed blow.

|Feature|Description|
|-|-|
|entropy|Shannon Entropy of Domain Names|
|readability|Domain readability, implemented via textsat|
|vowel_ratio|Proportion of vowels|
|gibberish|Also a readability test, see [here](https://github.com/rrenaud/Gibberish-Detector) for details|
|number_ratio|Proportion of numbers|
|character_ratio|Proportion of characters|
|repeated_char_ratio|Proportion of repeated chars|
|consecutive_ratio|Proportion of consecutive chars|
|consecutive_consonants_ratio|Proportion of consecutive consonants|
|unigram_rank_ave|Mean of rankings after domain unigram cutting|
|unigram_rank_std|Variance of rankings after domain unigram cutting|
|bigram_rank_ave|Mean of rankings after domain bigram cutting|
|bigram_rank_std|Variance of rankings after domain bigram cutting|
|trigram_rank_ave|Mean of rankings after domain trigram cutting|
|trigram_rank_std|Variance of rankings after domain trigram cutting|
|markov_prob_2|Based on the Markov implicit chain model, the transfer probability of dual chars|
|markov_prob_3|Based on the Markov implicit chain model, the transfer probability of tri-chars|
|length_domain|Length of domain|
|tld_rank|TLD rank of domain, based on Umbrella top 1m TLD ranking|

### Automatical Features
GRU is used to extract a 32-dimension feature matrix from purely raw char-level domain data.

## Datasets
* Benign: [Umbrella Top 1m](https://huggingface.co/datasets/c01dsnap/top-1m)
* Malicious: [DGArchive](https://dgarchive.caad.fkie.fraunhofer.de/site/)
* Generated from DGA genrating algorithms, which matained by [360netlab](https://github.com/360netlab/DGA)

**View [here](https://huggingface.co/datasets/c01dsnap/DGADetector-Mixed) for concated datasets.**

## Go
### Train
1. (Optional) Run `python dga_generater.py` for help, if you want to generate dga domains via dga generating algorithms.
2. Run `python train.py` to start the whole progress based on original datasets in **Datasets** folder, including pre-processing, training and evaluation.
3. Run `python train_processed.py` to re-train the model based on processed datasets in **Outputs/Datasets/** folder.

### Inference
* Run `python api.py` to start the API server, and default port is **8000**.

* Send `POST` requests to server nad embed query domains in the following json format.
```Json
{
    "id": 1,
    "domains": [
        "google.com",
        "oqdykzptntg33nzl38f52mxfyoxm49nyorau.ru"
    ]
}
```

* Server will response with corresponding request id, labels and probs.
```Json
{
    "request_id": 1,
    "labels": [
        "benign",
        "malicious"
    ],
    "Probs": [
        1.0,
        1.0
    ]
}
```
