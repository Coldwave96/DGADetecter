import torch
from torch.utils.data import Dataset

class DomainDataset(Dataset):
    def __init__(self, sequences, features, labels):
        self.sequences = sequences
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.features[idx], self.labels[idx]
    
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
        # predictions = torch.nn.functional.log_softmax(predictions, dim=1)
        predictions = torch.sigmoid(predictions) # Binary
        return predictions
