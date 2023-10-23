
import json
import pandas as pd

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils import DomainDataset, CombinedModel

print("[*] Loading datasets...")
padded_sequences_path = "Outputs/Datasets/Processed/padded_sequences_59.csv"
additional_features_normalized_path = "Outputs/Datasets/Processed/additional_features.csv"
labels_path = "Outputs/Datasets/Processed/labels.csv"

# Load processed datasets
padded_sequences = pd.read_csv(padded_sequences_path).iloc[:, 1:].to_numpy()
additional_features_normalized = pd.read_csv(additional_features_normalized_path).iloc[:, 1:].to_numpy()
labels = pd.read_csv(labels_path)['0'].to_numpy()

# Load label dict from local files
with open("Outputs/Datasets/label_dict.json", 'r') as json_file:
    labels_dict = json.load(json_file)

# Split datasets for training and testing
x_train_seq, x_test_seq, x_train_features, x_test_features, y_train, y_test = train_test_split(padded_sequences, additional_features_normalized, labels, test_size=0.2, random_state=42)

print("[*] All done!\n\n[*] Start training...")

train_dataset = DomainDataset(x_train_seq, x_train_features, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create a model instant
vocab_size = int(padded_sequences_path.split('.')[0].split('/')[-1].split('_')[-1])
embedding_size = 64
feature_size = 32
additional_features_size = additional_features_normalized.shape[1]
num_classes = len(labels_dict)
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
        
        if batch_idx % 1000 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "Outputs/Models/combined_model.pth")
print("[*] Model saved!")

# Evaluation
print("[*] Evaluation")
model.eval()
with torch.no_grad():
    y_pred_probs = model(torch.tensor(x_test_seq), torch.FloatTensor(x_test_features))
    predicted_classes = torch.argmax(y_pred_probs, dim=1)
    classification_report = classification_report(y_test, predicted_classes, target_names=[labels_dict[i] for i in sorted(labels_dict.keys())])
    print(classification_report)
