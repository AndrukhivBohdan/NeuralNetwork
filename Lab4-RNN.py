import torch
import torch.nn as nn
import os
import tarfile
import requests
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

def setup_nltk():
    resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'stopwords']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"Error downloading {res}: {e}")

setup_nltk()

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
if not os.path.exists('aclImdb'):
    print("Downloading dataset...")
    response = requests.get(url, stream=True)
    with open("imdb.tar.gz", "wb") as f:
        f.write(response.raw.read())
    with tarfile.open("imdb.tar.gz", "r:gz") as tar:
        tar.extractall()
    print("Dataset downloaded.")

def load_imdb_data(base_path, subset='train'):
    texts, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(base_path, subset, label_type)
        label = 1 if label_type == 'pos' else 0
        #for fname in os.listdir(dir_name)[:8192]: # для прискорення при тестуванні проміжних варіантів моделі можна використати частину набору даних
        for fname in os.listdir(dir_name):
            with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

print("Loading text data...")

train_texts, train_labels = load_imdb_data('aclImdb', 'train')
test_texts, test_labels = load_imdb_data('aclImdb', 'test')

print("Text data loaded.")

all_words = []
for text in train_texts:
    all_words.extend(word_tokenize(text))


vocab = {word: i+2 for i, (word, count) in enumerate(nltk.FreqDist(all_words).most_common(9998))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

def encode_text(texts, max_len=200):
    encoded = []
    for text in texts:
        tokens = word_tokenize(text)
        ids = [vocab.get(t, 1) for t in tokens[:max_len]]
        
        padding_len = max_len - len(ids)
        
        padded_ids = [0] * padding_len + ids
        
        encoded.append(padded_ids)
    return torch.tensor(encoded)

X_train = encode_text(train_texts)
y_train = torch.tensor(train_labels).view(-1, 1)

X_test = encode_text(test_texts)
y_test = torch.tensor(test_labels).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_lstm_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
            
        lstm_dropout = 0.5 if num_lstm_layers > 1 else 0

        self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                            num_layers=num_lstm_layers, 
                            batch_first=True, 
                            dropout=lstm_dropout)
        
        self.fc0 = nn.Linear(hidden_dim,128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16,1)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)

        out = hidden[-1]

        out = torch.relu(self.fc0(out))
        out = self.dropout(out)

        out = torch.relu(self.fc1(out))
        out = self.dropout(out)

        out = torch.relu(self.fc2(out))
        out = self.dropout(out)

        out = torch.relu(self.fc3(out))
        out = self.dropout(out)

        return torch.sigmoid(self.fc4(out))

model = SentimentRNN(10000, 128, 128)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

class EarlyStopping:

    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

early_stopping = EarlyStopping(patience=10)

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(15):

    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch_x, batch_y in train_loader:
        batch_y = batch_y.float() 
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        loss = criterion(outputs, batch_y)
        acc = binary_accuracy(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_acc = epoch_acc / len(train_loader)

    model.eval()
    val_loss = 0
    val_acc = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_y = batch_y.float()
            outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            acc = binary_accuracy(outputs, batch_y)
            
            val_loss += loss.item()
            val_acc += acc.item()
    
    avg_val_loss = val_loss / len(test_loader)
    avg_val_acc = val_acc / len(test_loader)

    scheduler.step(avg_val_loss)
    early_stopping(avg_val_loss)

    print(f'Epoch {epoch+1:02}:')
    print(f'\tTrain Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}%')
    print(f'\t Val. Loss: {avg_val_loss:.4f} |  Val. Acc: {avg_val_acc*100:.2f}%')

    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)
    train_acc_history.append(avg_train_acc)
    val_acc_history.append(avg_val_acc)

    if early_stopping.early_stop:
        print(">>> Early stopping triggered! Stopping training.")
        break

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-s', label='Training Acc')
    plt.plot(epochs, val_accs, 'r-s', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_metrics(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

reviews = [
    "I expected to hate it, but it was actually the best movie of the year.", 
    "The plot was as deep as a puddle.",                                     
    "It was not a bad film, but certainly not a great one either.",           
    "This movie was absolutely amazing and wonderful, I loved every minute!", 
    "Terrible acting, boring plot, and a complete waste of time and money."  
]

print("\n" + "="*50)
print("STARTING CUSTOM REVIEWS ANALYSIS")
print("="*50)

model.eval()

device = next(model.parameters()).device

with torch.no_grad():
    encoded_reviews = encode_text(reviews)
    
    encoded_reviews = encoded_reviews.to(device)
    
    predictions = model(encoded_reviews)
    
    for i, review in enumerate(reviews):
        score = predictions[i].item()
        
        sen = "POSITIVE " if score >= 0.5 else "NEGATIVE "
        
        confidence = abs(score - 0.5) * 2 * 100
        
        print(f"\nReview #{i+1}: \"{review}\"")
        print(f"  -> Raw Output Probability: {score:.4f}")
        print(f"  -> Predicted Sentiment   : {sentiment} (Confidence: {confidence:.1f}%)")

print("\n" + "="*50)

