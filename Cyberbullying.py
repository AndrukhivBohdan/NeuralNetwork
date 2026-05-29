# region IMPORT

import re
import os
import kagglehub
import random
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

from collections import Counter

import torch
import torch.nn as nn

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score
)

from torch.utils.data import (
    TensorDataset,
    DataLoader
)
# endregion

SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

nltk.download('punkt', quiet=True)

LABEL_COLS = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]

NUM_LABELS = len(LABEL_COLS)

VOCAB_SIZE = 20000
MAX_LEN = 200

WORD_DROPOUT_PROB = 0.1

EMBED_DIM = 128
HIDDEN_DIM = 128

BATCH_SIZE = 128
EPOCHS = 20

# region DATA_PROCESSIOG
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("julian3833/jigsaw-toxic-comment-classification-challenge")

print("Dataset path:", dataset_path)
train_csv_path = os.path.join(dataset_path, "train.csv")

df = pd.read_csv(train_csv_path)
#df = df.sample(n=20000, random_state=SEED).reset_index(drop=True)

# ОЧИЩЕННЯ ТЕКСТУ

def clean_text(text):

    text = str(text)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()

def augment_tokens(tokens, dropout_prob=0.1):

    if len(tokens) <= 3:
        return tokens

    new_tokens = []

    for token in tokens:

        if random.random() > dropout_prob:
            new_tokens.append(token)

    if len(new_tokens) == 0:
        return tokens

    return new_tokens


print("Cleaning text...")
df["clean_text"] = df["comment_text"].apply(clean_text)

print("Tokenizing text...")
df["tokens"] = df["clean_text"].apply(word_tokenize)

# РОЗПОДІЛ ТЕСТОВОЇ ТА НАВЧАЛЬНОЇ МНОЖИНИ

train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)

train_df["tokens"] = train_df["tokens"].apply(
    lambda x: augment_tokens(x, WORD_DROPOUT_PROB)
)

rare_df = train_df[
    (train_df["threat"] == 1) |
    (train_df["identity_hate"] == 1)
]

train_df = pd.concat(
    [train_df, rare_df, rare_df],
    ignore_index=True
)

print(f"Train size: {len(train_df)}")
print(f"Test size : {len(test_df)}")


# СТВОРЕННЯ СЛОВНИКА

all_tokens = []
for seq in train_df["tokens"]:
    all_tokens.extend(seq)

freq = Counter(all_tokens)

most_common = freq.most_common(VOCAB_SIZE - 2)

word2idx = {"<PAD>": 0, "<UNK>": 1}

for word, _ in most_common:
    word2idx[word] = len(word2idx)

print(f"Vocabulary size: {len(word2idx)}")


# КОДУВАННЯ ТЕКСТУ

def encode_and_pad(token_seqs, word2idx, max_len):

    encoded = []
    for seq in token_seqs:
        ids = [
            word2idx.get(tok, 1)
            for tok in seq
        ]
        ids = ids[:max_len]

        padding = [0] * (max_len - len(ids))

        ids = ids + padding

        encoded.append(ids)

    return np.array(encoded, dtype=np.int64)



X_train = encode_and_pad(train_df["tokens"], word2idx, MAX_LEN)

X_test = encode_and_pad(test_df["tokens"], word2idx, MAX_LEN)

y_train = train_df[LABEL_COLS].values.astype(np.float32)
y_test = test_df[LABEL_COLS].values.astype(np.float32)


# ПІДВАНТАЖУВАЧІ ДАНИХ

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
# endregion


# МОДЕЛЬ
class ToxicClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_labels)

    def forward(self, x):

        embedded = self.embedding(x)
        
        _, (hidden, _) = self.lstm(embedded)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        hidden_cat = torch.cat((forward_hidden, backward_hidden), dim=-1) 
        
        out = self.dropout(hidden_cat)
        
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        
        out = torch.relu(self.fc2(out))
        out = self.dropout(out)
        
        out = self.fc3(out) 
        return out

model = ToxicClassifier(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM, 
    hidden_dim=HIDDEN_DIM,     
    num_labels=NUM_LABELS         
).to(device)


class_counts = y_train.sum(axis=0)

pos_weights = len(y_train) / (2 * class_counts)

pos_weights = torch.tensor(
    pos_weights,
    dtype=torch.float32
).to(device)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weights
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2)


class EarlyStopping:

    def __init__(
        self, patience=4, min_delta=0):
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

early_stopping = EarlyStopping()


# МЕТРИКИ

def evaluate_metrics(y_true, y_prob):

    y_pred = (y_prob >= 0.5).astype(int)

    f1 = f1_score(y_true,y_pred,average="macro",zero_division=0)

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
            average="macro"
        )

    except:
        roc_auc = 0

    return f1, roc_auc

def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.05)

    best_t = 0.5
    best_f1 = 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1

def full_class_report(y_true, y_prob, labels):

    results = {}

    for i, label in enumerate(labels):

        best_t, _ = find_best_threshold(
            y_true[:, i],
            y_prob[:, i]
        )

        y_pred = (y_prob[:, i] >= best_t).astype(int)

        results[label] = {
            "threshold": best_t,
            "precision": precision_score(y_true[:, i], y_pred, zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred, zero_division=0),
            "f1": f1_score(y_true[:, i], y_pred, zero_division=0)
        }

    return results


train_loss_history = []
test_loss_history = []

train_f1_history = []
test_f1_history = []

for epoch in range(EPOCHS):

    # НАВЧАННЯ
    model.train()

    train_loss = 0
    train_probs = []
    train_targets = []

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        probs = torch.sigmoid(outputs)

        train_probs.append(probs.detach().cpu().numpy())
        train_targets.append(batch_y.cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)

    train_probs = np.vstack(train_probs)
    train_targets = np.vstack(train_targets)

    train_f1, train_auc = evaluate_metrics(train_targets, train_probs)

    # ТЕСТ

    model.eval()

    test_loss = 0
    test_probs = []
    test_targets = []

    with torch.no_grad():

        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            test_loss += loss.item()

            probs = torch.sigmoid(outputs)

            test_probs.append(probs.cpu().numpy())
            test_targets.append(batch_y.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)

    test_probs = np.vstack(test_probs)
    test_targets = np.vstack(test_targets)

    test_f1, test_auc = evaluate_metrics(test_targets, test_probs)

    scheduler.step(avg_test_loss)
    early_stopping(avg_test_loss)

    train_loss_history.append(avg_train_loss)
    test_loss_history.append(avg_test_loss)

    train_f1_history.append(train_f1)
    test_f1_history.append(test_f1)

    print(f"\nEpoch {epoch+1}")

    print(
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train F1: {train_f1:.4f} | "
        f"Train AUC: {train_auc:.4f}"
    )

    print(
        f"Test Loss: {avg_test_loss:.4f} | "
        f"Test F1: {test_f1:.4f} | "
        f"Test AUC: {test_auc:.4f}"
    )

    if early_stopping.early_stop:
        print("\nEarly stopping triggered!")
        break


model.eval()

all_probs = []
all_targets = []

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(device)

        outputs = model(batch_x)
        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu().numpy())
        all_targets.append(batch_y.numpy())

y_prob = np.vstack(all_probs)
y_true = np.vstack(all_targets)

y_pred = (y_prob >= 0.5).astype(int)

print("\n=== TEST REPORT (threshold=0.5) ===\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=LABEL_COLS,
    zero_division=0
))

print("\n=== OPTIMAL THRESHOLD PER CLASS (TEST) ===\n")

results = full_class_report(y_true, y_prob, LABEL_COLS)

for label, metrics in results.items():

    print("=" * 40)
    print(label)
    print(f"Threshold : {metrics['threshold']:.2f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
