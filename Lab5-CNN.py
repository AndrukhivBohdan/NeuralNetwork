import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Завантаження навчального набору CIFAR-100...")
train_set = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

print("Завантаження тестового набору CIFAR-100...")
test_set = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

print(f"\nУспішно завантажено!")
print(f"Кількість картинок для навчання: {len(train_set)}")
print(f"Кількість картинок для тесту: {len(test_set)}")


class SimpleCNN(nn.Module):
    def __init__(self): 
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,padding=1,padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3,padding=1, padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,padding=1, padding_mode='circular')
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 100)

        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1) 

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)
        return x
    
model = SimpleCNN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss() 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
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


def multiclass_accuracy(preds, y):
    top_preds = preds.argmax(dim=1, keepdim=True)
    correct = top_preds.eq(y.view_as(top_preds)).sum()
    return correct.float() / y.shape[0]


early_stopping = EarlyStopping(patience=5)

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

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

for epoch in range(50):

    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch_x, batch_y in train_loader:

        batch_y = batch_y.long() 
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        loss = criterion(outputs, batch_y)
        acc = multiclass_accuracy(outputs, batch_y)
        
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
            batch_y = batch_y.long()
            outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            acc = multiclass_accuracy(outputs, batch_y)
            
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

    plot_metrics(train_loss_history, val_loss_history, train_acc_history, val_acc_history)
