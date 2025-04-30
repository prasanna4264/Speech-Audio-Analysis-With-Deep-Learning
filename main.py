import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Model definition
class Conv1DModel(nn.Module):
    def __init__(self, input_shape, num_classes=6):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Encoding
EMOTION_MAP = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 'fear': 4, 'disgust': 5}

def encode(label):
    return EMOTION_MAP[label]

def plot_metrics(history, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.close()

def plot_conf_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTION_MAP.keys())
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def main(csv_path):
    df = pd.read_csv(csv_path)

    zcr_list, rms_list, mfccs_list, emotion_list = [], [], [], []

    print("Loading saved features into memory...")
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        try:
            relative_path = os.path.splitext(row.path.lstrip('./'))[0] + '.npy'
            zcr = np.load(os.path.join("zcr", relative_path))
            rms = np.load(os.path.join("rms", relative_path))
            mfccs = np.load(os.path.join("mfccs", relative_path))

            zcr_list.append(zcr)
            rms_list.append(rms)
            mfccs_list.append(mfccs)
            emotion_list.append(encode(row.emotion))
        except Exception as e:
            print(f"Failed to load features for {row.path}: {e}")

    X = np.concatenate((
        np.swapaxes(zcr_list, 1, 2),
        np.swapaxes(rms_list, 1, 2),
        np.swapaxes(mfccs_list, 1, 2)
    ), axis=2).astype('float32')

    y = np.expand_dims(np.array(emotion_list), axis=1).astype('int8')

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.12, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp)

    y_train_cat = to_categorical(y_train, num_classes=6)
    y_val_cat = to_categorical(y_val, num_classes=6)
    y_test_cat = to_categorical(y_test, num_classes=6)

    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train_cat)), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val_cat)), batch_size=batch_size, pin_memory=True, num_workers=4)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test_cat)), batch_size=batch_size)

    model = Conv1DModel(input_shape=X_train.shape[1:]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=6).to(device)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(200):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, torch.argmax(y_batch, dim=1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_acc += accuracy_metric(outputs, torch.argmax(y_batch, dim=1)).item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, torch.argmax(y_batch, dim=1))
                val_loss += loss.item()
                val_acc += accuracy_metric(outputs, torch.argmax(y_batch, dim=1)).item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        scheduler.step(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:03d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 10:
                print("Early stopping triggered!")
                break

    model.load_state_dict(best_model_state)

    torch.save(model.state_dict(), "best_model.pth")
    plot_metrics(history, ".")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(torch.argmax(y_batch, dim=1).cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_metric(torch.tensor(y_pred), torch.tensor(y_true)).item()
    print(f"\nFinal Test Accuracy: {acc:.2%}")
    plot_conf_matrix(y_true, y_pred, ".")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to df.csv")
    args = parser.parse_args()
    main(args.csv_path)
