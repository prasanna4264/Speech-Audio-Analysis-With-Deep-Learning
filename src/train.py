import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics

def train_model(model, train_loader, val_loader, device, num_epochs=100, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=6).to(device)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_acc = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y.argmax(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += accuracy(outputs, y.argmax(1)).item()
        total_loss /= len(train_loader)
        total_acc /= len(train_loader)

        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_loss += criterion(out, y.argmax(1)).item()
                val_acc += accuracy(out, y.argmax(1)).item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Acc={total_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    return model