import torch
import torchmetrics

def evaluate(model, test_loader, device):
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=6).to(device)
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            total_acc += accuracy(outputs, y.argmax(1)).item()
    return total_acc / len(test_loader)