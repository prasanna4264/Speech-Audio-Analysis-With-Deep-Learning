import torch.nn as nn
import torch

class Conv1DModel(nn.Module):
    def __init__(self, input_shape, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(input_shape[1], 128, 3)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.global_avg_pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)