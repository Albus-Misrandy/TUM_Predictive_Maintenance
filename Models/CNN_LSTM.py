import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义 CNN-LSTM 模型
class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, seq_length):
        super(CNN_LSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=3)
        
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # 调整维度以适应 LSTM 输入 (batch, seq_len, features)
        
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x = x[:, -1, :]  # 取 LSTM 最后一个时间步的输出
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class EarlyStopping:
    def __init__(self, patience=10, path="best_CNN_LSTM_model.pth"):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.path = path
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class CNN_LSTM_classification(nn.Module):
    def __init__(self, input_size, seq_len):
        super(CNN_LSTM_classification, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=3)
        
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=50, batch_first=True, num_layers=1)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度: (batch_size, features, seq_len)
        x = self.conv1(x)  # 现在 shape 正确
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.permute(0, 2, 1)  # 再调整回来给 LSTM
        x, _ = self.lstm1(x)
        x = torch.dropout(x, p=0.2, train=self.training)
        x, _ = self.lstm2(x)
        x = torch.dropout(x, p=0.2, train=self.training)

        x = x[:, -1, :]  # 取最后一个时间步
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x