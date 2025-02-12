import torch
import torch.nn as nn
import torch.optim as optim

class CNN_Model(nn.Module):
    def __init__(self, input_shape):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=256, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(256, 96, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(96, 32, kernel_size=7, padding=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)  # Regression output

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.global_avg_pool(x).squeeze(-1)  # Global Average Pooling
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Linear activation for regression
        return x


class LSTM_Model(nn.Module):
    def __init__(self, input_size):
        super(LSTM_Model, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, batch_first=True, dropout=0.2, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, dropout=0.2, num_layers=1)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, num_layers=1)
        self.fc1 = nn.Linear(32, 96)
        self.fc2 = nn.Linear(96, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        self.lstm1.flatten_parameters()  # 让 LSTM 权重变得连续
        x, _ = self.lstm1(x)
    
        self.lstm2.flatten_parameters()
        x, _ = self.lstm2(x)
    
        self.lstm3.flatten_parameters()
        x, _ = self.lstm3(x)
    
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_Classifier(nn.Module):
    def __init__(self, input_shape):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=256, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(256, 96, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(96, 32, kernel_size=7, padding=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling

        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)  # 分类任务，输出 1 维

        self.sigmoid = nn.Sigmoid()  # 用于二分类

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.global_avg_pool(x).squeeze(-1)  # Global Average Pooling
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)  # 确保输出在 [0,1] 之间（分类任务）
        return x

class LSTM_Classifier(nn.Module):
    def __init__(self, input_size):
        super(LSTM_Classifier, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, batch_first=True, dropout=0.2, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, dropout=0.2, num_layers=1)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, num_layers=1)
        self.fc1 = nn.Linear(32, 96)
        self.fc2 = nn.Linear(96, 128)
        self.fc3 = nn.Linear(128, 1)  # 输出 1 个值（用于二分类）
        self.sigmoid = nn.Sigmoid()   # 确保输出在 [0,1]

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)

        self.lstm2.flatten_parameters()
        x, _ = self.lstm2(x)

        self.lstm3.flatten_parameters()
        x, _ = self.lstm3(x)

        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)  # 归一化输出，适用于二分类
        return x
