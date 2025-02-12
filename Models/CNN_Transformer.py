import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # 形状 (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class CNN_Transformer(nn.Module):
    def __init__(self, input_channels, seq_length, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(CNN_Transformer, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=3)

        # 位置编码层
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全连接层（回归）
        self.fc1 = nn.Linear(d_model, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        # CNN 提取局部特征
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # 变换形状以适配 Transformer: (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        x = x.permute(0, 2, 1)

        # 位置编码
        x = self.positional_encoding(x)

        # Transformer Encoder
        x = self.transformer(x)

        # 取最后一个时间步的输出
        x = x[:, -1, :]

        # 通过全连接层输出最终回归值
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CNN_Transformer_Classifier(nn.Module):
    def __init__(self, input_channels, seq_length, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(CNN_Transformer_Classifier, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=3)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 50)
        self.fc2 = nn.Linear(50, 1)  # 输出一个值，接 `sigmoid`

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # CNN 提取局部特征
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # 变换形状以适配 Transformer: (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        x = x.permute(0, 2, 1)

        # 位置编码
        x = self.positional_encoding(x)

        # Transformer Encoder
        x = self.transformer(x)

        # 取最后一个时间步的输出（适合分类任务）
        x = x[:, -1, :]

        # 通过全连接层输出最终分类概率
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)  # 确保输出在 [0,1]

        return x
