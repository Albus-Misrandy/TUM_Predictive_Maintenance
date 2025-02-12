import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
from Models.CNN_LSTM import CNN_LSTM, EarlyStopping, CNN_LSTM_classification
from Models.CNN_Transformer import CNN_Transformer, CNN_Transformer_Classifier
from Models.CNNandLSTM_model import CNN_Model, LSTM_Model, CNN_Classifier, LSTM_Classifier
from utils import *

parser = argparse.ArgumentParser(description="Predictive Maintenance")

parser.add_argument("--train_data_path", type=str, default="./CMAPSSdata/train_FD003.txt", help="Training data path.")
parser.add_argument("--test_data_path", type=str, default="./CMAPSSdata/test_FD003.txt", help="Testing data path")
parser.add_argument("--rul_data_path", type=str, default="./CMAPSSdata/RUL_FD003.txt", help="Rul data path.")
parser.add_argument("--window_size", type=int, default=30, help="define your window's size.")
parser.add_argument("--num_epochs", type=int, default=300, help="Training epochs.")
parser.add_argument("--regression_model_save_path", type=str, default="./Models/best_CNN_Transformer_model.pth", help="model saving path.")
parser.add_argument("--classification_model_save_path", type=str, default="./Models/CNN_LSTM_Classification_model.pth", help="Classification model saving path.")

args = parser.parse_args()

def train_regression(num_epochs, model, optimizer, criterion, mae_criterion, train_loader, val_loader):
    train_losses = []
    val_losses = []
    train_mae = []
    val_mae = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        epoch_train_mae = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 2, 1))
            # outputs = model(inputs) # LSTM_Model
            loss = criterion(outputs.squeeze(), targets)
            mae_loss = mae_criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            epoch_train_mae += mae_loss.item()
    
        train_loss /= len(train_loader)

        train_losses.append(train_loss)
        train_mae.append(epoch_train_mae / len(train_loader))
        # print(f"Epoch {epoch + 1}: Training_loss:{train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        epoch_val_mae = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs.permute(0, 2, 1))
                # outputs = model(inputs) # LSTM_Model
                loss = criterion(outputs.squeeze(), targets)
                mae_loss = mae_criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
                epoch_val_mae += mae_loss.item()

            val_loss /= len(val_loader)

            val_losses.append(val_loss)
            val_mae.append(epoch_val_mae / len(val_loader))
            print(f"Epoch {epoch + 1}: Training_loss:{train_loss:.4f}, Val_loss:{val_loss}")

    plot_loss(train_losses, val_losses, train_mae, val_mae)

    torch.save(model.state_dict(), args.regression_model_save_path)

def train_classification(num_epochs, model, optimizer, criterion, train_loader, val_loader):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 计算训练准确率
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 计算验证集损失
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs)

    torch.save(model.state_dict(), args.classification_model_save_path)

def predict_regression(test_data, device, model, model_path):
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        outputs = model(test_tensor.permute(0, 2, 1))

    return outputs.cpu().numpy().reshape(-1)

def predict_regression_LSTM(test_data, device, model, model_path):
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        outputs = model(test_tensor)

    return outputs.cpu().numpy().reshape(-1)

def predict_classification(test_data, label_data, device, model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    y_true = []
    y_pred_prob = []
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    label_tensor = torch.tensor(label_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(test_tensor)
        y_pred_prob_batch = torch.sigmoid(outputs)
        y_pred_prob.extend(y_pred_prob_batch.cpu().numpy())
        y_true.extend(label_tensor.cpu().numpy())

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob).flatten()
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)
    return y_true, y_pred, y_pred_prob

if __name__ == "__main__":
    train_data, test_data, rul_data = raw_data_readin(args.train_data_path, args.test_data_path, args.rul_data_path)
    # print(train_data.shape)

    train_scaled_data, test_scaled_data= raw_data_processing(train_data, test_data)

    train_df, train_rul = train_data_labeling(train_data)
    test_df, test_rul = test_data_labeling(test_data, rul_data)

    processed_train_data, processed_train_targets = create_regression_time_windows(train_df, args.window_size, 1)
    processed_test_data, true_rul = create_regression_time_windows(test_df, args.window_size, 1)
    processed_test_data_L, true_rul_L = create_regression_time_windows_2(test_df, args.window_size, 1)
    # print(processed_train_data.shape)
    processed_train_data, processed_val_data, processed_train_targets, processed_val_targets = train_test_split(processed_train_data,
                                                                                                            processed_train_targets,
                                                                                                            test_size = 0.2,
                                                                                                            random_state = 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_data = torch.tensor(processed_train_data, dtype=torch.float32).to(device)
    train_targets = torch.tensor(processed_train_targets, dtype=torch.float32).to(device)

    val_data = torch.tensor(processed_val_data, dtype=torch.float32).to(device)
    val_targets = torch.tensor(processed_val_targets, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    input_channels = processed_train_data.shape[2]
    seq_length = processed_train_data.shape[1]

    num_epochs = args.num_epochs
    model = CNN_LSTM(input_channels, seq_length).to(device)
    model_2 = CNN_Transformer(input_channels, seq_length).to(device)
    model_3 = CNN_Model((input_channels, seq_length)).to(device)
    classification_1 = CNN_LSTM_classification(input_channels, seq_length).to(device)
    classification_2 = CNN_Transformer_Classifier(input_channels, seq_length).to(device)
    classification_3 = CNN_Classifier((input_channels, seq_length)).to(device)
    classification_4 = LSTM_Classifier(input_channels).to(device)
    model_4 = LSTM_Model(input_channels).to(device)

    criterion = nn.MSELoss()
    criterion_2 = nn.BCELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.RMSprop(model_2.parameters(), lr=0.001)
    optimizer_2 = optim.Adam(classification_4.parameters(), lr=0.001)

    # Regression training
    # train_regression(num_epochs, model_2, optimizer, criterion, mae_criterion, train_loader, val_loader)

    # rul_pred1 = predict_regression(processed_test_data_L, device, model, "./Models/CNN_LSTM_Regression_model.pth")
    # rul_pred2 = predict_regression(processed_test_data_L, device, model_2, "./Models/CNN_Transformer_Regression_model.pth")
    # rul_pred3 = predict_regression(processed_test_data_L, device, model_3, "./Models/CNN_Regression_model.pth")
    # rul_pred4 = predict_regression_LSTM(processed_test_data_L, device, model_4, "./Models/LSTM_Regression_model.pth")
    # d1 = to_dataframe(true_rul_L, rul_pred1, rul_pred2, rul_pred3, rul_pred4)
    # # # print(d1)
    # d1.to_csv("./predicted_data.csv", header=False, index=False)
    # rmse,mse,mae,mape=compute_regression_metrics(true_rul, rul_pred2)
    # plot_result(true_rul_L, rul_pred1)
    # plot_error(true_rul_L, rul_pred1)

    # For Classification
    # processed_train_data2, processed_train_targets_labels2 = create_time_windows_classification2(train_df, args.window_size, 1)
    # processed_test_data2, true_rul_labels2 = create_time_windows_classification2(test_df, args.window_size, 1)
    # processed_test_data_L2, true_rul_labels_L2 = create_time_windows_classification(test_df, args.window_size, 1)

    # X_tensor = torch.tensor(processed_train_data2, dtype=torch.float32)
    # y_tensor = torch.tensor(processed_train_targets_labels2, dtype=torch.float32).unsqueeze(1)  # 变成 (batch_size, 1)

    # train_size = int(0.8 * len(X_tensor))
    # val_size = len(X_tensor) - train_size
    # Train_dataset, Val_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, val_size])
    # Train_loader = DataLoader(Train_dataset, batch_size=200, shuffle=True)
    # Val_loader = DataLoader(Val_dataset, batch_size=200, shuffle=False)
    # train_classification(num_epochs, classification_4, optimizer_2, criterion_2, Train_loader, Val_loader)
    # y_true, y_pred, y_pred_prob = predict_classification(processed_test_data_L2, true_rul_labels_L2, device, classification_2, "./Models/CNN_Transformer_Classification_model.pth")
    # plot_classification_score(y_true, y_pred, y_pred_prob)