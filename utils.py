import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix, roc_curve, auc

def raw_data_readin(train_data_path, test_data_path, rul_data_path):
    train_data = pd.read_csv(train_data_path, sep = "\s+", header = None)
    test_data = pd.read_csv(test_data_path, sep= "\s+", header= None)
    test_rul_df = pd.read_csv(rul_data_path, sep="\s+", header=None)
    columns = ['engine_number', 'time_in_cycles'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]

    train_data.columns = columns
    train_data = train_data.iloc[0:].reset_index(drop=True)
    train_data = train_data.drop(index=0).reset_index(drop=True)

    test_data.columns = columns
    test_data = test_data.iloc[0:].reset_index(drop=True)
    test_data = test_data.drop(index=0).reset_index(drop=True)

    columns_to_be_dropped = ['op_setting_3', 'sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_19']
    train_df_dropped = train_data.drop(columns=columns_to_be_dropped)
    test_df_dropped = test_data.drop(columns=columns_to_be_dropped)
    return train_df_dropped, test_df_dropped, test_rul_df

def raw_data_processing(train_df, test_df):
    features_to_normalize = train_df.columns[2:]

    train_df[features_to_normalize] = train_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')
    test_df[features_to_normalize] = test_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')

    scaler = MinMaxScaler()
    train_df[features_to_normalize] = scaler.fit_transform(train_df[features_to_normalize])
    test_df[features_to_normalize] = scaler.transform(test_df[features_to_normalize])

    return train_df, test_df

# train dataset labeling
def train_data_labeling(train_df):
    train_df['time_in_cycles'] = pd.to_numeric(train_df['time_in_cycles'], errors='coerce')
    rul = pd.DataFrame(train_df.groupby('engine_number')['time_in_cycles'].max()).reset_index()
    rul.columns = ['engine_number', 'max_time_in_cycles']
    train_df = train_df.merge(rul, on='engine_number')

    train_df['RUL'] = train_df['max_time_in_cycles'] - train_df['time_in_cycles']
    train_df.drop('max_time_in_cycles', axis=1, inplace=True)

    train_df['RUL'] = train_df['RUL'].apply(lambda x: min(x, 130))

    time_window = 50
    train_df['label'] = train_df['RUL'].apply(lambda x: 1 if x > time_window else 0)
    trian_rul = pd.DataFrame(train_df.groupby('engine_number')['time_in_cycles'].max()).reset_index()
    return train_df, trian_rul

# test dataset labeling
def test_data_labeling(test_df, test_rul_df):
    time_window = 50

    test_df['time_in_cycles'] = pd.to_numeric(test_df['time_in_cycles'], errors='coerce')
    rul = pd.DataFrame(test_df.groupby('engine_number')['time_in_cycles'].max()).reset_index()
    rul.columns = ['engine_number', 'max_time_in_cycles']
    test_df = test_df.merge(rul, on='engine_number')

    test_df['RUL'] = test_df['max_time_in_cycles'] - test_df['time_in_cycles']
    test_df.drop('max_time_in_cycles', axis=1, inplace=True)

    test_rul_df.columns = ['RUL']
    test_rul_df['RUL'] = pd.to_numeric(test_rul_df['RUL'], errors='coerce')

    rul["RUL"] = test_rul_df["RUL"].values
    test_df = test_df.merge(rul[['engine_number', 'max_time_in_cycles', 'RUL']], on='engine_number')

    test_df['RUL'] = test_df['RUL_x'] + test_df['RUL_y']
    test_df['RUL'] = test_df['RUL'].apply(lambda x: min(x, 130))
    test_df.drop(columns=[ 'RUL_x', 'max_time_in_cycles', 'RUL_y'], inplace=True)

    test_df['label'] = test_df['RUL'].apply(lambda x: 1 if x > time_window else 0)

    test_rul = pd.DataFrame(test_df.groupby('engine_number')['time_in_cycles'].max()).reset_index()
    return test_df, test_rul

# create windows for regression
def create_regression_time_windows(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(0, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end, 2:-2].values)
            y.append(engine_df.iloc[end - 1, -2])
    return np.array(X), np.array(y)

def create_regression_time_windows_2(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(len(engine_df) - window_size , len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end, 2:-2].values)
            y.append(engine_df.iloc[end - 1, -2])
    return np.array(X), np.array(y)

def create_time_windows_classification(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(len(engine_df) - window_size , len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end, 2:-2].values)
            y.append(engine_df.iloc[end - 1, -1]) # thats for classification
    return np.array(X), np.array(y)

def create_time_windows_classification2(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(0, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end, 2:-2].values)
            y.append(engine_df.iloc[end - 1, -1])
    return np.array(X), np.array(y)


def classification_array(rul_array):
    class_array = np.where(rul_array > 50, 1, 0)
    return class_array

def calculate_classification_accuracy(target_array, pred_array):
    TN = np.sum(np.logical_and(target_array == 0, pred_array == 0))
    TP = np.sum(np.logical_and(target_array == 1, pred_array == 1))
    FN = np.sum(np.logical_and(target_array == 1, pred_array == 0))
    FP = np.sum(np.logical_and(target_array == 0, pred_array == 1))
    class_sum = TN + TP + FP + FN
    Accuracy = (TN + TP) / class_sum
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
    print("Accuracy:", Accuracy)
    print("F1_Score:", F1_Score)


# save as a csv file
def save_as_csv(dataframe, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    dataframe.to_csv(f"./{folder_path}/train_completed_data.csv", header=False, index=False)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def compute_regression_metrics(true_rul,rul_pred):
    # Calculate errors
    rmse = np.sqrt(mean_squared_error(true_rul, rul_pred))
    mse = mean_squared_error(true_rul, rul_pred)
    mae = mean_absolute_error(true_rul, rul_pred)
    mape = mean_absolute_percentage_error(true_rul, rul_pred)

    # Print the errors
    print("RMSE: ", rmse)
    print("MSE: ", mse)
    print("MAE: ", mae)
    print("MAPE: ", mape)

    return rmse,mse,mae,mape

def to_dataframe(array1, array2, array3, array4, array5):
    # 按行合并
    df = pd.DataFrame(np.vstack((array1, array2, array3, array4, array5)), index=['Row1', 'Row2', "Row3", "Row4", "Row5"]).T
    df.columns = ['Real', 'CNN_LSTM', "CNN_Transformer", "CNN", "LSTM"]
    return df


# draw MSE and MAE plot
def plot_loss(train_losses, val_losses, train_mae, val_mae):
    plt.figure(figsize=(12, 5))

    # 绘制 MSE 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses, label='Val MSE')
    plt.title('Model Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc='upper right')

    # 绘制 MAE 曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_mae, label='Train MAE')
    plt.plot(val_mae, label='Val MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_result(true_rul_L, rul_pred):
    # 设置 Matplotlib 风格
    plt.style.use('default')  # 取消 seaborn 样式
    plt.figure(figsize=(20, 5))  # 设置画布大小

    # 画折线图
    plt.plot(np.arange(0, true_rul_L.shape[0]), true_rul_L, color='green', label='actual', linewidth=2)
    plt.plot(np.arange(0, rul_pred.shape[0]), rul_pred, color='steelblue', label='predictions', linewidth=2)

    # 设置图例
    plt.legend()

    # 设置标签
    plt.xlabel('Engine nr')
    plt.ylabel('RUL')

    # 显示图形
    plt.show()

def plot_error(true_rul_L, rul_pred):
    concat = np.vstack((rul_pred,true_rul_L))
    sorted_concat = np.sort(concat, axis=None)

    sorted_concat = sorted_concat.reshape(concat.shape)

    true_rul_L = true_rul_L.astype(float)
    rul_pred = rul_pred.astype(float)

    # Sort the data based on true RUL values
    indices = np.argsort(true_rul_L)
    sorted_true_rul = true_rul_L[indices]
    sorted_predicted_rul = rul_pred[indices]

    plt.figure(figsize=(12, 6))

    plt.scatter(range(len(sorted_true_rul)), sorted_true_rul, color='red', label='Actual RUL')
    plt.scatter(range(len(sorted_predicted_rul)), sorted_predicted_rul, color='blue', label='Predicted RUL')

    for i in range(len(sorted_true_rul)):
        plt.plot([i, i], [sorted_true_rul[i], sorted_predicted_rul[i]], color='gray', linestyle='--')

    plt.xlabel('Index')
    plt.ylabel('RUL')
    plt.title('Actual vs Predicted RUL in asceneding order')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.show()

def plot_classification_score(y_true, y_pred, y_pred_prob):
    report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], output_dict=True)
    precision = report['Class 1']['precision']
    recall = report['Class 1']['recall']
    f1_score = report['Class 1']['f1-score']
    accuracy = report['accuracy']

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1_score:.4f}')
    print(f'Accuracy: {accuracy:.4f}')

    print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))

    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # 计算并绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()