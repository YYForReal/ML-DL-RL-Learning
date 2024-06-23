import os
import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn import tree
import pickle
from tqdm import tqdm
import wandb
import datetime
import time

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)


# 定义加载DEAP数据集的函数
def load_deap_data(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")
    return data


# 提取数据和标签
def get_data_and_labels(participant_id, label_idx):
    data_dir = "data_preprocessed_python"
    if participant_id == 0:
        X_list = []
        y_list = []
        for i in range(1, 33):
            data_file = os.path.join(data_dir, f"s{i:02d}.dat")
            subject = load_deap_data(data_file)
            X_list.append(subject["data"])
            y_list.append(subject["labels"][:, label_idx])
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
    else:
        data_file = os.path.join(data_dir, f"s{participant_id:02d}.dat")
        subject = load_deap_data(data_file)
        X = subject["data"]
        y = subject["labels"][:, label_idx]

    y_binary = np.where(y >= 5, 1, 0)
    return X, y_binary


# 特征缩放并划分训练集和测试集
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
    X_scaled = X_scaled.reshape(X.shape[0], X.shape[1], -1)  # 保持原始的40通道形状
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=num_channels, out_channels=64, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.fc1 = nn.Linear(
            256 * 1008, 128
        )  # 根据最后一层卷积层的输出维度调整输入大小
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # 展平操作
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 训练和评估模型
def train_and_evaluate_model(
    X_train, X_test, y_train, y_test, label_name, num_epochs, device
):
    num_channels = X_train.shape[1]
    num_classes = 2
    model = CNN(num_channels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.long).to(device),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(y_test, dtype=torch.long).to(device),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    wandb.init(project=f"DEAP-{label_name}", name=f"CNN_{label_name}")

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []
        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"
        ):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.argmax(outputs, axis=1).cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = accuracy_score(all_labels, all_predictions)
        train_f1 = f1_score(all_labels, all_predictions)
        wandb.log(
            {
                "Train Loss": epoch_loss,
                "Train Accuracy": train_accuracy,
                "Train F1": train_f1,
                "Epoch": epoch,
            }
        )

        # 测试模型
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        test_accuracy = accuracy_score(all_labels, all_predictions)
        test_f1 = f1_score(all_labels, all_predictions)
        wandb.log({"Test Accuracy": test_accuracy, "Test F1": test_f1, "Epoch": epoch})

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

    # 打印分类报告
    print("分类报告:")
    classification_rep = classification_report(
        all_labels, all_predictions, output_dict=True
    )
    print(classification_report(all_labels, all_predictions))
    print("混淆矩阵:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    # 保存分类报告和混淆矩阵到文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    with open(
        f"output/CNN_classification_report_{label_name}_{timestamp}.txt", "w"
    ) as f:
        f.write(f"Label: {label_name}\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_predictions))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    wandb.finish()


# 训练和评估传统决策树模型
def train_and_evaluate_decision_tree(X_train, X_test, y_train, y_test, label_name):
    # 展平数据以适应决策树输入要求
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    clf = tree.DecisionTreeClassifier()
    start_time = time.time()
    clf = clf.fit(X_train_reshaped, y_train)
    y_pred = clf.predict(X_test_reshaped)
    training_time = time.time() - start_time

    # 打印分类报告
    print("Classification Report (Decision Tree):")
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 保存分类报告和混淆矩阵到文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    with open(
        f"output/Decision_Tree_classification_report_{label_name}_{timestamp}.txt", "w"
    ) as f:
        f.write(f"Label: {label_name}\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write(f"\nTraining Time: {training_time} seconds\n")

    # 可视化决策树
    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, filled=True)
    plt.title(f"Decision Tree for {label_name}")
    plt.savefig(f"output/Decision_Tree_{label_name}_{timestamp}.png")
    plt.show()


# 选择要训练的标签
labels = ["valence", "arousal", "dominance", "liking"]
label_indices = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}

# 使用argparse定义可选参数
parser = argparse.ArgumentParser(
    description="Train and evaluate models on DEAP dataset"
)
parser.add_argument(
    "--participant",
    type=int,
    default=0,
    help="Participant ID (default: 0 for all participants)",
)
parser.add_argument(
    "--num_epochs", type=int, default=1000, help="Number of epochs (default: 1000)"
)
parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
args = parser.parse_args()

participant_id = args.participant
num_epochs = args.num_epochs
device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

for label_name in labels:
    print(f"训练和评估 {label_name} 标签...")
    X, y = get_data_and_labels(participant_id, label_indices[label_name])
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # 训练和评估CNN模型
    # train_and_evaluate_model(
    #     X_train, X_test, y_train, y_test, label_name, num_epochs, device
    # )

    # 训练和评估传统决策树模型
    train_and_evaluate_decision_tree(X_train, X_test, y_train, y_test, label_name)

print("所有标签的模型训练和评估已完成。")
