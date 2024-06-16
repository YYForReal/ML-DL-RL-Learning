import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn import tree
import pickle
import random
import time
import datetime
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from functools import reduce

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
    X_scaled = X_scaled.reshape(X.shape[0], X.shape[1], -1)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 创建数据加载器
def create_dataloader(X, y, batch_size=32):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 定义DNDT模型
class DNDT(nn.Module):
    def __init__(self, num_cut, num_class, temperature):
        super(DNDT, self).__init__()
        self.num_cut = num_cut
        self.num_leaf = np.prod(np.array(num_cut) + 1)
        self.num_class = num_class
        self.temperature = torch.tensor(temperature)
        self.cut_points_list = [torch.rand([i], requires_grad=True) for i in num_cut]
        self.leaf_score = torch.rand(
            [self.num_leaf, self.num_class], requires_grad=True
        )
        self.optimizer = torch.optim.Adam(
            self.cut_points_list + [self.leaf_score] + [self.temperature], lr=0.01
        )

    # 计算克罗内克积
    def torch_kron_prod(self, a, b):
        res = torch.einsum("ij,ik->ijk", [a, b])
        res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
        return res

    # 软分箱算法
    def torch_bin(self, x, cut_points, temperature):
        D = cut_points.shape[0]
        W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
        cut_points, _ = torch.sort(cut_points)
        b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0), 0)
        h = torch.matmul(x.unsqueeze(2), W.unsqueeze(0)) + b
        h = h / temperature
        res = F.softmax(h, dim=2).squeeze()
        return res

    # 建树
    def nn_decision_tree(self, x):
        leaf = reduce(
            self.torch_kron_prod,
            map(
                lambda z: self.torch_bin(x[:, z[0] : z[0] + 1], z[1], self.temperature),
                enumerate(self.cut_points_list),
            ),
        )
        return torch.matmul(leaf, self.leaf_score)

    def forward(self, x):
        return self.nn_decision_tree(x)

    def fit(self, dataloader, num_epochs):
        epoch_losses = []
        epoch_accuracies = []
        for epoch in range(num_epochs):
            all_labels = []
            all_preds = []
            for x_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.nn_decision_tree(x_batch)
                loss = F.cross_entropy(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                all_labels.extend(y_batch.numpy())
                all_preds.extend(torch.argmax(y_pred, axis=1).detach().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy)
            wandb.log(
                {"Loss": loss.item(), "Accuracy": accuracy, "F1": f1, "Epoch": epoch}
            )
            if epoch % (num_epochs // 10) == 0:
                print(
                    f"Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}, F1: {f1}"
                )
        print("Training complete.")
        return epoch_losses, epoch_accuracies

    def predict(self, dataloader):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                y_pred = self.nn_decision_tree(x_batch)
                all_preds.append(torch.argmax(y_pred, axis=1).detach().numpy())
                all_labels.append(y_batch.numpy())
        return np.concatenate(all_preds), np.concatenate(all_labels)


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


def train_and_evaluate_cnn(
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

    # 记录训练损失
    loss_history = []

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"
        ):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Loss: {epoch_loss:.4f}")

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

    accuracy = 100 * correct / total
    print(f"测试集准确率: {accuracy:.2f}%")
    print("分类报告:")
    report = classification_report(all_labels, all_predictions, output_dict=True)
    print(classification_report(all_labels, all_predictions))
    print("混淆矩阵:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    # 保存模型和结果
    save_model(model, f"CNN_{label_name}_model.pth")
    with open(
        os.path.join("output", f"CNN_{label_name}_classification_report.txt"), "w"
    ) as f:
        f.write(f"测试集准确率: {accuracy:.2f}%\n")
        f.write("分类报告:\n")
        f.write(classification_report(all_labels, all_predictions))
        f.write("混淆矩阵:\n")
        f.write(np.array2string(cm))

    # 绘制训练损失曲线
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{label_name.capitalize()} Training Loss")
    plt.legend()
    plt.savefig(os.path.join("output", f"CNN_{label_name}_training_loss.png"))
    # plt.show()

    # 可视化分类报告
    metrics = ["precision", "recall", "f1-score"]
    for metric in metrics:
        plt.figure()
        for i in range(num_classes):
            plt.bar(i, report[str(i)][metric], label=f"Class {i}")
        plt.xlabel("Classes")
        plt.ylabel(metric.capitalize())
        plt.title(f"{label_name.capitalize()} {metric.capitalize()}")
        plt.legend()
        plt.savefig(os.path.join("output", f"CNN_{label_name}_{metric}.png"))
        # plt.show()

    # 可视化混淆矩阵
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{label_name.capitalize()} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, range(len(cm)), rotation=45)
    plt.yticks(tick_marks, range(len(cm)))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join("output", f"CNN_{label_name}_confusion_matrix.png"))
    # plt.show()

    return loss_history, accuracy


# 定义保存模型函数
def save_model(model, filename):
    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("output", filename))


# 训练和评估函数
def train_and_evaluate_dndt(min_cut, X_train, X_test, y_train, y_test, num_epochs=2000):
    num_cut = [5] * min(X_train.shape[1], min_cut)
    dndt = DNDT(num_cut, num_class=2, temperature=0.1)

    # 创建数据加载器
    train_loader = create_dataloader(X_train, y_train, batch_size=32)
    test_loader = create_dataloader(X_test, y_test, batch_size=32)

    # 训练模型
    print(f"Training DNDT with min_cut={min_cut}")
    start_time = time.time()
    losses, accuracies = dndt.fit(train_loader, num_epochs=1000)
    print(
        f"Training time for DNDT with min_cut={min_cut} for 1000 epochs: {time.time() - start_time} seconds"
    )

    # 预测并记录1000个epoch的结果
    y_pred_1000, y_true_1000 = dndt.predict(test_loader)
    print(f"Classification Report for DNDT with min_cut={min_cut} at 1000 epochs:")
    print(classification_report(y_true_1000, y_pred_1000))
    wandb.log(
        {
            "Classification Report DNDT 1000": classification_report(
                y_true_1000, y_pred_1000, output_dict=True
            )
        }
    )

    # 继续训练到2000个epoch
    losses_2000, accuracies_2000 = dndt.fit(train_loader, num_epochs=1000)
    losses.extend(losses_2000)
    accuracies.extend(accuracies_2000)
    print(
        f"Training time for DNDT with min_cut={min_cut} for 2000 epochs: {time.time() - start_time} seconds"
    )

    # 预测并记录2000个epoch的结果
    y_pred_2000, y_true_2000 = dndt.predict(test_loader)
    print(f"Classification Report for DNDT with min_cut={min_cut} at 2000 epochs:")
    print(classification_report(y_true_2000, y_pred_2000))
    wandb.log(
        {
            "Classification Report DNDT 2000": classification_report(
                y_true_2000, y_pred_2000, output_dict=True
            )
        }
    )

    return losses, accuracies


# 训练和评估函数
def train_and_evaluate_tree(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()

    start_time = time.time()
    clf.fit(X_train, y_train)
    print(f"Training time for Decision Tree: {time.time() - start_time} seconds")

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Report for Decision Tree:")
    print(classification_report(y_test, y_pred))
    wandb.log(
        {
            "Classification Report Tree": classification_report(
                y_test, y_pred, output_dict=True
            )
        }
    )

    return accuracy


if __name__ == "__main__":
    wandb.init(project="DNDT-Comparison")

    label_idx = 0  # 使用第一个标签（效价）

    # 加载数据
    X, y = get_data_and_labels(0, label_idx)

    # 输出原始数据形状
    print(f"Original data shape: {X.shape}")

    # 从时间序列数据中提取特征
    # X = extract_features(X)

    # 输出特征提取后的数据形状
    print(f"Feature extracted data shape: {X.shape}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # 输出划分后的数据形状
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    min_cuts = [3, 5]  # 设定不同的min_cut值进行比较

    all_losses = {}
    all_accuracies = {}

    for min_cut in min_cuts:
        wandb.run.name = f"DNDT_min_cut_{min_cut}"
        wandb.run.save()
        losses, accuracies = train_and_evaluate_dndt(
            min_cut, X_train, X_test, y_train, y_test, num_epochs=2000
        )
        all_losses[min_cut] = losses
        all_accuracies[min_cut] = accuracies

    # 比较传统决策树
    wandb.run.name = "Decision_Tree"
    wandb.run.save()
    tree_accuracy = train_and_evaluate_tree(X_train, X_test, y_train, y_test)

    # 比较CNN模型
    wandb.run.name = "CNN"
    wandb.run.save()
    cnn_losses, cnn_accuracy = train_and_evaluate_cnn(
        X_train,
        X_test,
        y_train,
        y_test,
        "CNN",
        num_epochs=2000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    wandb.finish()
