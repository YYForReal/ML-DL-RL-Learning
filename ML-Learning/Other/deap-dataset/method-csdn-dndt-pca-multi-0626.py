import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn import tree
import pickle
import random
from torch.utils.data import DataLoader, TensorDataset
import time
import datetime
import wandb
import scipy.stats as stats

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)

# 确定设备（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 示例数据
# X.shape: (1280, 40, 8064)
# y.shape: (1280,)


def extract_features(X):
    features = []
    print(f"Extracting features from {X.shape[0]} samples...")
    for sample in X:
        sample_features = []
        for channel in sample:
            mean = np.mean(channel)
            std_dev = np.std(channel)
            skewness = stats.skew(channel)
            kurtosis = stats.kurtosis(channel)
            max_val = np.max(channel)
            min_val = np.min(channel)
            range_val = max_val - min_val
            sample_features.extend(
                [
                    mean,
                    std_dev,
                    skewness,
                    kurtosis,
                    max_val,
                    min_val,
                    range_val,
                ]
            )
        features.append(sample_features)
    return np.array(features)


# 定义加载DEAP数据集的函数
def load_deap_data(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")
    return data


# 提取数据和标签
def get_data_and_labels(label_idx):
    data_dir = "data_preprocessed_python"
    X_list = []
    y_list = []
    for i in range(1, 33):  # 32位参与者
        data_file = os.path.join(data_dir, f"s{i:02d}.dat")
        subject = load_deap_data(data_file)
        X_list.append(subject["data"])
        y_list.append(subject["labels"][:, label_idx])
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    y_binary = np.where(y >= 5, 1, 0)
    return X, y_binary


# 特征提取并预处理
def preprocess_data(X, y):
    print(f"Original data shape: {X.shape}")  # Original data shape: (1280, 40, 8064)
    X = X[:, :32, :]
    X_features = extract_features(X)
    print(f"Extracted features shape: {X_features.shape}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    print(f"Scaled data shape: {X_scaled.shape}")
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
        self.temperature = torch.tensor(temperature, device=device)
        self.cut_points_list = [
            torch.rand([i], requires_grad=True, device=device) for i in num_cut
        ]
        self.leaf_score = torch.rand(
            [self.num_leaf, self.num_class], requires_grad=True, device=device
        )
        self.optimizer = torch.optim.Adam(
            self.cut_points_list + [self.leaf_score] + [self.temperature], lr=0.01
        )

    def torch_kron_prod(self, a, b):
        res = torch.einsum("ij,ik->ijk", [a, b])
        res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
        return res

    def torch_bin(self, x, cut_points, temperature):
        D = cut_points.shape[0]
        W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
        W = W.to(device)  # 将W迁移到GPU
        cut_points, _ = torch.sort(cut_points)
        b = torch.cumsum(
            torch.cat([torch.zeros([1], device=device), -cut_points], 0), 0
        )
        h = torch.matmul(x, W) + b
        h = h / temperature
        res = F.softmax(h, dim=1)
        return res

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

    def fit(self, dataloader, num_epochs, start_index=0):
        epoch_losses = []
        epoch_accuracies = []
        for epoch in range(num_epochs):
            all_labels = []
            all_preds = []
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                self.optimizer.zero_grad()
                y_pred = self.nn_decision_tree(x_batch)
                loss = F.cross_entropy(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                all_labels.extend(y_batch.cpu().numpy())
                all_preds.extend(torch.argmax(y_pred, axis=1).cpu().detach().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy)
            wandb.log(
                {
                    "Loss": loss.item(),
                    "Accuracy": accuracy,
                    "F1": f1,
                    "Epoch": epoch + start_index,
                }
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
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = self.nn_decision_tree(x_batch)
                all_preds.append(torch.argmax(y_pred, axis=1).cpu().detach().numpy())
                all_labels.append(y_batch.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_labels)


def train_and_evaluate(min_cut, X_train, X_test, y_train, y_test, num_epochs=2000):
    feature_num = X_train.shape[1]
    if feature_num > 12:
        num_trees = 10
        features_per_tree = 10
        trees = []
        for _ in range(num_trees):
            selected_features = np.random.choice(
                feature_num, features_per_tree, replace=False
            )
            tree = DNDT([1] * features_per_tree, num_class=2, temperature=0.1).to(
                device
            )
            trees.append((tree, selected_features))

        train_loader = create_dataloader(X_train, y_train, batch_size=32)
        test_loader = create_dataloader(X_test, y_test, batch_size=32)

        all_train_losses = []
        all_train_accuracies = []

        for tree, selected_features in trees:
            X_train_selected = X_train[:, selected_features]
            X_test_selected = X_test[:, selected_features]
            train_loader_selected = create_dataloader(
                X_train_selected, y_train, batch_size=32
            )
            test_loader_selected = create_dataloader(
                X_test_selected, y_test, batch_size=32
            )

            losses, accuracies = tree.fit(train_loader_selected, num_epochs)
            all_train_losses.append(losses)
            all_train_accuracies.append(accuracies)

        def ensemble_predict(dataloader, trees):
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for x_batch, y_batch in dataloader:
                    batch_preds = []
                    for tree, selected_features in trees:
                        x_batch_selected = x_batch[:, selected_features].to(device)
                        y_pred = tree.nn_decision_tree(x_batch_selected)
                        batch_preds.append(
                            torch.argmax(y_pred, axis=1).cpu().detach().numpy()
                        )
                    ensemble_preds = stats.mode(batch_preds, axis=0)[0][0]
                    all_preds.append(ensemble_preds)
                    all_labels.append(y_batch.numpy())
            return np.concatenate(all_preds), np.concatenate(all_labels)

        y_pred_train, y_train_labels = ensemble_predict(train_loader, trees)
        y_pred_test, y_test_labels = ensemble_predict(test_loader, trees)

        train_accuracy = accuracy_score(y_train_labels, y_pred_train)
        test_accuracy = accuracy_score(y_test_labels, y_pred_test)
        f1_train = f1_score(y_train_labels, y_pred_train)
        f1_test = f1_score(y_test_labels, y_pred_test)

        print("Training Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        print("Training F1 Score:", f1_train)
        print("Test F1 Score:", f1_test)

        print(classification_report(y_test_labels, y_pred_test))
    else:
        model = DNDT([1] * min_cut, num_class=2, temperature=0.1).to(device)
        train_loader = create_dataloader(X_train, y_train, batch_size=32)
        test_loader = create_dataloader(X_test, y_test, batch_size=32)
        losses, accuracies = model.fit(train_loader, num_epochs)
        y_pred_train, _ = model.predict(train_loader)
        y_pred_test, _ = model.predict(test_loader)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        f1_train = f1_score(y_train, y_pred_train)
        f1_test = f1_score(y_test, y_pred_test)

        print("Training Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        print("Training F1 Score:", f1_train)
        print("Test F1 Score:", f1_test)

        print(classification_report(y_test, y_pred_test))


if __name__ == "__main__":
    wandb.init(project="EEG_classification")
    label_idx = 1  # 更改为你要使用的标签索引
    X, y = get_data_and_labels(label_idx)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    min_cut = 32
    start_time = time.time()
    train_and_evaluate(min_cut, X_train, X_test, y_train, y_test, num_epochs=2000)
    end_time = time.time()
    duration = end_time - start_time
    print("Training and evaluation time: ", duration)
