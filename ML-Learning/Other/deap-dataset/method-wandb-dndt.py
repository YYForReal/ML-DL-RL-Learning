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

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)


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


# 使用PCA进行降维
def extract_features(X):
    pca = PCA(n_components=min(40, X.shape[2]))  # 保留最多40个主成分
    X_pca = np.array([pca.fit_transform(sample.T).T for sample in X])
    return X_pca


# 特征缩放并划分训练集和测试集
def preprocess_data(X, y):
    X_reshaped = X.reshape(X.shape[0], -1)  # 展平数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
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
        h = torch.matmul(x, W) + b
        h = h / temperature
        res = F.softmax(h, dim=1)
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

    def fit(self, train_loader, test_loader, num_epochs, stage, start_epoch=0):
        epoch_losses = []
        epoch_accuracies = []
        for epoch in range(num_epochs):
            all_labels = []
            all_preds = []
            for x_batch, y_batch in train_loader:
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
            current_epoch = start_epoch + epoch
            wandb.log(
                {
                    f"Train Loss_{stage}": loss.item(),
                    f"Train Accuracy_{stage}": accuracy,
                    f"Train F1_{stage}": f1,
                    "Epoch": current_epoch,
                },
                step=current_epoch,
            )

            # 在每个epoch结束时评估测试集
            test_labels = []
            test_preds = []
            for x_batch, y_batch in test_loader:
                y_pred = self.nn_decision_tree(x_batch)
                test_labels.extend(y_batch.numpy())
                test_preds.extend(torch.argmax(y_pred, axis=1).detach().numpy())

            test_accuracy = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds)
            wandb.log(
                {
                    f"Test Accuracy_{stage}": test_accuracy,
                    f"Test F1_{stage}": test_f1,
                    "Epoch": current_epoch,
                },
                step=current_epoch,
            )

            if current_epoch % (num_epochs // 10) == 0:
                print(
                    f"Epoch {current_epoch}, Loss: {loss.item()}, Accuracy: {accuracy}, F1: {f1}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}"
                )

        print(f"Training complete for stage {stage}.")
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


# 训练和评估函数
def train_and_evaluate(min_cut, X_train, X_test, y_train, y_test, num_epochs=2000):
    num_cut = [5] * min(X_train.shape[1], min_cut)  # 降低每个特征的切分点数
    dndt = DNDT(num_cut, num_class=2, temperature=0.1)

    # 创建数据加载器
    train_loader = create_dataloader(X_train, y_train, batch_size=32)
    test_loader = create_dataloader(X_test, y_test, batch_size=32)

    # 训练模型
    print(f"Training with min_cut={min_cut}")
    start_time = time.time()
    losses_1000, accuracies_1000 = dndt.fit(
        train_loader, test_loader, num_epochs=1000, stage="1000"
    )
    print(
        f"Training time for min_cut={min_cut} for 1000 epochs: {time.time() - start_time} seconds"
    )

    # 预测并记录1000个epoch的结果
    y_pred_1000, y_true_1000 = dndt.predict(test_loader)
    print(f"Classification Report for min_cut={min_cut} at 1000 epochs:")
    print(classification_report(y_true_1000, y_pred_1000))
    wandb.log(
        {
            f"Classification Report 1000": classification_report(
                y_true_1000, y_pred_1000, output_dict=True
            )
        }
    )

    # 继续训练到2000个epoch
    losses_2000, accuracies_2000 = dndt.fit(
        train_loader, test_loader, num_epochs=1000, stage="2000", start_epoch=1000
    )
    losses_1000.extend(losses_2000)
    accuracies_1000.extend(accuracies_2000)
    print(
        f"Training time for min_cut={min_cut} for 2000 epochs: {time.time() - start_time} seconds"
    )

    # 预测并记录2000个epoch的结果
    y_pred_2000, y_true_2000 = dndt.predict(test_loader)
    print(f"Classification Report for min_cut={min_cut} at 2000 epochs:")
    print(classification_report(y_true_2000, y_pred_2000))
    wandb.log(
        {
            f"Classification Report 2000": classification_report(
                y_true_2000, y_pred_2000, output_dict=True
            )
        }
    )

    return losses_1000, accuracies_1000


if __name__ == "__main__":
    label_idx = 0  # 使用第一个标签（效价）

    # 加载数据
    X, y = get_data_and_labels(label_idx)

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

    min_cuts = [3, 4, 5]  # 设定不同的min_cut值进行比较

    all_losses = {}
    all_accuracies = {}

    for min_cut in min_cuts:
        wandb.init(project=f"DEAP-{label_idx}", name=f"DNDT_min_cut_{min_cut}")
        losses, accuracies = train_and_evaluate(
            min_cut, X_train, X_test, y_train, y_test, num_epochs=2000
        )
        all_losses[min_cut] = losses
        all_accuracies[min_cut] = accuracies
        wandb.finish()
