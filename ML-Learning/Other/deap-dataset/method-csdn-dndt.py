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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt

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
    data_file = os.path.join(data_dir, f"s{participant_id:02d}.dat")
    subject = load_deap_data(data_file)
    X = subject["data"]
    y = subject["labels"][:, label_idx]

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

    def fit(self, dataloader, writer, num_epochs):
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
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Accuracy/train", accuracy, epoch)
            writer.add_scalar("F1/train", f1, epoch)
            if epoch % (num_epochs // 10) == 0:
                print(
                    f"Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}, F1: {f1}"
                )
        print("Training complete.")

    def predict(self, dataloader):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                y_pred = self.nn_decision_tree(x_batch)
                all_preds.append(torch.argmax(y_pred, axis=1).detach().numpy())
                all_labels.append(y_batch.numpy())
        return np.concatenate(all_preds), np.concatenate(all_labels)


if __name__ == "__main__":
    # 数据准备
    participant_id = 0  # 使用第一个参与者的数据
    label_idx = 1  # 使用第一个标签（效价）

    # 加载数据
    X, y = get_data_and_labels(participant_id, label_idx)

    # 输出原始数据形状
    print(f"Original data shape: {X.shape}")

    # 从时间序列数据中提取特征
    X = extract_features(X)

    # 输出特征提取后的数据形状
    print(f"Feature extracted data shape: {X.shape}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 创建数据加载器
    train_loader = create_dataloader(X_train, y_train, batch_size=32)
    test_loader = create_dataloader(X_test, y_test, batch_size=32)

    # 调整 num_cut 以降低内存需求
    num_cut = [5] * min(X_train.shape[1], 3)  # 降低每个特征的切分点数

    # 打印调试信息
    print(f"num_leaf: {np.prod(np.array(num_cut) + 1)}")

    # 搭建模型
    # dndt = DNDT(num_cut, num_class=2, temperature=0.1)
    # writer = SummaryWriter(log_dir=os.path.join("logs", "DNDT"))

    # # 训练模型
    # start_time = time.time()
    # dndt.fit(train_loader, writer, num_epochs=1000)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # # 预测
    # y_pred, y_true = dndt.predict(test_loader)
    # print("Classification Report (DNDT):")
    # print(classification_report(y_true, y_pred))

    # 关闭TensorBoard写入器
    # writer.close()

    # 可视化训练损失
    os.system("tensorboard --logdir=logs")

    # 与传统决策树比较
    clf = tree.DecisionTreeClassifier()

    start_time = time.time()

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("--- %s seconds ---" % (time.time() - start_time))
    print("Classification Report (Decision Tree):")
    print(classification_report(y_test, y_pred))

    # 画图
    plt.figure(figsize=(8, 8))
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        marker="o",
        s=50,
        cmap="summer",
        edgecolors="black",
    )
    plt.title("Scatter plot of Decision Tree classification")
    plt.show()
