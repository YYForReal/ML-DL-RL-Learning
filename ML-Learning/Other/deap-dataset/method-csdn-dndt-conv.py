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
import datetime
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


# 使用卷积神经网络进行特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, out_channels=64, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(256 * 1008, 256)  # 调整全连接层输入维度
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(f"After conv1 and pool: {x.shape}")  # 添加特征维度输出
        x = self.pool(F.relu(self.conv2(x)))
        print(f"After conv2 and pool: {x.shape}")  # 添加特征维度输出
        x = self.pool(F.relu(self.conv3(x)))
        print(f"After conv3 and pool: {x.shape}")  # 添加特征维度输出
        x = x.view(x.size(0), -1)  # 展平操作
        print(f"After flatten: {x.shape}")  # 添加特征维度输出
        x = self.dropout(F.relu(self.fc(x)))
        return x


# 特征缩放并划分训练集和测试集
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
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
    label_idx = 0  # 使用第一个标签（效价）

    # 加载数据
    X, y = get_data_and_labels(label_idx)

    # 输出原始数据形状
    print(f"Original data shape: {X.shape}")

    # 使用卷积神经网络提取特征
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(
        0, 2, 1
    )  # 变换形状为 (batch_size, channels, sequence_length)
    print(f"X_tensor shape: {X_tensor.shape}")
    feature_extractor = FeatureExtractor(input_channels=X_tensor.shape[1])
    print(feature_extractor)
    with torch.no_grad():
        features = feature_extractor(X_tensor).numpy()

    # 输出特征提取后的数据形状
    print(f"Feature extracted data shape: {features.shape}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = preprocess_data(features, y)

    # 输出划分后的数据形状
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 创建数据加载器
    train_loader = create_dataloader(X_train, y_train, batch_size=32)
    test_loader = create_dataloader(X_test, y_test, batch_size=32)

    # 调整 num_cut 以降低内存需求
    num_cut = [5] * min(X_train.shape[1], 3)  # 降低每个特征的切分点数

    # 打印调试信息
    print(f"num_leaf: {np.prod(np.array(num_cut) + 1)}")

    # 搭建模型
    dndt = DNDT(num_cut, num_class=2, temperature=0.1)
    writer = SummaryWriter(log_dir=os.path.join("logs", "DNDT"))

    # 训练模型
    start_time = time.time()
    dndt.fit(train_loader, writer, num_epochs=1000)
    print("--- %s seconds ---" % (time.time() - start_time))

    # 预测
    y_pred, y_true = dndt.predict(test_loader)
    print("Classification Report (DNDT):")
    print(classification_report(y_true, y_pred))

    # 保存分类报告到文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    with open(f"output/classification_report_dndt_{timestamp}.txt", "w") as f:
        f.write("Classification Report (DNDT):\n")
        f.write(classification_report(y_true, y_pred))

    # 关闭TensorBoard写入器
    writer.close()

    # 可视化训练损失
    os.system("tensorboard --logdir=logs")

    # 与传统决策树比较
    clf = tree.DecisionTreeClassifier()

    start_time = time.time()

    clf = clf.fit(X_train, y_train)

    y_pred_tree = clf.predict(X_test)

    print("--- %s seconds ---" % (time.time() - start_time))
    print("Classification Report (Decision Tree):")
    print(classification_report(y_test, y_pred_tree))

    # 保存分类报告到文件
    with open(f"output/classification_report_tree_{timestamp}.txt", "w") as f:
        f.write("Classification Report (Decision Tree):\n")
        f.write(classification_report(y_test, y_pred_tree))

    # 将分类报告写入TensorBoard
    writer = SummaryWriter(log_dir=os.path.join("logs", "Comparison"))
    writer.add_text(
        "Classification Report (DNDT)", classification_report(y_true, y_pred)
    )
    writer.add_text(
        "Classification Report (Decision Tree)",
        classification_report(y_test, y_pred_tree),
    )

    # 关闭TensorBoard写入器
    writer.close()

    # 画散点图比较预测值与实际值
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_tree, alpha=0.5, label="Decision Tree Predictions")
    plt.scatter(y_test, y_pred, alpha=0.5, label="DNDT Predictions")
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect Prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predictions vs Actual")
    plt.legend()
    plt.savefig(f"output/predictions_vs_actual_{timestamp}.png")
    plt.show()

    # 使用plot_tree进行决策树可视化
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        clf, filled=True, rounded=True, max_depth=3
    )  # 限制树的深度为3，使图更简洁
    plt.savefig(f"output/decision_tree_{timestamp}.png")
    plt.show()
