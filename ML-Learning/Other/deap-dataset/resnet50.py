import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report


# 定义加载数据的函数
def load_deap_data(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")
    return data


# 加载本地数据
participant_id = 1
data_file = f"data/data_preprocessed_python/s{participant_id:02d}.dat"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"数据文件未找到: {data_file}")
subject = load_deap_data(data_file)

# 提取数据和标签
X = subject["data"]
y_valence = subject["labels"][:, 0]  # 使用效价标签

# 输出原始数据的维度
print(f"原始数据维度: {X.shape}")

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 参数设定
USING_PCA = False

# 使用PCA降维
if USING_PCA:
    n_samples, n_channels, n_times = X.shape
    n_components = min(n_samples, n_channels * n_times)  # 确保降维组件数在允许范围内
    pca = PCA(n_components=n_components)
    X_processed = pca.fit_transform(X_scaled.reshape(X.shape[0], -1))
    print(f"降维后的数据维度: {X_processed.shape}")
    X_processed = X_processed.reshape(X_processed.shape[0], 1, n_channels, n_times)
else:
    X_processed = X_scaled.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    print(f"原始数据维度: {X_processed.shape}")

# 对标签进行多分类（例如，分成3类：低、中、高）
y_valence_multi = np.digitize(y_valence, bins=[3, 6])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_valence_multi, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 构建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载预训练的ResNet50模型
resnet50 = models.resnet50(pretrained=True)

# 修改ResNet50的输入层和输出层
resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 3)  # 假设有3个类别

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    resnet50.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 评估模型
resnet50.eval()
correct = 0
total = 0
all_labels = []
all_preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = resnet50(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
print(f"准确率: {accuracy:.2f}%")
print("分类报告:")
print(classification_report(all_labels, all_preds))
