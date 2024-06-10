## WESAD 数据集分析与模型训练
### 加载数据
# 我们首先加载WESAD数据集中的特定受试者的数据。每个受试者的数据存储在一个.pkl文件中，包含了信号数据、标签和受试者信息。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import pickle
import torch.nn.functional as F
from tqdm import tqdm

# 固定随机种子以便复现
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 加载数据函数
# 该函数从指定路径加载特定受试者的数据文件，并返回解压后的数据字典。
def load_wesad_data(participant_id, data_path='WESAD'):
    file_path = os.path.join(data_path, f'S{participant_id}', f'S{participant_id}.pkl')
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

# 数据预处理函数
# 该函数对ECG、EDA、Temp和ACC数据进行标准化处理，并将不同类型的数据组合在一起，同时重新映射标签以确保标签值从0到6连续。
def preprocess_data(data):
    ecg_data = data['signal']['chest']['ECG']
    eda_data = data['signal']['chest']['EDA'].reshape(-1, 1)
    temp_data = data['signal']['chest']['Temp'].reshape(-1, 1)
    acc_data = data['signal']['chest']['ACC']
    
    # 保证所有数据的长度一致
    min_length = min(len(ecg_data), len(eda_data), len(temp_data), len(acc_data))
    ecg_data = ecg_data[:min_length]
    eda_data = eda_data[:min_length]
    temp_data = temp_data[:min_length]
    acc_data = acc_data[:min_length]

    # 将所有数据合并为一个数组
    combined_data = np.hstack((ecg_data, eda_data, temp_data, acc_data))
    
    # 数据标准化
    scaler = StandardScaler()
    combined_data = scaler.fit_transform(combined_data)
    
    labels = data['label'][:min_length]
    
    # 重新映射标签，使得标签从0到6连续
    label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6}
    mapped_labels = np.vectorize(label_mapping.get)(labels)
    
    print(f"标签的唯一值: {np.unique(mapped_labels)}")
    for i, label_name in enumerate(["ECG", "EDA", "Temp", "ACC_X", "ACC_Y", "ACC_Z"]):
        print(f"维度 {i+1} ({label_name}) 的标签值: {combined_data[:, i]}")
    
    return combined_data, mapped_labels

# 加载特定受试者的数据
participant_id = 2
data = load_wesad_data(participant_id)
preprocessed_data, labels = preprocess_data(data)
print(preprocessed_data.shape)
print(labels.shape)

# 将数据转换为PyTorch张量
# 使用torch.tensor将numpy数组转换为PyTorch张量
X_tensor = torch.tensor(preprocessed_data, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)

# 划分训练集和测试集
# 使用torch.utils.data.random_split将数据集随机划分为训练集和测试集
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
X_train, X_test = torch.utils.data.random_split(TensorDataset(X_tensor, y_tensor), [train_size, test_size])

# 创建数据加载器
# 使用DataLoader创建训练和测试数据的加载器
train_loader = DataLoader(X_train, batch_size=32, shuffle=True)
test_loader = DataLoader(X_test, batch_size=32, shuffle=False)

# 定义CNN模型
# 该模型包含一个一维卷积层、一个池化层和两个全连接层。
class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (input_size // 2), 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 64 * (x.shape[2]))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
# 创建CNN模型实例，并定义损失函数和优化器
input_size = preprocessed_data.shape[1]
num_classes = len(np.unique(labels))
model = SimpleCNN(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
# 训练模型时输出每个epoch的损失值
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.unsqueeze(1)  # 添加通道维度
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Loss: {epoch_loss:.4f}')

# 测试模型
# 测试模型并输出准确率和分类报告
model.eval()
correct = 0
total = 0
all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.unsqueeze(1)  # 添加通道维度
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

accuracy = 100 * correct / total
print(f'测试集准确率: {accuracy:.2f}%')
print('分类报告:')
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(all_labels, all_predictions))
print('混淆矩阵:')
print(confusion_matrix(all_labels, all_predictions))
