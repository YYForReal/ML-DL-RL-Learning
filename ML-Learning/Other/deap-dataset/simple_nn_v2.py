import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# 定义加载数据的函数
def load_deap_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

# 加载本地数据
participant_id = 1
data_file = f'data_preprocessed_python/s{participant_id:02d}.dat'
subject = load_deap_data(data_file)

# 提取数据和标签
X = subject['data']
y_valence = subject['labels'][:, 0]  # 效价标签

# 输出原始数据的维度
print(f'原始数据维度: {X.shape}')

# 对标签进行多分类（例如，分成3类：低、中、高）
# y_valence_multi = np.digitize(y_valence, bins=[3, 6])
y_valence_multi = np.where(y_valence >= 5, 1, 0)

# 将数据展平成2D数组
X_flat = X.reshape(X.shape[0], -1)

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# 使用PCA降维
pca = PCA(n_components=30)  # 降维到30维
X_pca = pca.fit_transform(X_scaled)

# 输出降维后的数据维度
print(f'降维后的数据维度: {X_pca.shape}')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_valence_multi, test_size=0.2, random_state=42)

# 输出数据集的基本信息
print(f'训练集样本数: {X_train.shape[0]}')
print(f'测试集样本数: {X_test.shape[0]}')
print(f'特征维度: {X_train.shape[1]}')

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# 定义模型参数
input_dim = X_train.shape[1]
hidden_dim1 = 512
hidden_dim2 = 256
output_dim = len(np.unique(y_valence_multi))

# 创建模型
model = SimpleNN(input_dim, hidden_dim1, hidden_dim2, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 数据加载
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练模型
model.train()
for epoch in range(25):  # 训练20个epoch
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i == 0:
            print(f'第 {epoch+1} 个epoch，第 {i+1} 个批次，样本输出: {outputs[0]}, 标签: {labels[0]}')
    avg_loss = total_loss / len(train_loader)
    print(f'第 {epoch+1} 个epoch，平均损失: {avg_loss:.4f}')

# 保存模型权重
torch.save(model.state_dict(), 'simple_nn_deap.pth')

# 评估模型
model.eval()
correct = 0
total = 0
all_labels = []
all_preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
print(f'准确率: {accuracy:.2f}%')
print('分类报告:')
print(classification_report(all_labels, all_preds))
