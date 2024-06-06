# 导入必要的库
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 没有改进，没有降维

# 定义加载数据的函数
def load_deap_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

# 加载本地数据
participant_id = 1
data_file = f'data_preprocessed_python/s{participant_id:02d}.dat'  # 修改为数据的实际路径
subject = load_deap_data(data_file)

# 提取数据和标签
X = subject['data']
y_valence = subject['labels'][:, 0]  # 效价标签

# 对标签进行多分类（例如，分成3类：低、中、高）
y_valence_multi = np.digitize(y_valence, bins=[3, 6])

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_valence_multi, test_size=0.2, random_state=42)

# 输出数据集的基本信息
print(f'Training set size: {X_train.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples')
print(f'Feature size: {X_train.shape[1]} features')

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 定义模型参数
input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = len(np.unique(y_valence_multi))

# 创建模型
model = SimpleNN(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据加载
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练模型
model.train()
for epoch in range(20):  # 训练20个epoch
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i == 0:
            print(f'Epoch [{epoch+1}/20], Batch [{i+1}], Sample Outputs: {outputs[0]}, Label: {labels[0]}')
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/20], Loss: {avg_loss:.4f}')

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
print(f'Accuracy: {accuracy:.2f}%')
print('Classification Report:')
print(classification_report(all_labels, all_preds))
