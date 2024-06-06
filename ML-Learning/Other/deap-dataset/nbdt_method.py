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
from nbdt.model import SoftNBDT
from nbdt.loss import SoftTreeSupLoss
from nbdt.hierarchy import generate_hierarchy
from nbdt.models import wrn28_10_cifar10

# 定义加载数据的函数
def load_deap_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

# 加载数据
participant_id = 1
data_file = f'data_preprocessed_python/s{participant_id:02d}.dat'
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

# 加载预训练模型
model = wrn28_10_cifar10(pretrained=True)

# 使用我们自己的层级结构
hierarchy_path = 'config/hierarchy_deap.json'

# 定义SoftNBDT模型
nbdt_model = SoftNBDT(
    model=model,
    dataset='deap',  # 使用合适的数据集名称
    hierarchy=hierarchy_path
)

# 定义损失函数
criterion = nn.CrossEntropyLoss()
tree_sup_criterion = SoftTreeSupLoss(
    nbdt_model.graph,
    dataset='deap',
    hierarchy=nbdt_model.hierarchy,
    criterion=criterion
)

# 数据加载
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义优化器
optimizer = optim.Adam(nbdt_model.parameters(), lr=0.001)

# 训练模型
nbdt_model.train()
for epoch in range(20):  # 训练20个epoch
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = nbdt_model(inputs)
        loss = tree_sup_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i == 0:
            print(f'Epoch [{epoch+1}/20], Batch [{i+1}], Sample Outputs: {outputs[0]}, Label: {labels[0]}')
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/20], Loss: {avg_loss:.4f}')

# 评估模型
nbdt_model.eval()
correct = 0
total = 0
all_labels = []
all_preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = nbdt_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
print('Classification Report:')
print(classification_report(all_labels, all_preds))
