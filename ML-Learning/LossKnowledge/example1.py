import torch.nn.functional as F
import torch

# 假设有一个样本，其真实标签为类 `2`（即one-hot编码为 [0, 0, 1]），模型的预测输出 logits 为 [-4.1116, 5.8398, 2.5449]。我们首先将 logits 通过 softmax 函数转化为概率分布，然后计算交叉熵损失。

# 示例数据
logits = torch.tensor([-4.1116, 5.8398, 2.5449])

# 计算 softmax 概率分布
probabilities = F.softmax(logits, dim=0)
print(f"概率分布: {probabilities}")

# 计算交叉熵损失
labels_predict1 = torch.tensor([0])
labels_predict2 = torch.tensor([1])
labels_predict3 = torch.tensor([2])

# 计算交叉熵损失1
loss = F.cross_entropy(logits.unsqueeze(0), labels_predict1)
print(f"交叉熵损失1: {loss.item()}")

# 计算交叉熵损失2
loss = F.cross_entropy(logits.unsqueeze(0), labels_predict2)
print(f"交叉熵损失2: {loss.item()}")

# 计算交叉熵损失3
loss = F.cross_entropy(logits.unsqueeze(0), labels_predict3)
print(f"交叉熵损失3: {loss.item()}")
