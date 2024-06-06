在你的训练过程中，`loss`（损失）是通过以下方式计算的：

1. **损失函数定义**：
   在代码中，损失函数被定义为交叉熵损失（CrossEntropyLoss），这是一种常用的分类任务损失函数。

   ```python
   criterion = nn.CrossEntropyLoss()
   ```

2. **前向传播计算输出**：
   对于每个批次的输入数据，模型会进行前向传播计算出预测输出（logits）。

   ```python
   outputs = model(inputs)
   ```

3. **计算损失**：
   利用模型的预测输出和真实标签，计算出当前批次的损失值。`criterion(outputs, labels)` 会计算交叉熵损失。

   ```python
   loss = criterion(outputs, labels)
   ```

4. **反向传播和优化**：
   计算损失之后，进行反向传播来计算梯度，并通过优化器更新模型的权重。

   ```python
   loss.backward()
   optimizer.step()
   ```

### 交叉熵损失函数的详细解释

交叉熵损失函数（CrossEntropyLoss）是深度学习中常用的损失函数，特别适用于分类问题。它的计算公式如下：

\[ \text{Loss} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i) \]

其中：
- \( y_i \) 是真实标签的one-hot编码（即目标类为1，其余类为0）。
- \( \hat{y}_i \) 是模型预测的概率分布（通常通过softmax函数计算得到）。

对于每个样本，交叉熵损失会衡量模型预测的概率分布与真实分布之间的差异。当模型的预测概率与真实标签越接近时，损失值越小；反之，损失值越大。

在代码中，每个epoch和batch中都会计算该损失，并逐渐通过反向传播和优化过程减少该损失，从而提高模型的准确性。

### 具体例子

example1 

