from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假设你有一个包含特征和标签的数据集
# 特征通常是一个二维数组，标签是一个一维数组
features = [[2, 4], [4, 6], [3, 7], [6, 2], [7, 4], [5, 8]]
labels = ['A', 'B', 'A', 'B', 'B', 'A']

# 将数据集拆分为训练集和测试集
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

# 创建KNN分类器对象，设定K值为3
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# 在训练集上训练KNN分类器
knn.fit(train_features, train_labels)

# 使用训练好的分类器进行预测
predictions = knn.predict(test_features)

# 计算预测的准确率
accuracy = accuracy_score(test_labels, predictions)

print(f"测试标签: {test_labels}")
print(f"预测结果: {predictions}")
print(f"准确率: {accuracy}")
