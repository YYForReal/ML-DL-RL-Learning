from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 加载鸢尾花数据集
data = load_iris()
features = data.data
labels = data.target
# print(data.DESCR)
print(data.target_names)
print(data.feature_names)
print(features)
print(labels)
# 将数据集拆分为训练集和测试集
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

# 创建决策树分类器对象
dt_classifier = DecisionTreeClassifier()

# 在训练集上训练决策树分类器
dt_classifier.fit(train_features, train_labels)

# 查看特征重要性
feature_importance = dt_classifier.feature_importances_
print("特征重要性:", feature_importance)
