from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# 加载iris数据集
iris = load_iris()
X = iris.data
y = iris.target

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("测试集结果y_test",y_test)



# 创建决策树分类器实例
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)

# 在训练集上训练决策树分类器
dtree.fit(X_train, y_train)

# 对测试集进行预测
predictions = dtree.predict(X_test)
print("预测结果predictions",predictions)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print(f'决策树分类器的准确率: {accuracy}')
