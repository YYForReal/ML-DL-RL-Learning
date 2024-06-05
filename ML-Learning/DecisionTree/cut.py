from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 加载加利福尼亚州房价数据集
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树回归器实例，应用预剪枝策略
# 限制树的最大深度为3
# 设置节点在分裂前必须拥有的最小样本数为10
# 设置分裂节点时所需的最小增益为0.01
dtree_reg = DecisionTreeRegressor(max_depth=3, min_samples_split=10, min_impurity_decrease=0.01, random_state=42)

# 在训练集上训练决策树回归器
dtree_reg.fit(X_train, y_train)

# 对测试集进行预测
predictions = dtree_reg.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, predictions)
print(f'决策树回归器的均方误差: {mse}')

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(dtree_reg, feature_names=housing.feature_names, filled=True, rounded=True, max_depth=3)
plt.show()
