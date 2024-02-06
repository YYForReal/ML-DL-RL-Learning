from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
"""
使用加利福尼亚州房价数据集来代替原本的Boston房价数据集.
load_boston函数已经从scikit-learn 1.2版本开始被移除了，原因是Boston房价数据集存在道德问题。该数据集的创建者在变量"B"的设计中假设种族自我隔离对房价有正面影响，这一假设及其研究目的受到了质疑。
"""
# 加载加利福尼亚州房价数据集
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树回归器实例
dtree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)

# 在训练集上训练决策树回归器
dtree_reg.fit(X_train, y_train)

# 对测试集进行预测
predictions = dtree_reg.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, predictions)
print(f'决策树回归器的均方误差: {mse}')


