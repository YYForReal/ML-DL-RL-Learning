from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 假设你有一个包含特征和目标值的数据集
# 特征通常是一个二维数组，目标值是一个一维数组
features = [[2, 4], [4, 6], [3, 7], [6, 2], [7, 4], [5, 8]]
targets = [5, 8, 9, 3, 2, 6]

# 将数据集拆分为训练集和测试集
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)

# 创建KNN回归器对象，设定K值为3，并指定距离度量为欧几里德距离
knn = KNeighborsRegressor(n_neighbors=3, metric='manhattan')

# 在训练集上训练KNN回归器
knn.fit(train_features, train_targets)

# 使用训练好的回归器进行预测
predictions = knn.predict(test_features)
print("predictions:", predictions)
print("test_targets:", test_targets)
# 计算预测结果的均方误差（Mean Squared Error，MSE）
mse = mean_squared_error(test_targets, predictions)
print("均方误差:", mse)
