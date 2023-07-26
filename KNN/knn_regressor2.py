from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(train_features, train_targets, test_features, test_predictions):
    # 将特征数据转换为NumPy数组
    train_features = np.array(train_features)
    test_features = np.array(test_features)

    # 绘制训练集数据点（蓝色）
    plt.scatter(train_features[:, 0], train_features[:, 1], c='blue', label='Train')

    # 绘制预测结果数据点（红色）
    plt.scatter(test_features[:, 0], test_features[:, 1], c='red', label='Predictions')

    # 设置图例
    plt.legend(loc='upper right')

    # 设置坐标轴标签
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # 显示图形
    plt.show()


# 假设你有一个包含特征和目标值的数据集
# 特征通常是一个二维数组，目标值是一个一维数组
features = [[2, 4], [4, 6], [3, 7], [6, 2], [7, 4], [5, 8]]
targets = [5, 8, 9, 3, 2, 6]

# 将数据集拆分为训练集和测试集
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)

# 创建KNN回归器对象，设定K值为3，并指定距离度量为欧几里德距离
knn = KNeighborsRegressor(n_neighbors=3, metric='euclidean')

# 在训练集上训练KNN回归器
knn.fit(train_features, train_targets)

# 使用训练好的回归器进行预测
predictions = knn.predict(test_features)
print("要预测的特征点:", test_features)
print("预测的结果:", predictions)
print("实际的结果:", test_targets)
# 计算预测结果的均方误差（Mean Squared Error，MSE）
mse = mean_squared_error(test_targets, predictions)
print("均方误差:", mse)

plot_predictions(train_features, train_targets, test_features, predictions)
