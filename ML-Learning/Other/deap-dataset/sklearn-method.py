# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

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
y_arousal = subject['labels'][:, 1]  # 觉醒度标签

# 对标签进行二分类（高效价/高觉醒度 vs 低效价/低觉醒度）
y_valence_binary = np.where(y_valence >= 5, 1, 0)
y_arousal_binary = np.where(y_arousal >= 5, 1, 0)

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_valence_binary, test_size=0.2, random_state=42)

# 逻辑回归
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f'Logistic Regression Accuracy: {accuracy_log_reg}')
print(classification_report(y_test, y_pred_log_reg))

# 支持向量机
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm}')
print(classification_report(y_test, y_pred_svm))

# 随机森林
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')
print(classification_report(y_test, y_pred_rf))

# 将相同的步骤应用于觉醒度标签
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_arousal_binary, test_size=0.2, random_state=42)

# 逻辑回归
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f'Logistic Regression Accuracy (Arousal): {accuracy_log_reg}')
print(classification_report(y_test, y_pred_log_reg))

# 支持向量机
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy (Arousal): {accuracy_svm}')
print(classification_report(y_test, y_pred_svm))

# 随机森林
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy (Arousal): {accuracy_rf}')
print(classification_report(y_test, y_pred_rf))
