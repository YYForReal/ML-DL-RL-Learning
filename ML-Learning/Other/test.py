import numpy as np
import torch
import torch.nn.functional as F
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import time
import random
from sklearn.base import BaseEstimator
import warnings


# one_hot编码
def ont_hot_code(labels):
    labels = np.array(labels)
    ohe = OneHotEncoder()
    ohe.fit([[0], [1]])
    code_df = ohe.transform(labels.reshape(-1, 1)).toarray()

    cn_label = code_df[:, 0]  # narray
    mci_label = code_df[:, 1]

    return cn_label, mci_label


class NNDT_Classifier(BaseEstimator):
    def __init__(
        self,
        num_cut,
        num_class,
        epoch,
        temperature,
        num_leaf=None,
        cut_points_list=None,
        leaf_score=None,
        W=None,
        b=None,
    ):
        self.num_cut = num_cut
        self.num_leaf = np.prod(np.array(num_cut) + 1)
        self.num_class = num_class
        self.temperature = torch.tensor(temperature)
        self.cut_points_list = [torch.rand([i], requires_grad=True) for i in num_cut]
        self.leaf_score = torch.rand(
            [self.num_leaf, self.num_class], requires_grad=True
        )
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.cut_points_list + [self.leaf_score] + [self.temperature], lr=0.01
        )
        self.epoch = epoch
        self.W = W
        self.b = b

    # 计算克罗内克积
    def torch_kron_prod(self, a, b):
        res = torch.einsum("ij,ik->ijk", [a, b])
        res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
        return res

    # 软分箱算法
    def torch_bin(self, x, cut_points, temperature):
        D = cut_points.shape[0]
        self.W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
        cut_points, _ = torch.sort(cut_points)
        self.b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0), 0)
        h = torch.matmul(x, self.W) + self.b
        h = h / self.temperature
        res = F.softmax(h, dim=1)
        return res

    # 建树
    def nn_decision_tree(self, x):
        leaf = reduce(
            self.torch_kron_prod,
            map(
                lambda z: self.torch_bin(x[:, z[0] : z[0] + 1], z[1], self.temperature),
                enumerate(self.cut_points_list),
            ),
        )
        return torch.matmul(leaf, self.leaf_score)

    def fit(self, X_train, y_train):
        x = torch.from_numpy(X_train.astype(np.float32))
        y = torch.from_numpy(np.argmax(y_train, axis=1))
        for i in range(1000):
            self.optimizer.zero_grad()
            y_pred = self.nn_decision_tree(x)
            loss = self.loss_function(y_pred, y)
            loss.backward()
            self.optimizer.step()
            if i % 200 == 0:
                print("epoch %d loss= %f" % (i, loss.detach().numpy()))
        print(
            "error rate %.2f"
            % (
                1
                - np.mean(
                    np.argmax(y_pred.detach().numpy(), axis=1)
                    == np.argmax(y_train, axis=1)
                )
            )
        )
        return y_pred

    def predict(self, X_test):
        x = torch.from_numpy(X_test.astype(np.float32))
        y_pred = self.nn_decision_tree(x)
        return y_pred

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn(
                    "From version 0.24, get_params will raise an "
                    "AttributeError if a parameter cannot be "
                    "retrieved as an instance attribute. Previously "
                    "it would return None.",
                    FutureWarning,
                )
                value = None
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out


if __name__ == "__main__":
    # 数据准备
    np.random.seed(1943)
    torch.manual_seed(1943)

    cancers = datasets.load_breast_cancer()
    x = cancers["data"]
    _x = x[:, :8]
    y = cancers["target"]

    # 输出数据维度和示例值
    print("Original data shape:", x.shape)
    print("First 3 rows of original data:", x[:3])
    print("Selected features shape:", _x.shape)
    print("First 3 rows of selected features:", _x[:3])
    print("Target shape:", y.shape)
    print("First 10 target values:", y[:10])

    cn_hat, mci_hat = ont_hot_code(y)
    y = np.vstack((cn_hat, mci_hat))
    _y = y.T

    # 输出one-hot编码后的维度和示例值
    print("One-hot encoded target shape:", _y.shape)
    print("First 3 rows of one-hot encoded target:", _y[:3])

    feature_names = cancers["feature_names"]

    seed = random.seed(1990)

    X_train, X_test, y_train, y_test = train_test_split(
        _x, _y, train_size=0.70, random_state=seed
    )

    # 输出训练集和测试集维度
    print("Training data shape:", X_train.shape)
    print("Training target shape:", y_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Testing target shape:", y_test.shape)

    d = X_train.shape[1]
    print("Number of features:", d)

    num_cut = [1, 1, 1, 1, 1, 1, 1, 1]
    num_class = 2
    epoch = 1000
    temperature = 0.1

    start_time = time.time()

    # 搭建模型
    nndt = NNDT_Classifier(num_cut, num_class, epoch, temperature)
    nndt.fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start_time))

    # 预测
    y_pred = nndt.predict(X_test)
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
    print("====================================================================")
    print("NNDT： ", classification_report(y_test[:, 1], y_pred))

    # 与传统决策树比较
    from sklearn import tree

    clf = tree.DecisionTreeClassifier()

    start_time = time.time()

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("====================================================================")
    print("DT： ", classification_report(y_test, y_pred))

    # 画图
    plt.figure(figsize=(8, 8))
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=np.argmax(y_test, axis=1),
        marker="o",
        s=50,
        cmap="summer",
        edgecolors="black",
    )
    # plt.show()
