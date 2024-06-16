import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)


# 定义加载DEAP数据集的函数
def load_deap_data(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")
    return data


# 提取数据和标签
def get_data_and_labels(participant_id, label_idx):
    data_dir = "data_preprocessed_python"
    if participant_id == 0:
        X_list = []
        y_list = []
        for i in range(1, 33):
            data_file = os.path.join(data_dir, f"s{i:02d}.dat")
            subject = load_deap_data(data_file)
            X_list.append(subject["data"])
            y_list.append(subject["labels"][:, label_idx])
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
    else:
        data_file = os.path.join(data_dir, f"s{participant_id:02d}.dat")
        subject = load_deap_data(data_file)
        X = subject["data"]
        y = subject["labels"][:, label_idx]

    y_binary = np.where(y >= 5, 1, 0)
    return X, y_binary


# 特征缩放并划分训练集和测试集
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
    X_scaled = X_scaled.reshape(X.shape[0], X.shape[1], -1)  # 保持原始的40通道形状
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 定义深度神经网络加持的决策树模型
class DNDT(nn.Module):
    def __init__(self, input_dim, num_cut_points, num_classes):
        super(DNDT, self).__init__()
        self.num_cut_points = num_cut_points
        # 定义切分点参数
        self.cut_points = nn.ParameterList(
            [nn.Parameter(torch.rand(num_cut_points)) for _ in range(input_dim)]
        )
        # 定义叶节点的分数参数
        self.leaf_scores = nn.Parameter(
            torch.rand((num_cut_points + 1) ** 2, num_classes)
        )

    def forward(self, x):
        N, D = x.shape
        bin_assignments = []
        for i in range(D):
            # 计算每个特征的二进制分配
            W = torch.linspace(
                1.0, self.num_cut_points + 1.0, self.num_cut_points + 1
            ).to(x.device)
            b = torch.cumsum(
                torch.cat([torch.zeros([1]).to(x.device), -self.cut_points[i]], 0), 0
            )
            h = torch.matmul(x[:, i : i + 1], W.view(1, -1)) + b
            bin_assignments.append(torch.softmax(h, dim=-1))
        leaf_assignments = reduce(
            lambda a, b: a.unsqueeze(-1) * b.unsqueeze(1), bin_assignments[:2]
        )
        leaf_assignments = leaf_assignments.view(N, -1)
        return torch.matmul(leaf_assignments, self.leaf_scores)


# 保存模型
def save_model(model, filename):
    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("output", filename))


# 训练模型并保存结果
def train_and_evaluate_model(
    X_train, X_test, y_train, y_test, label_name, num_epochs, device
):
    input_dim = X_train.shape[1]
    num_cut_points = 3  # 可调整
    num_classes = 2
    model = DNDT(input_dim, num_cut_points, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.long).to(device),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(y_test, dtype=torch.long).to(device),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 记录训练损失
    loss_history = []

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"
        ):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Loss: {epoch_loss:.4f}")

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"测试集准确率: {accuracy:.2f}%")
    print("分类报告:")
    report = classification_report(all_labels, all_predictions, output_dict=True)
    print(classification_report(all_labels, all_predictions))
    print("混淆矩阵:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    # 保存模型和结果
    save_model(model, f"DNDT_{label_name}_model.pth")
    with open(
        os.path.join("output", f"DNDT_{label_name}_classification_report.txt"), "w"
    ) as f:
        f.write(f"测试集准确率: {accuracy:.2f}%\n")
        f.write("分类报告:\n")
        f.write(classification_report(all_labels, all_predictions))
        f.write("混淆矩阵:\n")
        f.write(np.array2string(cm))

    # 绘制训练损失曲线
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{label_name.capitalize()} Training Loss")
    plt.legend()
    plt.savefig(os.path.join("output", f"DNDT_{label_name}_training_loss.png"))
    plt.show()

    # 可视化分类报告
    metrics = ["precision", "recall", "f1-score"]
    for metric in metrics:
        plt.figure()
        for i in range(num_classes):
            plt.bar(i, report[str(i)][metric], label=f"Class {i}")
        plt.xlabel("Classes")
        plt.ylabel(metric.capitalize())
        plt.title(f"{label_name.capitalize()} {metric.capitalize()}")
        plt.legend()
        plt.savefig(os.path.join("output", f"DNDT_{label_name}_{metric}.png"))
        plt.show()

    # 可视化混淆矩阵
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{label_name.capitalize()} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, range(len(cm)), rotation=45)
    plt.yticks(tick_marks, range(len(cm)))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join("output", f"DNDT_{label_name}_confusion_matrix.png"))
    plt.show()


# 选择要训练的标签
labels = ["valence", "arousal", "dominance", "liking"]
label_indices = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}

# 使用argparse定义可选参数
parser = argparse.ArgumentParser(description="Train and evaluate DNDT on DEAP dataset")
parser.add_argument(
    "--participant",
    type=int,
    default=0,
    help="Participant ID (default: 0 for all participants)",
)
parser.add_argument(
    "--num_epochs", type=int, default=10, help="Number of epochs (default: 10)"
)
parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
args = parser.parse_args()

participant_id = args.participant
num_epochs = args.num_epochs
device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

for label_name in labels:
    print(f"训练和评估 {label_name} 标签...")
    X, y = get_data_and_labels(participant_id, label_indices[label_name])
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    train_and_evaluate_model(
        X_train, X_test, y_train, y_test, label_name, num_epochs, device
    )

print("所有标签的模型训练和评估已完成。")
