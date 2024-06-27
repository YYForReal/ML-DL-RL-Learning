import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats
import wandb
from model import DNDT
import csv


def create_dataloader(X, y, batch_size=32):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 定义统计特征名称
STAT_FEATURES = [
    "mean",
    "std_dev",
    "skewness",
    "kurtosis",
    "max_val",
    "min_val",
    "range_val",
]


def train_and_evaluate(
    X_train, X_test, y_train, y_test, num_epochs=2000, branch_num=1, lr=0.001
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    feature_num = X_train.shape[1]
    wandb.config.update(
        {
            "num_epochs": num_epochs,
            "feature_num": feature_num,
            "branch_num": branch_num + 1,
        }
    )

    # 初始化CSV文件
    with open("tree_accuracy.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Tree Name", "Train Accuracy", "Test Accuracy"])

    if feature_num > 12:
        num_trees = 10
        features_per_tree = 10
        wandb.config.update(
            {
                "num_trees": num_trees,
                "features_per_tree": features_per_tree,
            }
        )
        trees = []

        for i in range(num_trees):
            selected_features = np.random.choice(
                feature_num, features_per_tree, replace=False
            )
            tree = DNDT(
                [branch_num] * features_per_tree,
                num_class=2,
                temperature=0.1,
                device=device,
                name=f"Tree_{i}",  # 为每棵树添加名称
                lr=lr,
            ).to(device)
            trees.append((tree, selected_features))

            # 映射特征名
            selected_feature_names = [
                f"channel_{(f // len(STAT_FEATURES)) + 1}_{STAT_FEATURES[f % len(STAT_FEATURES)]}"
                for f in selected_features
            ]

            # 打印并保存选定的特征
            print(f"Tree {i}: Selected Features: {selected_features}")
            with open("selected_features.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([i, selected_features, selected_feature_names])

        train_loader = create_dataloader(X_train, y_train, batch_size=32)
        test_loader = create_dataloader(X_test, y_test, batch_size=32)

        all_train_losses = []
        all_train_accuracies = []

        for tree, selected_features in trees:
            # 确保选择的是随机特征
            X_train_selected = X_train[:, selected_features]
            X_test_selected = X_test[:, selected_features]
            train_loader_selected = create_dataloader(
                X_train_selected, y_train, batch_size=32
            )
            test_loader_selected = create_dataloader(
                X_test_selected, y_test, batch_size=32
            )

            losses, accuracies = tree.fit(train_loader_selected, num_epochs)
            all_train_losses.append(losses)
            all_train_accuracies.append(accuracies)

            # 在 wandb 上记录训练和准确率
            for epoch, (loss, acc) in enumerate(zip(losses, accuracies)):
                wandb.log(
                    {
                        f"{tree.name}_loss": loss,
                        f"{tree.name}_accuracy": acc,
                        "epoch": epoch,
                    }
                )

            # 单独测试每棵树并输出结果到 CSV 文件
            train_preds, train_labels = tree.predict(train_loader_selected)
            test_preds, test_labels = tree.predict(test_loader_selected)

            train_accuracy = accuracy_score(train_labels, train_preds)
            test_accuracy = accuracy_score(test_labels, test_preds)

            print(f"{tree.name} - Training Accuracy: {train_accuracy:.4f}")
            print(f"{tree.name} - Testing Accuracy: {test_accuracy:.4f}")
            print(classification_report(test_labels, test_preds))

            with open("tree_accuracy.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([tree.name, train_accuracy, test_accuracy])

        def ensemble_predict(dataloader, trees):
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for x_batch, y_batch in dataloader:
                    batch_preds = []
                    for tree, selected_features in trees:
                        x_batch_selected = x_batch[:, selected_features].to(device)
                        y_pred = tree.nn_decision_tree(x_batch_selected)
                        batch_preds.append(
                            torch.argmax(y_pred, axis=1).cpu().detach().numpy()
                        )
                    ensemble_preds = stats.mode(batch_preds, axis=0)[0].flatten()
                    all_preds.append(ensemble_preds)
                    all_labels.append(y_batch.cpu().numpy())

                    # 打印每个批次的预测结果
                    print(f"Batch Predictions: {batch_preds}")
                    print(f"Ensemble Predictions: {ensemble_preds}")

            return np.concatenate(all_preds), np.concatenate(all_labels)

        train_preds, train_labels = ensemble_predict(train_loader, trees)
        test_preds, test_labels = ensemble_predict(test_loader, trees)

        train_accuracy = accuracy_score(train_labels, train_preds)
        test_accuracy = accuracy_score(test_labels, test_preds)

        print(f"Ensemble Training Accuracy: {train_accuracy:.4f}")
        print(f"Ensemble Testing Accuracy: {test_accuracy:.4f}")
        print(classification_report(test_labels, test_preds))

        # 保存集成训练和测试的准确率到CSV文件
        with open("tree_accuracy.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ensemble", train_accuracy, test_accuracy])

    else:
        tree = DNDT(
            [branch_num] * feature_num,
            num_class=2,
            temperature=0.1,
            device=device,
            lr=lr,
        ).to(device)
        train_loader = create_dataloader(X_train, y_train, batch_size=32)
        test_loader = create_dataloader(X_test, y_test, batch_size=32)

        losses, accuracies = tree.fit(train_loader, num_epochs)

        train_preds, train_labels = tree.predict(train_loader)
        test_preds, test_labels = tree.predict(test_loader)

        train_accuracy = accuracy_score(train_labels, train_preds)
        test_accuracy = accuracy_score(test_labels, test_preds)

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        print(classification_report(test_labels, test_preds))

        # 保存单棵树训练和测试的准确率到CSV文件
        with open("tree_accuracy.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Train Accuracy", "Test Accuracy"])
            writer.writerow(["single_tree", train_accuracy, test_accuracy])
