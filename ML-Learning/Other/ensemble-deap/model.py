import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from sklearn.metrics import accuracy_score, f1_score  # 导入所需的函数
import wandb


class DNDT(nn.Module):
    def __init__(self, num_cut, num_class, temperature, device, name="", lr=0.001):
        super(DNDT, self).__init__()
        self.num_cut = num_cut
        self.num_leaf = np.prod(np.array(num_cut) + 1)
        self.num_class = num_class
        self.temperature = torch.tensor(temperature, device=device)
        self.name = name
        self.cut_points_list = [
            torch.rand([i], requires_grad=True, device=device) for i in num_cut
        ]
        self.leaf_score = torch.rand(
            [self.num_leaf, self.num_class], requires_grad=True, device=device
        )
        self.optimizer = torch.optim.Adam(
            self.cut_points_list + [self.leaf_score] + [self.temperature], lr=lr
        )

    def torch_kron_prod(self, a, b):
        res = torch.einsum("ij,ik->ijk", [a, b])
        res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
        return res

    def torch_bin(self, x, cut_points, temperature):
        D = cut_points.shape[0]
        W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
        W = W.to(x.device)  # Ensure W is on the same device as x
        cut_points, _ = torch.sort(cut_points)
        b = torch.cumsum(
            torch.cat([torch.zeros([1], device=x.device), -cut_points], 0), 0
        )
        h = torch.matmul(x, W) + b
        h = h / temperature
        res = F.softmax(h, dim=1)
        return res

    def nn_decision_tree(self, x):
        leaf = reduce(
            self.torch_kron_prod,
            map(
                lambda z: self.torch_bin(x[:, z[0] : z[0] + 1], z[1], self.temperature),
                enumerate(self.cut_points_list),
            ),
        )
        return torch.matmul(leaf, self.leaf_score)

    def forward(self, x):
        return self.nn_decision_tree(x)

    def fit(self, dataloader, num_epochs, start_index=0):
        epoch_losses = []
        epoch_accuracies = []
        for epoch in range(num_epochs):
            all_labels = []
            all_preds = []
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(x_batch.device), y_batch.to(
                    y_batch.device
                )
                self.optimizer.zero_grad()
                y_pred = self.nn_decision_tree(x_batch)
                loss = F.cross_entropy(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                all_labels.extend(y_batch.cpu().numpy())
                all_preds.extend(torch.argmax(y_pred, axis=1).cpu().detach().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            wandb.log(
                {
                    "Loss": loss.item(),
                    "Accuracy": accuracy,
                    "F1": f1,
                    "Epoch": epoch + start_index,
                }
            )
            if epoch % (num_epochs // 10) == 0:
                print(
                    f"Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}, F1: {f1}"
                )
        print("Training complete.")
        return epoch_losses, epoch_accuracies

    def predict(self, dataloader):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(x_batch.device), y_batch.to(
                    y_batch.device
                )
                y_pred = self.nn_decision_tree(x_batch)
                all_preds.append(torch.argmax(y_pred, axis=1).cpu().detach().numpy())
                all_labels.append(y_batch.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_labels)
