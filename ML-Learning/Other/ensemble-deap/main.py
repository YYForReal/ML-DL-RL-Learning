import wandb
from data_processing import get_data_and_labels, preprocess_data
from train import train_and_evaluate


def main():
    wandb.init(project="DEAP Emotion Classification with DNDT")

    label_idx = 0  # Adjust this index based on the desired label (valence=0, arousal=1, dominance=2, liking=3)
    X, y = get_data_and_labels(label_idx)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    num_epochs = 4000
    branch_num = 1
    learning_rate = 0.001
    train_and_evaluate(
        X_train, X_test, y_train, y_test, num_epochs, branch_num, learning_rate
    )


if __name__ == "__main__":
    main()
