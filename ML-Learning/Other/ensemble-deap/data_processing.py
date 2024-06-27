import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


def extract_features(X):
    features = []
    # Original data shape: (1280, 40, 8064)
    print(f"Extracting features from {X.shape[0]} samples...")
    # Extracting features from 1280 samples...

    for sample in X:
        sample_features = []
        for channel in sample:
            mean = np.mean(channel)
            std_dev = np.std(channel)
            skewness = stats.skew(channel)
            kurtosis = stats.kurtosis(channel)
            max_val = np.max(channel)
            min_val = np.min(channel)
            range_val = max_val - min_val
            sample_features.extend(
                [mean, std_dev, skewness, kurtosis, max_val, min_val, range_val]
            )
        features.append(sample_features)
    # Extracted features shape: (1280, 224)
    return np.array(features)


def load_deap_data(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")
    return data


def get_data_and_labels(label_idx, data_dir="data_preprocessed_python"):
    X_list = []
    y_list = []
    for i in range(1, 33):  # 32 participants
        data_file = os.path.join(data_dir, f"s{i:02d}.dat")
        subject = load_deap_data(data_file)
        X_list.append(subject["data"])
        y_list.append(subject["labels"][:, label_idx])
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    y_binary = np.where(y >= 5, 1, 0)
    return X, y_binary


def preprocess_data(X, y):
    print(f"Original data shape: {X.shape}")  # Original data shape: (1280, 40, 8064)
    X = X[:, :32, :]  # Use the first 32 channels
    X_features = extract_features(X)
    print(f"Extracted features shape: {X_features.shape}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    print(f"Scaled data shape: {X_scaled.shape}")
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
