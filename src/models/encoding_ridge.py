import numpy as np
from sklearn.linear_model import Ridge
import argparse
import torch
import wandb
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Multi-output regression (RidgeRegression supports multi-output regression)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

sys.path.append("C:/Users/sachi/SensoriumDecoding/src/utils")

from data import load_args, get_training_ds

# Change working directory to src
os.chdir("C:/Users/sachi/SensoriumDecoding")

class Args:
    def __init__(self):
        self.device = torch.device("cpu")
        self.output_dir = "runs/ridge_reg"
args = Args()
load_args(args)

def main():
    train_ds, val_ds, test_ds = get_training_ds(
        args,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        data_dir=args.dataset,
        device=args.device,
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

    # Training data
    for idx, mouse in enumerate(['A', 'B', 'C', 'D', 'E']):
        model = Ridge(alpha=1.0)
        # Train data
        X_train, y_train = [], []
        for i, batch in enumerate(train_ds[mouse]):
            X_train_batch = np.array([sample.flatten().numpy() for sample in batch['image']])
            y_train_batch = np.array([sample.flatten().numpy() for sample in batch['response']])
            X_train.extend(X_train_batch)
            y_train.extend(y_train_batch)

        print(f"X_train shape (images): {np.array(X_train).shape}")
        print(f"y_train shape (responses): {np.array(y_train).shape}")

        # Fit model
        model.fit(X_train, y_train)

        # Test data
        X_test, y_test = [], []
        for i, batch in enumerate(test_ds[mouse]):
            X_test_batch = np.array([sample.flatten().numpy() for sample in batch['image']])
            y_test_batch = np.array([sample.flatten().numpy() for sample in batch['response']])
            X_test.extend(X_test_batch)
            y_test.extend(y_test_batch)
        
        print(f"X_test shape (images): {np.array(X_test).shape}")
        print(f"y_test shape (responses): {np.array(y_test).shape}")
        
        y_test = np.array(y_test)
        y_pred = np.array(model.predict(X_test))

        correlations = [pearsonr(y_test[:, i], y_pred[:, i])[0] if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0 else 0.0 for i in range(y_test.shape[1])]
        rmses = [np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])) for i in range(y_test.shape[1])]
        explained_variances = [r2_score(y_test[:, i], y_pred[:, i]) if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0 else 0.0 for i in range(y_test.shape[1])]

        # Average correlations and RMSEs
        avg_correlation = np.mean(correlations)
        avg_rmse = np.mean(rmses)

        print(f"Average Column-wise Pearson Correlation Mouse {mouse}:", avg_correlation)
        print(f"Average Column-wise RMSE Mouse {mouse}:", avg_rmse)
        print(f"Average Column-wise Explained Variance Mouse {mouse}:", np.mean(explained_variances))

        # Plot correlations histogram for each mouse
        row = idx // 3
        col = idx % 3
        axes[row, col].hist(correlations, bins=200, range=(-1, 1), color='blue', alpha=0.7)
        axes[row, col].set_title(f'Mouse {mouse}')
        axes[row, col].set_xlabel('Correlation')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True)
        
        # Plot explained variances histogram for each mouse
        axes2[row, col].hist(explained_variances, bins=200, range=(-1, 1), color='blue', alpha=0.7)
        axes2[row, col].set_title(f'Mouse {mouse}')
        axes2[row, col].set_xlabel('Explained Variance (R^2)')
        axes2[row, col].set_ylabel('Frequency')
        axes2[row, col].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()