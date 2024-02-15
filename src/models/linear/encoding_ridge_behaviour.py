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
from sklearn.linear_model import RidgeCV, Ridge
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

def explained_variance_paper(y, y_hat):
    # Calculate the variance of the actual response for each voxel
    var_y = np.var(y, axis=0)
    
    # Calculate the variance of the residuals for each voxel
    var_residuals = np.var(y - y_hat, axis=0)
    
    # Calculate the explained variance for each voxel
    explained_variances = (var_y - var_residuals) / var_y
    
    return explained_variances

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
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 10))

    # Training data
    for idx, mouse in enumerate(['A', 'B', 'C', 'D', 'E']):
        model = RidgeCV(alphas=[40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000], scoring='r2', alpha_per_target=True)

        # Train data
        X_train, y_train, behaviour_train = [], [], []
        for i, batch in enumerate(train_ds[mouse]):
            X_train_batch = np.array([sample.flatten().numpy() for sample in batch['image']])
            y_train_batch = np.array([sample.flatten().numpy() for sample in batch['response']])
            behaviour_batch = np.array([sample.flatten().numpy() for sample in batch['behavior']])
            pupil_batch = np.array([sample.flatten().numpy() for sample in batch['pupil_center']])
            X_train.extend(X_train_batch)
            y_train.extend(y_train_batch)
            behaviour_train.extend(np.concatenate((behaviour_batch, pupil_batch), axis=1))
        
        # Divide my sqrt(N) like in paper
        y_train = np.array(y_train) / np.sqrt(len(y_train))
        y_train = np.concatenate((y_train, np.array(behaviour_train)), axis=1)
        print(f"X_train shape (images): {np.array(X_train).shape}")
        print(f"y_train shape (responses): {np.array(y_train).shape}")

        # Fit model
        model.fit(X_train, y_train)
        print(f"Min alpha: {min(model.alpha_)}, Max alpha: {max(model.alpha_)}")

        # Test data
        X_test, y_test, behaviour_test = [], [], []
        for i, batch in enumerate(test_ds[mouse]):
            X_test_batch = np.array([sample.flatten().numpy() for sample in batch['image']])
            y_test_batch = np.array([sample.flatten().numpy() for sample in batch['response']])
            behaviour_batch = np.array([sample.flatten().numpy() for sample in batch['behavior']])
            pupil_batch = np.array([sample.flatten().numpy() for sample in batch['pupil_center']])
            X_test.extend(X_test_batch)
            y_test.extend(y_test_batch)
            behaviour_test.extend(np.concatenate((behaviour_batch, pupil_batch), axis=1))
        
        # Divide my sqrt(N) like in paper
        y_test = np.array(y_test) / np.sqrt(len(y_train))
        y_test = np.concatenate((y_test, np.array(behaviour_test)), axis=1)
        print(f"X_test shape (images): {np.array(X_test).shape}")
        print(f"y_test shape (responses): {np.array(y_test).shape}")
        
        y_pred = np.array(model.predict(X_test))

        correlations = [pearsonr(y_test[:, i], y_pred[:, i])[0] if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0 else 0.0 for i in range(y_test.shape[1])]
        rmses = [np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])) for i in range(y_test.shape[1])]
        r2s = [r2_score(y_test[:, i], y_pred[:, i]) if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0 else 0.0 for i in range(y_test.shape[1])]
        explained_variance = explained_variance_paper(y_test, y_pred)

        # Average correlations and RMSEs
        print(f"Average Column-wise Pearson Correlation Mouse {mouse}:", np.mean(correlations))
        print(f"Average Column-wise RMSE Mouse {mouse}:", np.mean(rmses))
        print(f"Average Column-wise R2 Mouse {mouse}:", np.mean(r2s))
        print(f"Average Column-wise Explained Variance Mouse {mouse}:", np.mean(explained_variance))

        # Save model (sklearn)
        model_path = os.path.join("saved_models/downsampled/with_behaviour", f"ridge_reg_{mouse}.pkl")
        torch.save(model, model_path)

        # Plot correlations histogram for each mouse
        row = idx // 3
        col = idx % 3
        axes[row, col].hist(correlations, bins=200, range=(-1, 1), color='blue', alpha=0.7)
        axes[row, col].set_title(f'Mouse {mouse}')
        axes[row, col].set_xlabel('Correlation')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True)
        
        # Plot explained variances histogram for each mouse
        axes2[row, col].hist(r2s, bins=200, range=(-1, 1), color='blue', alpha=0.7)
        axes2[row, col].set_title(f'Mouse {mouse}')
        axes2[row, col].set_xlabel('R2 Scores')
        axes2[row, col].set_ylabel('Frequency')
        axes2[row, col].grid(True)

        axes3[row, col].hist(explained_variance, bins=200, range=(-1, 1), color='blue', alpha=0.7)
        axes3[row, col].set_title(f'Mouse {mouse}')
        axes3[row, col].set_xlabel('Explained Variance')
        axes3[row, col].set_ylabel('Frequency')
        axes3[row, col].grid(True)

        # Sort explained variances
        sorted_explained_variance = sorted(explained_variance, reverse=True)
        # Plot with log scale on x-axis
        axes4[row, col].plot(sorted_explained_variance, color='b')
        axes4[row, col].set_title(f'Mouse {mouse}')
        axes4[row, col].set_xlabel('Neuron')
        axes4[row, col].set_ylabel('Explained Variance')
        axes4[row, col].grid(True)
        axes4[row, col].set_xscale('log')  # Set x-axis to log scale
        axes4[row, col].set_xlim(1, len(sorted_explained_variance)+100)


    plt.tight_layout()
    fig.savefig('figures/linear/with_behaviour/correlations.png')
    fig2.savefig('figures/linear/with_behaviour/r2s.png')
    fig3.savefig('figures/linear/with_behaviour/explained_variance.png')
    fig4.savefig('figures/linear/with_behaviour/explained_variance_sorted.png')
    plt.show()


if __name__ == "__main__":
    main()