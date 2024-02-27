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

    for idx, mouse in enumerate(['A', 'B', 'C', 'D', 'E']):
        # Load model
        model = torch.load(f"saved_models/downsampled/with_behaviour/ridge_reg_{mouse}.pkl")        

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
        neurons = len(y_train[0])
        y_train = np.concatenate((y_train, behaviour_train), axis=1)
        y_train = np.array(y_train) / np.sqrt(len(y_train))
        X_train = np.array(X_train)
        print(f"X_train shape (images): {np.array(X_train).shape}")
        print(f"y_train shape (responses): {np.array(y_train).shape}")

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
        y_test = np.concatenate((y_test, np.array(behaviour_test)), axis=1)
        y_test = np.array(y_test) / np.sqrt(len(y_train))
        print(f"X_test shape (images): {np.array(X_test).shape}")
        print(f"y_test shape (responses): {np.array(y_test).shape}")

        
        # Get best performing voxel weight matrix (B) (m x pixels)
        y_pred = np.array(model.predict(X_test))
        explained_variance = explained_variance_paper(y_test, y_pred)
        # Get m best voxels & their weights
        m = 1000
        # Get indices of the top m voxels with the highest explained variances
        top_voxel_indices = np.argsort(explained_variance)[::-1][:m]
        # Get m voxel coefficients which include behavioural coefficients
        B_neurons = (model.coef_[top_voxel_indices, :])
        B_behaviors = model.coef_[neurons:, :]
        print(f"B_neurons shape: {B_neurons.shape}")
        print(f"B_behaviors shape: {B_behaviors.shape}")
        B = np.concatenate((B_neurons, B_behaviors), axis=0)
        print(f"B shape: {B.shape}")
        
        # Get y_test for the top m voxels (y) + behaviour + pupil (n_samples x m + 5)
        y_test_selected = np.array(y_test)[:, top_voxel_indices]
        y_test_selected_behaviours = np.concatenate((y_test_selected, behaviour_test), axis=1)
        print(f"y_test_selected_behaviours shape: {y_test_selected_behaviours.shape}")

        # Compute the covariance matrix of the image prior (R)
        R = np.cov(np.array(X_train), rowvar=False)
        print(f"R shape {R.shape}")

        # Compute the covariance matrix of residuals (Σ) on the training set (m x m)
        y_pred_train = np.array(model.predict(X_train))
        print(f"y_pred_train shape: {y_pred_train.shape}")
        del X_train
        y_train = np.concatenate((y_train[:, top_voxel_indices], y_train[:, neurons:]), axis=1)
        y_pred_train = np.concatenate((y_pred_train[:, top_voxel_indices], y_pred_train[:, neurons:]), axis=1)
        residuals_train = y_train - y_pred_train
        Sigma = np.diag(np.var(residuals_train, axis=0))
        print(f"Sigma shape: {Sigma.shape}")

        term1 = R - R @ B.T @ np.linalg.inv(Sigma + B @ R @ B.T) @ B @ R
        term2 = B.T @ np.linalg.inv(Sigma)

        del R, Sigma, B

        print(f"term1 shape: {term1.shape}")
        print(f"term2 shape: {term2.shape}")

        # Compute the decoding model
        X_test_pred = term1 @ term2 @ y_test_selected_behaviours.T
        del term1, term2, y_test_selected
        print(f"X_test_pred shape: {(X_test_pred.T).shape}")

        # Save np matrix as .npy file
        np.save(f"output/m_1000_with_behaviour/X_test_pred_{mouse}.npy", X_test_pred.T)
        del X_test_pred
        

if __name__ == "__main__":
    main()


