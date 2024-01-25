import numpy as np
from sklearn.linear_model import Ridge
import argparse
import torch
import wandb
import os
import sys
sys.path.append("C:/Users/sachi/SensoriumDecoding/src/utils")

# Local imports
from metrics import pearson_correlation, rmse
from data import load_args, get_training_ds


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

    wandb.init(project='my-awesome-project', config={"alpha": 0.1})
    alpha = wandb.config.alpha
    ridge_model = Ridge(alpha)

    # Training data
    for mouse in ['S1']:#, 'A', 'B', 'C', 'D', 'E']:
        X_train, y_train = [], []
        for i, batch in enumerate(train_ds[mouse]):
            X_train_batch = np.array([sample.flatten().numpy() for sample in batch['response']])
            y_train_batch = np.array([image.flatten().numpy() for image in batch['image']])
            X_train.extend(X_train_batch)
            y_train.extend(y_train_batch)

        ridge_model.fit(X_train, y_train)

        # Test data
        X_test, y_test = [], []
        for i, batch in enumerate(test_ds[mouse]):
            X_test_batch = np.array([sample.flatten().numpy() for sample in batch['response']])
            y_test_batch = np.array([image.flatten().numpy() for image in batch['image']])
            X_test.extend(X_test_batch)
            y_test.extend(y_test_batch)
    
        y_pred = ridge_model.predict(X_test)

        # Metrics
        pearson = pearson_correlation(np.array(y_test), np.array(y_pred))
        rmse_score = rmse(np.array(y_test), np.array(y_pred))
    
        wandb.log({"Pearson Correlation": pearson, "RMSE": rmse_score})
        model_save_path = os.path.join(wandb.run.dir, f"{mouse}_ridge_model.pth")
        torch.save(ridge_model, model_save_path)
        wandb.save(model_save_path)



# important params: include behaviour, include pupil, ridge reg parameters

if __name__ == "__main__":
    main()