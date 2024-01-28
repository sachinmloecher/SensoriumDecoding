import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt

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

    # Training data
    for idx, mouse in enumerate(['A', 'B', 'C', 'D', 'E']):
        # Train data
        X_train, y_train = [], []
        for i, batch in enumerate(train_ds[mouse]):
            X_train_batch = np.array([sample.flatten().numpy() for sample in batch['image']])
            y_train_batch = np.array([sample.flatten().numpy() for sample in batch['response']])
            X_train.extend(X_train_batch)
            y_train.extend(y_train_batch)

        X_train = np.array(X_train).flatten()
        row = idx // 3
        col = idx % 3
        axes[row, col].hist(X_train, bins=256, density=True, color='blue', alpha=0.7)
        axes[row, col].set_title(f'Mouse {mouse}')
        axes[row, col].set_xlabel('Pixel Intensity')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()