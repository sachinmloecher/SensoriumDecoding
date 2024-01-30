import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

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

    # Create subplots for pixel intensities
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    fig2, axes2 = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes2 = axes2.flatten()

    response_data = []

    for idx, mouse in enumerate(['A', 'B', 'C', 'D', 'E']):
        print(mouse)
        # Train data for pixel intensities
        X_train = []
        y_train = []
        for i, batch in enumerate(train_ds[mouse]):
            print(i)
            X_train_batch = np.array([sample.flatten().numpy() for sample in batch['image']])
            y_train_batch = np.array([sample.flatten().numpy() for sample in batch['response']])
            X_train.extend(X_train_batch)
            y_train.extend(y_train_batch)
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Plot histograms for pixel intensities
        axes[idx].hist(X_train.flatten(), bins=256, density=True, color='blue', alpha=0.7)
        axes[idx].set_title(f'Mouse {mouse}')
        axes[idx].set_xlabel('Pixel Intensity')
        axes[idx].set_ylabel('Frequency')

        # Plot violinplots for responses
        sns.violinplot(data=y_train.flatten(), ax=axes2[idx], inner="quartile", color='blue', alpha=0.7)
        axes2[idx].set_title(f'Mouse {mouse}')
        axes2[idx].set_ylabel('Response (Log(1 + response))')

    # Adjust layout and save the figure
    plt.tight_layout()
    fig.savefig(os.path.join('figures/EDA/logtransform', 'pixel_intensity.png'))
    fig2.savefig(os.path.join('figures/EDA/logtransform', 'response.png'))
    plt.show()

if __name__ == "__main__":
    main()