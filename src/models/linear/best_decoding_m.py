import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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

    fig, ax = plt.subplots(figsize=(15, 10))

    ms = [100, 200, 500, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500]
    mouse_ids = ['A', 'B', 'C', 'D', 'E']
    average_correlations = []
    for m in ms:
        mouse_correlations = []
        for idx, mouse in enumerate(mouse_ids):
            X_test = []
            for i, batch in enumerate(test_ds[mouse]):
                X_test_batch = np.array([sample.flatten().numpy() for sample in batch['image']])
                X_test.extend(X_test_batch)
            
            X_test = np.array(X_test)

            # Load predicted X_test
            X_test_pred = np.load(f"output/downsampled/m_{m}/X_test_pred_{mouse}.npy")
            print(f"X_test_pred shape for m={m}, mouse={mouse}:", X_test_pred.shape)

            # Pixel correlations
            pixel_correlations = [pearsonr(X_test[:, i], X_test_pred[:, i])[0] for i in range(X_test.shape[1])]
            mouse_correlations.append(np.mean(pixel_correlations))

        average_correlations.append(np.mean(mouse_correlations))
        print(f"Average pixel correlation for m={m}:", np.mean(mouse_correlations))
    fig.suptitle("Average pixel correlation by m")
    ax.plot(ms, average_correlations, marker='o', color='b', linestyle='-')
    ax.set_xlabel('m')
    ax.set_ylabel('Average Pixel-wise Correlation')
    ax.set_ylim(-0.1, 1)
    ax.grid(True)
    plt.show()
    plt.savefig("figures/linear/downsampled/average_pixel_correlation_by_m.png")

if __name__ == "__main__":
    main()