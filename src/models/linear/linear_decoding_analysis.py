import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import ttest_1samp

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

def reshape_to_original(image_flat):
    return image_flat.reshape((args.target_size[0], args.target_size[1]))


def main():
    train_ds, val_ds, test_ds = get_training_ds(
        args,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        data_dir=args.dataset,
        device=args.device,
    )

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig2, axes2 = plt.subplots(figsize=(8, 6))
    for ax in axes.flatten():
        ax.axis('off')

    mouse_ids = ['A', 'B', 'C', 'D', 'E']
    average_correlations = []
    std_error = []
    significance = {}
    for idx, mouse in enumerate(mouse_ids):
        X_test = []
        for i, batch in enumerate(test_ds[mouse]):
            X_test_batch = np.array([sample.flatten().numpy() for sample in batch['image']])
            X_test.extend(X_test_batch)
        
        X_test = np.array(X_test)

        # Load predicted X_test
        X_test_pred = np.load(f"output/m_1000_with_behaviour/X_test_pred_{mouse}.npy")
        print("X_test_pred shape:", X_test_pred.shape)

        # Pixel correlations
        pixel_correlations = [pearsonr(X_test[:, i], X_test_pred[:, i])[0] for i in range(X_test.shape[1])]
        print("Average pixel correlation:", np.mean(pixel_correlations))
        average_correlations.append(np.mean(pixel_correlations))

        significance[mouse] = ttest_1samp(pixel_correlations, 0)

        std_error.append(np.std(pixel_correlations) / np.sqrt(len(pixel_correlations)))

        sample_correlations = [pearsonr(X_test[i, :], X_test_pred[i, :])[0] for i in range(X_test.shape[0])]
        print("Average sample correlation:", np.mean(sample_correlations))

        best_index = np.argmax(sample_correlations)
        print("Best sample correlation:", sample_correlations[best_index])

        # Image reconstruction examples
        original = reshape_to_original(X_test[best_index])
        reconstructed = reshape_to_original(X_test_pred[best_index])

        # Plot the original image
        axes[0, idx].imshow(original, cmap='gray')
        axes[0, idx].set_title(f'Mouse {mouse}')

        # Plot the reconstructed image
        axes[1, idx].imshow(reconstructed, cmap='gray')
    
    print("Average pixel correlations:", np.mean(average_correlations))
    for mouse in mouse_ids:
        t_val, p_val = significance[mouse]
        print(f"Significance test for mouse {mouse}: t = {t_val}, p = {p_val}")

    # Add titles for each column
    for i, mouse in enumerate(['A', 'B', 'C', 'D', 'E']):
        axes[0, i].set_title(f'Mouse {mouse}', fontsize=16)

    # Add titles for each row
    fig.text(0.03, 0.70, 'Original', va='center', rotation='vertical', fontsize=16)
    fig.text(0.03, 0.33, 'Reconstruction', va='center', rotation='vertical', fontsize=16)

    # Add figure title
    fig.suptitle('Best Linear Image Reconstructions with behavioural features (m=1000)', fontsize=20)

    fig.tight_layout(rect=[0.06, 0.06, 1, 1], pad=0.5, h_pad=0.5, w_pad=0.5)

    # Bar plot of average pixel correlations
    axes2.bar(mouse_ids, average_correlations, yerr=std_error, color=['blue', 'green', 'red', 'purple', 'orange'], capsize=5)
    axes2.set_xlabel('Mouse')
    axes2.set_ylabel('Average Pixel-wise Correlation (r)')
    axes2.set_title('Average Pixel-wise Correlation for Each Mouse with behavioural features (m=1000)')
    axes2.set_ylim([0, 1])

    plt.show()
    fig.savefig("figures/linear/with_behaviour/linear_reconstructions.png")
    fig2.savefig("figures/linear/with_behaviour/average_pixel_correlations.png")

if __name__ == "__main__":
    main()
