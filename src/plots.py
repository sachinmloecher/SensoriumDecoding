import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("C:/Users/sachi/SensoriumDecoding/src/utils")

from data import load_args, get_training_ds

# Change working directory to src
os.chdir("C:/Users/sachi/SensoriumDecoding")

class Args:
    def __init__(self, log_response=False):
        self.device = torch.device("cpu")
        self.output_dir = "runs/DNN"
        self.log_response = log_response

def main():
    mouse_ids = ['A', 'B', 'C', 'D', 'E']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), sharey=True)
    # Load and process the first dataset without log transformation
    args = Args(log_response=False)
    load_args(args)
    train_ds, _, _ = get_training_ds(
            args,
            batch_size=args.batch_size,
            data_dir=args.dataset,
            mouse_ids=args.mouse_ids,
            device=args.device,
    )
    positions = range(len(mouse_ids))
    for i, mouse in enumerate(mouse_ids):
        responses = []
        for j, batch in enumerate(train_ds[mouse]):
            responses.extend(np.array(batch['response'].flatten()))
        responses = np.array(responses)
        print(f"Mouse {mouse} responses stored")
        violin = axes[0].violinplot(
            responses,
            positions=[positions[i]],
            showmeans=False,
            showmedians=False,
            quantiles=None,  # Hide min, max, and quartiles
            vert=True,  # Vertical violin plots as requested
            widths=1,  # Adjust line width
        )

    axes[0].set_title('Responses without Log Transform', fontsize=18)
    axes[0].set_ylabel('Response', fontsize=16)
    axes[0].set_xlabel('Mouse', fontsize=16)
    axes[0].tick_params(axis='x', rotation=0, labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].set_xticks(positions)  # Set tick positions before labels
    axes[0].set_xticklabels(mouse_ids)
    del train_ds
    
    # Load and process the second dataset with log transformation
    args = Args(log_response=True)
    load_args(args)
    train_ds_log, _, _ = get_training_ds(
            args,
            batch_size=args.batch_size,
            data_dir=args.dataset,
            mouse_ids=args.mouse_ids,
            device=args.device,
    )
    for i, mouse in enumerate(mouse_ids):
        responses = []
        for j, batch in enumerate(train_ds_log[mouse]):
            responses.extend(np.array(batch['response'].flatten()))
        responses = np.array(responses)
        print(f"Mouse {mouse} responses stored")
        violin = axes[1].violinplot(
            responses,
            positions=[positions[i]],
            showmeans=False,
            showmedians=False,
            quantiles=None,  # Hide min, max, and quartiles
            vert=True,  # Vertical violin plots as requested
            widths=1,  # Adjust line width
        )

    axes[1].set_title('Responses without Log Transform', fontsize=18)
    axes[1].set_ylabel('Response (log(1 + response))', fontsize=16)
    axes[1].set_xlabel('Mouse', fontsize=16)
    axes[1].tick_params(axis='x', rotation=0, labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].set_xticks(positions)  # Set tick positions before labels
    axes[1].set_xticklabels(mouse_ids)
    del train_ds_log

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()