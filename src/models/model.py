import torch.nn as nn
import torch

from mouseMLP import MouseMLP
from behaviourMLP import BehaviourMLP
from MLP_core import MLP_core

activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
}

# Combine mouse-specific MLP outputs and behavior MLP output
class FullModel(nn.Module):
    def __init__(self, mouse_mlp_dict, behaviour_mlp, core_mlp, device='cuda'):
        super(FullModel, self).__init__()
        self.device = device
        self.mouse_mlp_dict = {mouse_id: mouse_mlp.to(device) for mouse_id, mouse_mlp in mouse_mlp_dict.items()}
        self.behaviour_mlp = behaviour_mlp.to(device) if behaviour_mlp is not None else None
        self.core_mlp = core_mlp.to(device)

    def forward(self, x, behaviours, pupil_centers, mouse_id):
        # Get output from mouse-specific MLP for the given mouse_id
        mouse_output = self.mouse_mlp_dict[mouse_id](x)

        # Get output from behavior MLP if it exists
        if self.behaviour_mlp is not None:
            behaviour_output = self.behaviour_mlp(behaviours)
        else:
            behaviour_output = torch.tensor([]).to(self.device)

        # Concatenate mouse and behavior outputs
        combined_features = torch.cat((mouse_output, behaviour_output), dim=1)

        # Pass concatenated features through the core MLP
        output = self.core_mlp(combined_features)

        return output


def get_model(args):
    # Define mouse-specific MLP for each mouse
    mouse_mlp_dict = {}
    for mouse_id in ['A', 'B', 'C', 'D', 'E']:
        input_size = args.neurons_per_mouse[mouse_id]
        hidden_sizes = args.hidden_sizes
        output_size = args.n_latent_features
        dropout_prob = args.dropout_prob
        activation = activations[args.activation]
        mouse_mlp_dict[mouse_id] = MouseMLP(input_size, hidden_sizes, output_size, dropout_prob=dropout_prob, activation=activation)
    
    # Define behaviour MLP
    behaviour_mlp = None
    if args.behavior_mode:
        input_size = args.input_size_beh
        hidden_sizes = args.hidden_sizes_beh
        output_size = args.n_latent_features
        dropout_prob = args.dropout_prob_beh
        activation = activations[args.activation_beh]
        behaviour_mlp = BehaviourMLP(input_size, hidden_sizes, output_size, dropout_prob=dropout_prob, activation=activation)
    
    # Define core MLP
    input_size = args.n_latent_features
    hidden_sizes = args.hidden_sizes_mlp
    output_size = args.output_size
    dropout_prob = args.dropout_prob_mlp
    layer_norm = args.layer_norm
    activation = activations[args.activation_mlp]
    core_mlp = MLP_core(input_size, hidden_sizes, output_size, dropout_prob=dropout_prob, activation=activation, layerNorm=layer_norm)
    
    return FullModel(mouse_mlp_dict, behaviour_mlp, core_mlp, device=args.device)