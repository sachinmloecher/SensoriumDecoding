import torch.nn as nn
import torch

from mouseMLP import MouseMLP
from behaviourMLP import BehaviourMLP
from MLP_core import MLP_core
from mouseGCN import MouseGCN
from mouseGAT import MouseGAT
from mouseGraphConv import MouseGraphConvNet
from decoderCore import Decoder

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

    def forward(self, x, edge_index, batch, behaviours, pupil_centers, mouse_id):
        # Get output from mouse-specific MLP for the given mouse_id
        if isinstance(self.mouse_mlp_dict[mouse_id], MouseGCN) or isinstance(self.mouse_mlp_dict[mouse_id], MouseGAT) or isinstance(self.mouse_mlp_dict[mouse_id], MouseGraphConvNet):
            mouse_output = self.mouse_mlp_dict[mouse_id](x, edge_index, batch)
        else:
            mouse_output = self.mouse_mlp_dict[mouse_id](x)
        
        # Concatenate pupil centers with behaviours
        behaviours = torch.cat((behaviours, pupil_centers), dim=1).to(self.device)

        # Get output from behavior MLP if it exists
        if self.behaviour_mlp is not None:
            behaviour_output = self.behaviour_mlp(behaviours)
            combined_features = torch.add(mouse_output, behaviour_output).to(self.device)
        else:
            combined_features = mouse_output
                

        # Pass concatenated features through the core MLP
        output = self.core_mlp(combined_features)

        return output


def get_model(args):
    # Define mouse-specific MLP for each mouse
    mouse_mlp_dict = {}
    for mouse_id in ['A', 'B', 'C', 'D', 'E']:
        input_size = args.neurons_per_mouse[mouse_id]
        output_size = args.n_latent_features
        dropout_prob = args.dropout_prob
        activation = activations[args.activation]
        if args.mouse_module == 'MLP':
            hidden_sizes = args.hidden_sizes
            mouse_mlp_dict[mouse_id] = MouseMLP(input_size, hidden_sizes, output_size, dropout_prob=dropout_prob, activation=activation)
        elif args.mouse_module == 'GCN':
            layers = args.layers
            hidden_size = args.hidden_size
            mouse_mlp_dict[mouse_id] = MouseGCN(1, hidden_size, output_size, num_gcn_layers=layers, dropout=dropout_prob, activation=activation)
        elif args.mouse_module == 'GAT':
            layers = args.layers
            hidden_size = args.hidden_size
            mouse_mlp_dict[mouse_id] = MouseGAT(1, hidden_size, output_size, num_gat_layers=layers, dropout=dropout_prob, activation=activation)
        elif args.mouse_module == 'GraphConv':
            layers = args.layers
            hidden_size = args.hidden_size
            mouse_mlp_dict[mouse_id] = MouseGraphConvNet(1, hidden_size, output_size, num_graph_conv_layers=layers, dropout=dropout_prob, activation=activation)

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
    if args.core == 'MLP':
        core = MLP_core(input_size, hidden_sizes, output_size, dropout_prob=dropout_prob, activation=activation, layerNorm=layer_norm)
    elif args.core == 'Decoder':
        core = Decoder(input_size, args.downsampled_input, output_size, dropout_prob=dropout_prob, activation=activation, layerNorm=layer_norm)

    return FullModel(mouse_mlp_dict, behaviour_mlp, core, device=args.device)