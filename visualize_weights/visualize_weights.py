import sys
sys.path.insert(1, "../mae")
sys.path.insert(1, "../mnist_mae_vit") # Insert there to ensure the loaded mae_vit is the one in mnist_mae_vit
sys.path.insert(1, "../cifar10_task")

import argparse
import torch
from matplotlib import pyplot as plt
from mae_vit import MaskedAutoencoderViTForMNIST
from dae import ConvDenoiser
from cifar10_models import CIFAR10DenoisingAutoencoder, CustomCIFAR10MaskedAutoencoder

def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    # Count weight parameters
    num_graphs = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            num_graphs += 1

    # Create subplots
    fig, axes = plt.subplots(num_graphs, 1, figsize=(8, 4 * num_graphs))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    # plt.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(top=0.925)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="The type of model to load.",
        choices=["mnist_dae", "mnist_mae", "cifar10_dae", "cifar10_mae"],
        default="mae_vit",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to the model weights to analyze.",
        default="../mnist_mae_vit/mae_weights_vit_patch2_20epochs.pth",
    )
    args = parser.parse_args()

    if args.model == "mnist_dae":
        model = ConvDenoiser()
    elif args.model == "mnist_mae":
        model = MaskedAutoencoderViTForMNIST(
            img_size=28, patch_size=2, in_chans=1, 
            embed_dim=64, depth=4, num_heads=4, 
            decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4
        )
    elif args.model == "cifar10_dae":
        model = CIFAR10DenoisingAutoencoder()
    elif args.model == "cifar10_mae":
        model = CustomCIFAR10MaskedAutoencoder()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    plot_weight_distribution(model)
    model = model.to(device)
