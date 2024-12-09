from mae_vit import MaskedAutoencoderViTForMNIST

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

def compute_cross_correlation(latent_vectors, labels, num_classes=10):
    """
    Computes the cross-correlation matrix for the latent space features.
    """
    latent_vectors = np.array(latent_vectors)
    num_features = latent_vectors.shape[1]
    
    cross_corr_matrices = {}
    for class_label in range(num_classes):
        class_indices = np.where(labels == class_label)[0]
        class_latents = latent_vectors[class_indices]
        
        # Compute correlation matrix for the current class
        mean_centered = class_latents - np.mean(class_latents, axis=0)
        correlation_matrix = np.corrcoef(mean_centered, rowvar=False)  # Shape: [num_features, num_features]
        cross_corr_matrices[class_label] = correlation_matrix
    
    return cross_corr_matrices

def visualize_cross_correlation(cross_corr_matrices, class_label):
    """
    Visualizes the cross-correlation matrices.
    """
    num_classes = len(cross_corr_matrices)
    fig, axes = plt.subplots(1, 1, figsize=(20, 5))
    
    # for class_label, matrix in cross_corr_matrices.items():
    matrix = cross_corr_matrices[class_label]
    cax = axes.matshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=axes)
    axes.set_title(f"Class {class_label}")
    axes.axis('off')
    
    plt.suptitle("Cross-Correlation Matrices for Latent Features")
    plt.show()

def visualize_latent_space_with_cluster_radius_and_correlation(mae_model, dataloader, device, num_samples_per_class=10):
    """
    Visualizes latent space using TSNE and computes cross-correlation matrices.
    """
    class_sample_count = defaultdict(int)
    latent_vectors = []
    labels = []
    mae_model.eval()

    with torch.no_grad():
        for i, (images, lbls) in enumerate(dataloader):
            if all(class_sample_count[label] >= num_samples_per_class for label in range(10)):  # 10 classes in MNIST
                break

            images = images.to(device)
            lbls = lbls.numpy()
            
            latent, _, _, _ = mae_model.forward_encoder(images, mask_ratio=0.75, return_attention=True)  # Assuming MAE has a forward_encoder method
            latent = latent.view(latent.shape[0], -1).cpu().numpy()
            
            for img, lbl in zip(latent, lbls):
                if class_sample_count[lbl] < num_samples_per_class:
                    latent_vectors.append(img)
                    labels.append(lbl)
                    class_sample_count[lbl] += 1
                    
                    if all(count >= num_samples_per_class for count in class_sample_count.values()):
                        break

    latent_vectors = np.array(latent_vectors)
    labels = np.array(labels)
    
    # Compute and visualize cross-correlation matrices
    cross_corr_matrices = compute_cross_correlation(latent_vectors, labels)
    visualize_cross_correlation(cross_corr_matrices, 1)
    
    # TSNE visualization
    reduced_vectors = TSNE(n_components=2, random_state=42).fit_transform(latent_vectors)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='tab10', s=15)
    plt.colorbar(scatter, label='Class Label')
    plt.title(f"MAE Latent Space Visualization using TSNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("data loaded")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae_model = MaskedAutoencoderViTForMNIST(
    img_size=28, patch_size=2, in_chans=1, 
    embed_dim=64, depth=4, num_heads=4, 
    decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4
).to(device)
mae_model.load_state_dict(torch.load('mae_weights_vit_patch2_20epochs.pth'))  # Load the saved weights
print("model loaded")

visualize_latent_space_with_cluster_radius_and_correlation(mae_model, test_loader, device, num_samples_per_class=100)
