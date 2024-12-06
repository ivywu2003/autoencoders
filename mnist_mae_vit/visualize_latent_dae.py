from dae import ConvDenoiser
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict

def visualize_latent_space_dae_with_cluster_radius(dae_model, dataloader, device, method='tsne', num_samples_per_class=10):
    """
    Visualize the latent space of a DAE using t-SNE or PCA and calculate cluster radii for each class.
    
    Args:
        dae_model: Trained DAE model.
        dataloader: DataLoader providing input images and labels.
        device: Device (CPU or GPU).
        method: Dimensionality reduction method ('tsne' or 'pca').
        num_samples_per_class: Number of samples to use for each class.
    
    Returns:
        None (plots the latent space visualization and prints cluster radii).
    """
    # Initialize a dictionary to track the number of samples per class
    class_sample_count = defaultdict(int)
    latent_vectors = []
    labels = []
    dae_model.eval()
    
    with torch.no_grad():
        for i, (images, lbls) in enumerate(dataloader):
            # Only collect samples if there are still classes with less than 10 samples
            if all(class_sample_count[label] >= num_samples_per_class for label in class_sample_count.keys()) and \
               len(set(lbls.numpy())) == 0:  # No more classes to collect
                break
            
            images = images.to(device)
            lbls = lbls.numpy()
            print(sorted(lbls))
            # Pass through the encoder to get latent representations
            _, latent = dae_model(images)  # Encoder output is the latent space
            latent = latent.view(latent.size(0), -1).cpu().numpy()  # Flatten the latent space
            
            # Only collect 10 samples per class
            for img, lbl in zip(latent, lbls):
                if class_sample_count[lbl] < num_samples_per_class:
                    latent_vectors.append(img)
                    labels.append(lbl)
                    class_sample_count[lbl] += 1
                    
                    # Stop if we've already collected 10 samples per class
                    if all(count >= num_samples_per_class for count in class_sample_count.values()):
                        break
    
    # Concatenate all latent vectors and labels
    latent_vectors = np.array(latent_vectors)
    labels = np.array(labels)
    
    # --- Direct calculation of clusters and radii from the latent space ---
    class_points_original = defaultdict(list)
    for point, label in zip(latent_vectors, labels):
        class_points_original[label].append(point)
    
    cluster_centers_original = {}
    cluster_radii_original = {}
    for label, points in class_points_original.items():
        points = np.array(points)
        cluster_center = points.mean(axis=0)
        distances = np.linalg.norm(points - cluster_center, axis=1)
        cluster_radius = distances.mean()  # Average distance to center
        cluster_centers_original[label] = cluster_center
        cluster_radii_original[label] = cluster_radius
        print(f"Class {label} (Original Latent Space): Cluster Radius = {cluster_radius:.4f}")
    
    # --- Dimensionality reduction for visualization ---
    if method == 'tsne':
        reduced_vectors = TSNE(n_components=2, random_state=42).fit_transform(latent_vectors)
    elif method == 'pca':
        reduced_vectors = PCA(n_components=2).fit_transform(latent_vectors)
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'pca'.")
    
    # --- Calculate cluster radii in the reduced space ---
    class_points_reduced = defaultdict(list)
    for point, label in zip(reduced_vectors, labels):
        class_points_reduced[label].append(point)
    
    cluster_radii_reduced = {}
    for label, points in class_points_reduced.items():
        points = np.array(points)
        cluster_center = points.mean(axis=0)
        distances = np.linalg.norm(points - cluster_center, axis=1)
        cluster_radius = distances.mean()
        cluster_radii_reduced[label] = cluster_radius
        print(f"Class {label} (Reduced Space): Cluster Radius = {cluster_radius:.4f}")
    
    # --- Plot the reduced latent space ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='tab10', s=15)
    plt.colorbar(scatter, label='Class Label')
    plt.title(f"DAE Latent Space Visualization using {method.upper()}")
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("data loaded")

dae_model = ConvDenoiser()  # Initialize the model
dae_model.load_state_dict(torch.load('dae_weights.pth')) 
print("model loaded")

visualize_latent_space_dae_with_cluster_radius(dae_model, test_loader, device, method='tsne', num_samples_per_class=50)
