import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from cifar10_models import CIFAR10Autoencoder, CustomCIFAR10MaskedAutoencoder
from sklearn.decomposition import PCA



def visualize_latent_space_with_cluster_radius(mae_model, dataloader, device, num_samples_per_class=10):

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
            
            latent, _, = mae_model.encoder(images)  # Assuming MAE has a forward_encoder method
            latent = torch.swapaxes(latent, 0,1)
            latent = latent[:, 1:, :]  # Ignore [CLS] token, shape: [B, num_patches, embed_dim]
            print("latent shape", latent.shape)
            latent = latent.view(latent.shape[0], -1).cpu().numpy()
            for img, lbl in zip(latent, lbls):
                if class_sample_count[lbl] < num_samples_per_class:
                    # print("img shape", img.shape)
                    latent_vectors.append(img)
                    labels.append(lbl)
                    class_sample_count[lbl] += 1
                    
                    if all(count >= num_samples_per_class for count in class_sample_count.values()):
                        break

    latent_vectors = np.array(latent_vectors)
    labels = np.array(labels)
    
    class_points_original = defaultdict(list)
    for point, label in zip(latent_vectors, labels):
        class_points_original[label].append(point)
    
    cluster_centers_original = {}
    cluster_radii_original = {}
    for label, points in class_points_original.items():
        points = np.array(points)
        print("mae point shpae", points.shape)
        cluster_center = points.mean(axis=0)
        distances = np.linalg.norm(points - cluster_center, axis=1)
        cluster_radius = distances.mean()  # Average distance to center
        cluster_centers_original[label] = cluster_center
        cluster_radii_original[label] = cluster_radius
        print(f"Class {label} (Original Latent Space): Cluster Radius = {cluster_radius:.4f}")
    
    reduced_vectors = TSNE(n_components=2, random_state=42).fit_transform(latent_vectors)
    
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
    

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='tab10', s=15)
    plt.colorbar(scatter, label='Class Label')
    plt.title(f"MAE Latent Space Visualization using TSNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                       (0.2023, 0.1994, 0.2010))
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=8, shuffle=True)
print("data loaded")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae_model = model = CustomCIFAR10MaskedAutoencoder().to(device)
mae_model.load_state_dict(torch.load('cifar10_mae_weights_10_epochs_custom.pth',  map_location=torch.device('cpu')))  # Load the saved weights
print("model loaded")

visualize_latent_space_with_cluster_radius(mae_model, testloader, device, num_samples_per_class=100)


