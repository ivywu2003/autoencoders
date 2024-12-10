import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from mae_vit import MaskedAutoencoderViTForMNIST
from dae import ConvDenoiser


class MNISTClassifier(nn.Module):
    def __init__(self, encoder_model, encoder_type):
        super().__init__()

        self.encoder_model = encoder_model
        self.encoder_type = encoder_type

        # Freeze encoder weights
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        
        # For MAE, we need to match the feature dimension from the encoder
        # if encoder_type == 'ae':
        #     input_dim = 64  # Fixed for our AE architecture
        # else:
        # Get the feature dimension from a forward pass
        dummy_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            if encoder_type == 'mae':
                features, _, _ = self.encoder_model.forward_encoder(dummy_input, mask_ratio=0.75)
                input_dim = features.shape[-1]  # Last dimension is feature dim
            elif encoder_type == 'dae':
                features = self.encoder_model.encoder(dummy_input)
                input_dim = features.view(features.size(0), -1).shape[-1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        if self.encoder_type == 'mae':
            features, _, _ = self.encoder_model.forward_encoder(x, mask_ratio=0.75)  # B x (N+1) x D
            features = features[:, 0]  # Take CLS token: B x D
        elif self.encoder_type == 'dae':
            features = self.encoder_model.encoder(x)
            features = features.view(features.size(0), -1)  # Flatten to [batch, 48*4*4]
        
        return self.classifier(features)

class ViT_Classifier(torch.nn.Module):
    def __init__(self, model, num_classes=10) -> None:
        super().__init__()
        self.cls_token = model.cls_token
        self.pos_embedding = model.pos_embed
        self.patchify = model.patchify
        self.transformer = model.blocks
        self.layer_norm = model.norm
        self.encoder = model.forward_encoder
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        # print("input image shape", img.shape)
        features, mask, ids_restore = self.encoder(img, mask_ratio = 0.75)
        # print("features shape", features.shape)
        logits = self.head(features[:,0,:])
        # print("logits shape", logits.shape)
        return logits

def get_data_loaders(batch_size=128):
    """Set up MNIST data loaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def train_classifier(model, trainloader, testloader, num_epochs=100, 
                    device='cuda', save_path='cifar10_classifier.pth'):
    """Train the classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    model = model.to(device)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss/len(pbar),
                'acc': 100.*correct/total
            })
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100.*correct/total
        print(f'Test Accuracy: {acc:.2f}%')
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f'Saved model with accuracy: {best_acc:.2f}%')
        
        scheduler.step(acc)
    
    return model

def train_vit_classifier(model, device, train_dataloader, val_dataloader, epochs):
    """Train a ViT classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            # print("outputs shape", outputs.shape)
            # print("labels shape", labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{train_loss/len(train_pbar):.3f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{val_loss/len(val_pbar):.3f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'New best validation accuracy: {best_val_acc:.2f}%')
            torch.save(model.state_dict(), 'mnist_classifier_mae_custom.pth')
        
        scheduler.step()
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_dataloader):.3f}, Train Acc: {100.*train_correct/train_total:.2f}%')
        print(f'Val Loss: {val_loss/len(val_dataloader):.3f}, Val Acc: {100.*val_correct/val_total:.2f}%')
        print('-' * 60)
    
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['ae', 'mae', 'dae'], default='ae')
    parser.add_argument('--epochs', type=int, default=20)
    # parser.add_argument('--load_pretrained', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # Load the desired encoder model
    if args.model == 'dae':
        enc_model = ConvDenoiser().to(device)
        enc_model.load_state_dict(torch.load(f'dae_weights.pth'))
        enc_model.eval()
        classifier = MNISTClassifier(encoder_model=enc_model, encoder_type=args.model).to(device)
    elif args.model == 'mae':
        enc_model = MaskedAutoencoderViTForMNIST(
                        img_size=28, patch_size=2, in_chans=1, 
                        embed_dim=64, depth=4, num_heads=4, 
                        decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4
                    ).to(device)
        enc_model.load_state_dict(torch.load(f'mae_weights_vit_patch2_20epochs.pth', map_location=device))
        classifier = ViT_Classifier(enc_model).to(device)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    
    trainloader, testloader = get_data_loaders(batch_size=128)

    # Train the classifier
    if args.model == 'ae' or args.model == 'dae':
        train_classifier(classifier, trainloader, testloader, 
                                  num_epochs=args.epochs, device=device, 
                                  save_path=f'mnist_classifier_{args.model}.pth')
    elif args.model == 'mae':
        train_vit_classifier(classifier, device, trainloader, testloader, args.epochs)