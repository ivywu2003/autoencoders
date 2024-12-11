import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from einops import repeat, rearrange
from tqdm import tqdm

from jitter_images import load_mnist_data, load_cifar10_data
from evaluation.loaders import load_jittered_dae, load_jittered_mae
from evaluation.loaders import load_jittered_cifar10_for_dae, load_jittered_cifar10_for_mae

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class CIFAR10_DAE_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        dae = load_jittered_dae()
        self.encoder_model = dae.encoder

        # Freeze encoder weights
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            features = self.encoder_model(dummy_input)
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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.encoder_model(x)
        features = features.view(features.size(0), -1)  # Flatten to [batch, 48*4*4]
        
        return self.classifier(features)

class CIFAR10_MAE_Classifier(torch.nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        
        mae = load_jittered_mae()
        self.encoder_model = mae

        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            features, _, _ = self.encoder_model.forward_encoder(dummy_input, mask_ratio=0.75)
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
            nn.Linear(128, num_classes)
        )

    def forward(self, img):
        features, _, _ = self.encoder_model.forward_encoder(img, mask_ratio=0.75)  # B x (N+1) x D
        features = features.view(features.size(0), -1)  # Flatten
        # features = features[:, 0]  # Take CLS token: B x D
        
        return self.classifier(features) 

def train_classifier(model, trainloader, testloader, num_epochs=10, 
                    device='cuda', save_path='cifar10_classifier.pth'):
    """Train the classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
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


if __name__ == "__main__":
    classifier = CIFAR10_MAE_Classifier(num_classes=10)

    trainloader, testloader = load_jittered_cifar10_for_mae()

    classifier = train_classifier(classifier, trainloader, testloader, num_epochs=10, save_path='color_jittering/cifar10_mae_classifier.pth')