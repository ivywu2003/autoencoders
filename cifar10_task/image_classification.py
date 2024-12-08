import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# CIFAR-10 classes
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class CIFAR10Classifier(nn.Module):
    def __init__(self, encoder_model, encoder_type):
        super().__init__()

        self.encoder_model = encoder_model
        self.encoder_type = encoder_type

        # Freeze encoder weights
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        
        # Classification head for features of shape Bx64
        self.classifier = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        if self.encoder_type == 'ae':
            features = self.encoder_model.encoder(x)  # B x 64 x 1 x 1
            features = features.squeeze(-1).squeeze(-1)  # B x 64
        elif self.encoder_type == 'mae':
            features, _, _ = self.encoder_model.forward_encoder(x, 0)  # B x 65 x 64
            features = features[:, 0]  # Take CLS token: B x 64
        
        return self.classifier(features)

def get_data_loaders(batch_size=128):
    """Set up CIFAR-10 data loaders"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False)
    
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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['ae', 'mae'], default='ae')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # Load the desired encoder model
    if args.model == 'ae':
        enc_model = CIFAR10Autoencoder().to(device)
    elif args.model == 'mae':
        enc_model = CIFAR10MaskedAutoencoder().to(device)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    enc_model.load_state_dict(torch.load(f'cifar10_{args.model}_weights.pth'))
    enc_model.eval()

    trainloader, testloader = get_data_loaders(batch_size=128)

    # Train the classifier
    classifier = CIFAR10Classifier(encoder=enc_model, encoder_type=args.model).to(device)
    classifier = train_classifier(classifier, trainloader, testloader, 
                                  num_epochs=args.epochs, device=device, 
                                  save_path=f'cifar10_classifier_{args.model}.pth')