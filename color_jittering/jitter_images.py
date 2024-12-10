import torch
import torchvision
import torchvision.transforms as transforms

def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
    testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,shuffle=False)
    
    return trainloader, testloader


def load_cifar10_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False)
    
    return trainloader, testloader