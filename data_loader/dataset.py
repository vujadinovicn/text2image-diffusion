from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),                 
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_datasets(transform):
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def get_subset(dataset, num_samples):
    return Subset(dataset, range(num_samples))

def get_dataloaders(train_dataset, test_dataset, slice=False):
    if get_subset:
        train_dataset = get_subset(train_dataset, 1000)
        test_dataset = get_subset(test_dataset, 200)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader