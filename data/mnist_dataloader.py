from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    def __init__(self, split="train", allowed_classes=[3]):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), # Normalize to [-1, 1]
            transforms.Resize((32, 32))
            ])

        if split == "train":
            train = True
        else:
            train = False

        self.mnist_data = datasets.MNIST(root='./data/download', train=train, download=True, transform=self.transform)
        # Filter to only include allowed classes
        self.mnist_data = [(img, label) for img, label in self.mnist_data if label in allowed_classes]

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]
        return image, label

def get_mnist_dataloader(batch_size=64, split="train", allowed_classes=[3]):
    dataset = MNISTDataset(split=split, allowed_classes=allowed_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"))
    return dataloader

# Testing the MNIST dataloader
if __name__ == "__main__":
    train_loader = get_mnist_dataloader(batch_size=32, split="train")
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        print((images[0].max(), images[0].min()))
        print((images[2].max(), images[2].min()))
        print((images[4].max(), images[4].min()))
        print((images[30].max(), images[30].min()))
        break