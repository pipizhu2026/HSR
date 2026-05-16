import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model.hsr_dsmamba import HSRDSMamba


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f'Test Accuracy: {acc:.2f}%')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(
        root='dataset/test',
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = HSRDSMamba(num_classes=len(test_dataset.classes))
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    model = model.to(device)

    evaluate(model, test_loader, device)


if __name__ == '__main__':
    main()