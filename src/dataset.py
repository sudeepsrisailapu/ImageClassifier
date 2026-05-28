from torchvision import transforms
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision.datasets import ImageFolder
from config import CLASS_NAMES, IMAGE_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_dataloaders():
    train_dataset = ImageFolder(root=TRAIN_DIR,transform=train_transforms)
    val_dataset = ImageFolder(root=TEST_DIR,transform=val_transform)
    test_dataset = ImageFolder(root=TEST_DIR,transform=val_transform)

    class_counts = [train_dataset.targets.count(i) for i in range(len(CLASS_NAMES))]
    weights = [1.0 / class_counts[label] for label in train_dataset.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader