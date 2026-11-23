"""
Training Script for Devanagari Full Character Recognition Model.
Trains a lightweight 3-stage CNN on 64x64 single-channel glyphs.
Default input dataset: model/fullcharnew/dataset (train/ and test/)
Default model output:  model/fullcharnew/fullcharnew.pth
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "dataset")
DEFAULT_MODEL_OUT = os.path.join(BASE_DIR, "full_character.pth")


class FullCharacterCNN(nn.Module):
    """
    3-stage CNN for 64x64 Devanagari full-character glyphs.
    Follows Kore et al. (ELCVIA 2025):
      Conv(32) -> MaxPool -> Conv(64) -> MaxPool -> Conv(128) -> MaxPool -> Linear(256) -> Linear(128) -> Out
    """

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.20),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.30),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_dataloaders(data_dir, batch_size=64, num_workers=0):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: 1.0 - tensor),  # Invert: ink = 1.0, paper = 0.0
        ]
    )

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Missing train or test folder in '{data_dir}'")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.classes


def train(data_dir=DEFAULT_DATA_DIR, model_out=DEFAULT_MODEL_OUT, epochs=20, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading data from: {data_dir}")
    train_loader, test_loader, classes = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    model = FullCharacterCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=f"{train_correct / total_train:.3f}")

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train

        # 2. Testing Phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                total_test += labels.size(0)

        epoch_test_loss = test_loss / total_test
        epoch_test_acc = test_correct / total_test

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
            f"Test Loss: {epoch_test_loss:.4f} Acc: {epoch_test_acc:.4f}"
        )

        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "num_classes": num_classes,
                    "best_test_acc": best_test_acc,
                    "epoch": epoch,
                },
                model_out,
            )
            print(f"  -> Saved new best model to '{model_out}' (Acc: {best_test_acc:.4f})")

    print(f"\nTraining Complete. Best Test Accuracy: {best_test_acc:.4f}")
    return model, best_test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Devanagari full-character CNN model.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to dataset root with train/ and test/")
    parser.add_argument("--output", default=DEFAULT_MODEL_OUT, help="Path to save trained weights .pth")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_out=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
