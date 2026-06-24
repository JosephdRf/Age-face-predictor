import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from data_preprocessing_age import raw_data_loader_age
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class FaceAgeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        age = torch.tensor(age, dtype=torch.float32)
        return image, age


train_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

from torch.utils.data import DataLoader

train_data, val_data = raw_data_loader_age()

train_dataset = FaceAgeDataset(train_data, train_transform)
val_dataset = FaceAgeDataset(val_data, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)


import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class AgeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = AgeEstimator().to(device)

criterion = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)


def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, ages in pbar:
        images = images.to(device)
        ages = ages.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, ages)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Display current batch loss
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()

    all_true = []
    all_pred = []

    for images, ages in tqdm(loader, desc="Collecting preds", leave=False):
        images = images.to(device)
        ages = ages.to(device)

        preds = model(images)

        all_true.extend(ages.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())

    return np.array(all_true), np.array(all_pred)


def plot_age_distribution_2d(y_true, y_pred, bins=40, epoch=0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    min_age = min(y_true.min(), y_pred.min())
    max_age = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(7, 7))

    h = plt.hist2d(
        y_true,
        y_pred,
        bins=bins,
        range=[[min_age, max_age], [min_age, max_age]],
        norm=LogNorm(),  # échelle logarithmique
        cmap="viridis",
    )

    plt.colorbar(h[3], label="Image number (log)")

    # Ligne idéale
    plt.plot(
        [min_age, max_age], [min_age, max_age], "r--", lw=2, label="Prédiction parfaite"
    )

    # Axes égaux
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("True age")
    plt.ylabel("Predicted age")
    plt.title("True age vs Predicted age, Epoch " + str(epoch + 1))
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)  # Wait 10 seconds to view
    plt.close()


@torch.no_grad()
def validate(model, loader, epoch):
    model.eval()
    true_ages, pred_ages = collect_predictions(model, loader)
    plot_age_distribution_2d(true_ages, pred_ages, bins=30, epoch=epoch)
    loss = criterion(torch.tensor(pred_ages), torch.tensor(true_ages))
    return loss.item()


if __name__ == "__main__":
    EPOCHS = 20
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader)
        val_loss = validate(model, val_loader, epoch)

        print(f"Train MAE: {train_loss:.2f}")
        print(f"Val   MAE: {val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), "model_age_checkpoints/age_estimator_best.pth"
            )
        torch.save(
            model.state_dict(),
            f"model_age_checkpoints/age_estimator_epoch_{epoch + 1}.pth",
        )
