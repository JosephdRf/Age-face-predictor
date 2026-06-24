import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from data_preprocessing_gender import raw_data_loader_gender


class FaceGenderDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gender = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        gender = torch.tensor(gender, dtype=torch.float32)
        return image, gender


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

train_data, val_data = raw_data_loader_gender()

train_dataset = FaceGenderDataset(train_data, train_transform)
val_dataset = FaceGenderDataset(val_data, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)


import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class GenderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = GenderClassifier().to(device)

criterion = (
    nn.BCEWithLogitsLoss()
)  # Binary cross-entropy with logits for gender classification
optimizer = optim.AdamW(model.parameters(), lr=1e-4)


def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, genders in pbar:
        images = images.to(device)
        genders = genders.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, genders)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Display current batch loss
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    running_loss = 0.0

    for images, genders in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        genders = genders.to(device)

        preds = model(images)
        loss = criterion(preds, genders)
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


if __name__ == "__main__":
    EPOCHS = 20
    best_val_loss = float("inf")

    torch.save(
        model.state_dict(),
        "model_gender_checkpoints/gender_classifier_best.pth",
    )

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val   Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                "model_gender_checkpoints/gender_classifier_best.pth",
            )
        torch.save(
            model.state_dict(),
            f"model_gender_checkpoints/gender_classifier_epoch_{epoch + 1}.pth",
        )
