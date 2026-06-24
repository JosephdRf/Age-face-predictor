import kagglehub
import os
import random

# Download latest version
# path = kagglehub.dataset_download("frabbisw/facial-age")
DATASET_PATH = "/Users/josephderoffignac/.cache/kagglehub/datasets/frabbisw/facial-age/versions/1/face_age"
print("Path to dataset files:", DATASET_PATH)


TRAIN_RATIO = 0.8
SEED = 42


def raw_data_loader_age(dataset_path=DATASET_PATH, train_ratio=TRAIN_RATIO, seed=SEED):
    random.seed(seed)

    train_data = []
    val_data = []

    # Parcours des dossiers âge
    for age_folder in sorted(os.listdir(dataset_path)):
        age_path = os.path.join(dataset_path, age_folder)
        if not os.path.isdir(age_path):
            continue

        # Skip non-numeric folder names
        if not age_folder.isdigit():
            continue

        age = int(age_folder)  # dossier = âge
        images = os.listdir(age_path)

        # Shuffle des images pour cet âge
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for img in train_imgs:
            train_data.append((os.path.join(age_path, img), age))

        for img in val_imgs:
            val_data.append((os.path.join(age_path, img), age))

    # Shuffle global
    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    return train_data, val_data


if __name__ == "__main__":
    raw_data_loader_age()
