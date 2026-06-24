import kagglehub
import os
import random

# Download latest version
# DATASET_PATH = kagglehub.dataset_download("cashutosh/gender-classification-dataset")
DATASET_PATH = "/Users/josephderoffignac/.cache/kagglehub/datasets/cashutosh/gender-classification-dataset/versions/1"
print("Path to dataset files:", DATASET_PATH)

SEED = 42


def raw_data_loader_gender(dataset_path=DATASET_PATH, seed=SEED):
    random.seed(seed)

    train_data = []
    val_data = []

    # Parcours des dossiers genre
    for dataset_path in os.listdir(DATASET_PATH):
        complete_dataset_path = os.path.join(DATASET_PATH, dataset_path)
        for gender_folder in sorted(os.listdir(complete_dataset_path)):
            complete_gender_path = os.path.join(complete_dataset_path, gender_folder)

            true_gender = 0 if gender_folder == "female" else 1

            for img in os.listdir(complete_gender_path):
                complete_img_path = os.path.join(complete_gender_path, img)

                if dataset_path.endswith("Training"):
                    train_data.append((complete_img_path, true_gender))
                else:
                    val_data.append((complete_img_path, true_gender))

    # Shuffle global
    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    return train_data, val_data


if __name__ == "__main__":
    raw_data_loader_gender()
