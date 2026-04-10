'''This script will: 

Move images into train, val, and test categories
Separate images into clean and stego folders
Remove any empty directories
Ensure an optimized dataset structure before training

This makes sure that your dataset is in the correct format before running the training model.'''

import os
import shutil
from tqdm import tqdm

DATASET_PATH = "C:/Users/PC1-SVO2/Downloads/stego-images-dataset"


TRAIN_SRC = os.path.join(DATASET_PATH, "train", "train")
VAL_SRC = os.path.join(DATASET_PATH, "val", "val")
TEST_SRC = os.path.join(DATASET_PATH, "test", "test")


TRAIN_DEST = os.path.join(DATASET_PATH, "train")
VAL_DEST = os.path.join(DATASET_PATH, "val")
TEST_DEST = os.path.join(DATASET_PATH, "test")


CATEGORIES = ["clean", "stego"]


for subset in [TRAIN_DEST, VAL_DEST, TEST_DEST]:
    for category in CATEGORIES:
        os.makedirs(os.path.join(subset, category), exist_ok=True)

def move_images(source_folder, dest_folder):
    """Moves image files from source to destination while maintaining structure."""
    if os.path.exists(source_folder):
        for file in tqdm(os.listdir(source_folder), desc=f"Moving files from {source_folder}"):
            file_path = os.path.join(source_folder, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.png', '.jpeg')):
                shutil.move(file_path, dest_folder)

move_images(os.path.join(TRAIN_SRC, "clean"), os.path.join(TRAIN_DEST, "clean"))
move_images(os.path.join(TRAIN_SRC, "stego"), os.path.join(TRAIN_DEST, "stego"))

move_images(os.path.join(VAL_SRC, "clean"), os.path.join(VAL_DEST, "clean"))
move_images(os.path.join(VAL_SRC, "stego"), os.path.join(VAL_DEST, "stego"))

move_images(os.path.join(TEST_SRC, "clean"), os.path.join(TEST_DEST, "clean"))
move_images(os.path.join(TEST_SRC, "stego"), os.path.join(TEST_DEST, "stego"))


def delete_empty_folders(directory):
    """Deletes any empty folders in the dataset."""
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

delete_empty_folders(DATASET_PATH)
print("✅ Dataset has been organized successfully!")
