import os
import shutil
from pathlib import Path
import random
from math import floor

def ensure_dir_exists(dir_path):
    """Ensure the directory exists. If not, create it."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def split_data(source_dir, train_dir, val_dir, test_dir, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    # Ensure the split directories exist
    ensure_dir_exists(train_dir)
    ensure_dir_exists(val_dir)
    ensure_dir_exists(test_dir)
    
    for category in os.listdir(source_dir):
        category_path = Path(source_dir) / category
        if category_path.is_dir():
            files = list(category_path.glob('*'))
            random.shuffle(files)
            train_end = floor(train_prop * len(files))
            val_end = train_end + floor(val_prop * len(files))

            train_files = files[:train_end]
            val_files = files[train_end:val_end]
            test_files = files[val_end:]

            for file in train_files:
                dest_path = Path(train_dir) / category
                ensure_dir_exists(dest_path)
                shutil.copy2(file, dest_path / file.name)
                print(f"Copied {file} to {dest_path / file.name}")
            
            for file in val_files:
                dest_path = Path(val_dir) / category
                ensure_dir_exists(dest_path)
                shutil.copy2(file, dest_path / file.name)
                print(f"Copied {file} to {dest_path / file.name}")
            
            for file in test_files:
                dest_path = Path(test_dir) / category
                ensure_dir_exists(dest_path)
                shutil.copy2(file, dest_path / file.name)
                print(f"Copied {file} to {dest_path / file.name}")

if __name__ == "__main__":
    data_dir = '/Users/justinhuang/Documents/Developer/ML/medVizML3/medVizData_unsplit'

    # Define paths for new dirs
    split_train_dir = '/Users/justinhuang/Documents/Developer/ML/medVizML3/medVizData_split/train'
    split_val_dir = '/Users/justinhuang/Documents/Developer/ML/medVizML3/medVizData_split/valid'
    split_test_dir = '/Users/justinhuang/Documents/Developer/ML/medVizML3/medVizData_split/test'
    
    # Split the combined data
    split_data(data_dir, split_train_dir, split_val_dir, split_test_dir, train_prop=0.7, val_prop=0.15, test_prop=0.15)
