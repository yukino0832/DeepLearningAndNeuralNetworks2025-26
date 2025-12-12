import os
import shutil
import random
import argparse

def split_dataset(target_dir, split_ratio=0.8):
    """
    Splits images in each class folder into train and val subfolders.
    """
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    for class_name in os.listdir(target_dir):
        class_dir = os.path.join(target_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue

        print(f"Processing class: {class_name}")

        train_dir = os.path.join(class_dir, 'train')
        val_dir = os.path.join(class_dir, 'val')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        images = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
        
        for f in os.listdir(class_dir):
            file_path = os.path.join(class_dir, f)
            if os.path.isfile(file_path) and f.lower().endswith(valid_extensions):
                images.append(f)

        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_dir, img)
            shutil.move(src, dst)
            
        for img in val_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_dir, img)
            shutil.move(src, dst)

        print(f"  - Total images: {len(images)}")
        print(f"  - Train: {len(train_images)}")
        print(f"  - Val: {len(val_images)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train and val folders.')
    parser.add_argument('--target_dir', type=str, required=True, help='Path to the target directory containing class folders')
    
    args = parser.parse_args()
    
    split_dataset(args.target_dir)
