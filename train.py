import os
import random
import shutil
from ultralytics import YOLO


dataset_dir = "dataset/train"
balanced_dir = "dataset/balanced/train"

os.makedirs(balanced_dir, exist_ok=True)

class_images = {}
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_path):
        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if images:
            class_images[class_name] = images

min_count = min(len(imgs) for imgs in class_images.values())
print(f"Menor classe tem {min_count} imagens.")

for class_name, images in class_images.items():
    selected = random.sample(images, min_count)
    dest_dir = os.path.join(balanced_dir, class_name)
    os.makedirs(dest_dir, exist_ok=True)
    for img_path in selected:
        shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))
    print(f"{class_name}: {len(selected)} imagens copiadas.")

print("\nDataset balanceado criado em:", balanced_dir)

model = YOLO("yolo11n-cls.pt")

model.train(
    data=balanced_dir, 
    epochs=100,            
    imgsz=224             
)