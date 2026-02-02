import os
import shutil
import random
import json

# Get all images with labels
images_dir = 'tracknet_data/train/images'
labels_dir = 'tracknet_data/train/labels'
annotations_file = 'dataset/labels/train/annotations.json'  # your JSON file

# Load JSON annotations
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

def frame_idx_from_name(filename):
    return str(int(filename.replace('frame_', '').replace('.jpg', '')))

images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
labels = [f.replace('.jpg', '.png') for f in images 
          if os.path.exists(os.path.join(labels_dir, f.replace('.jpg', '.png')))]

# Only use images that have labels
valid_images = [f for f in images 
                if os.path.exists(os.path.join(labels_dir, f.replace('.jpg', '.png')))]

print(f"Found {len(valid_images)} images with labels")
# Shuffle and split (80% train, 20% val)
random.shuffle(valid_images)
split_idx = int(len(valid_images) * 0.8)

train_images = valid_images[:split_idx]
val_images = valid_images[split_idx:]

# train_frame_ids = set(frame_idx_from_name(f) for f in train_images)
# val_frame_ids   = set(frame_idx_from_name(f) for f in val_images)

train_annotations = {}
val_annotations = {}

for k, v in annotations.items():
    if k in train_images:
        train_annotations[k] = v
    elif k in val_images:
        val_annotations[k] = v

print("Train images: ", len(train_images))
print("Val images: ", len(val_images))
print("Train annotations: ", len(train_annotations))
print("Val annotations: ", len(val_annotations))

os.makedirs('tracknet_data/val/images', exist_ok=True)
os.makedirs('tracknet_data/val/labels', exist_ok=True)

for img in val_images:
    label = img.replace('.jpg', '.png')
    
    shutil.move(os.path.join(images_dir, img), os.path.join('tracknet_data/val/images', img))
    shutil.move(os.path.join(labels_dir, label), os.path.join('tracknet_data/val/labels', label))

# Split JSON annotations

with open('tracknet_data/train/annotations_train.json', 'w') as f:
    json.dump(train_annotations, f, indent=2)

with open('tracknet_data/val/annotations_val.json', 'w') as f:
    json.dump(val_annotations, f, indent=2)

print("Train annotations:", len(train_annotations))
print("Val annotations:", len(val_annotations))

print(f"Done! Validation set ready with {len(val_images)} images")