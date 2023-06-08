import os
import random
import shutil

dataset_path = '/home/sjp/lh/labeled_dataset'

train_ratio =0.7
val_ratio = 0.2
test_ratio = 0.1

output_dir = '/home/sjp/lh/yolov5/mydata2'

os.makedirs(output_dir,exist_ok=True)
image_files = [file for file in os.listdir(dataset_path) if file.endswith('.jpg')]
label_files = [file for file in os.listdir(dataset_path) if file.endswith('.txt')]

num_samples = len(image_files)
num_train = int(num_samples * train_ratio)
num_val = int(num_samples * val_ratio)
num_test = num_samples - num_train - num_val

random.shuffle(image_files)

train_files = image_files[:num_train]
val_files = image_files[num_train:num_train+num_val]
test_files = image_files[num_train+num_val:]

for file in train_files:
    image_path = os.path.join(dataset_path, file)
    label_path = os.path.join(dataset_path, file.replace('.jpg', '.txt'))
    shutil.copy(image_path, os.path.join(output_dir, 'train/images'))
    shutil.copy(label_path, os.path.join(output_dir, 'train/labels'))

for file in val_files:
    image_path = os.path.join(dataset_path, file)
    label_path = os.path.join(dataset_path, file.replace('.jpg', '.txt'))
    shutil.copy(image_path, os.path.join(output_dir, 'val/images'))
    shutil.copy(label_path, os.path.join(output_dir, 'val/labels'))

for file in test_files:
    image_path = os.path.join(dataset_path, file)
    label_path = os.path.join(dataset_path, file.replace('.jpg', '.txt'))
    shutil.copy(image_path, os.path.join(output_dir, 'test/images'))
    shutil.copy(label_path, os.path.join(output_dir, 'test/labels'))

