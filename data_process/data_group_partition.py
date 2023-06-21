import os
import random
import shutil

input_folder = "/Users/lihui/Desktop/labeled_dataset"
output_folder = "/Users/lihui/Desktop/result"

os.makedirs(output_folder,exist_ok=True)

# 创建类别文件夹
for class_name in ["yellow", "red", "green"]:
    class_folder = os.path.join(output_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".jpg"):
        file_path = os.path.join(input_folder, file_name)
        label_file = os.path.join(input_folder, file_name[:-4] + ".txt")

        if os.path.isfile(label_file):
            with open(label_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    label_line = lines[0].strip()
                    labels = label_line.split(' ')
                    if len(labels) > 0:
                        label = int(labels[0])

                        if label == 0:
                            class_name = "red"
                        elif label == 1:
                            class_name = "green"
                        elif label == 2:
                            class_name = "yellow"
                        else:
                            continue

                        class_folder = os.path.join(output_folder, class_name)
                        shutil.copy(file_path, class_folder)
                        shutil.copy(label_file, class_folder)

print("分组完成。")



input_folder = "/Users/lihui/Desktop/result"
output_folder = "/Users/lihui/Desktop/dataset"

# 创建训练集、验证集、测试集文件夹
for split_name in ["train", "val", "test"]:
    split_folder = os.path.join(output_folder, split_name)
    os.makedirs(os.path.join(split_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(split_folder, "labels"), exist_ok=True)

class_names = ["yellow", "red", "green"]
split_ratios = [0.7, 0.15, 0.15]  # 训练集、验证集、测试集的划分比例

# 分别处理每个类别
for class_name in class_names:
    class_folder = os.path.join(input_folder, class_name)
    images = [f for f in os.listdir(class_folder) if f.endswith(".jpg")]
    random.shuffle(images)

    # 计算划分的数量
    total_images = len(images)
    train_split = int(total_images * split_ratios[0])
    val_split = int(total_images * split_ratios[1])
    test_split = total_images - train_split - val_split

    # 划分数据集
    train_images = images[:train_split]
    val_images = images[train_split:train_split + val_split]
    test_images = images[train_split + val_split:]

    # 将图片和对应的标签文件复制到相应的文件夹中
    for split_name, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
        split_folder = os.path.join(output_folder, split_name)
        images_folder = os.path.join(split_folder, "images")
        labels_folder = os.path.join(split_folder, "labels")

        for image_file in split_images:
            image_path = os.path.join(class_folder, image_file)
            label_file = os.path.join(class_folder, image_file[:-4] + ".txt")
            shutil.copy(image_path, images_folder)
            shutil.copy(label_file, labels_folder)

print("数据集划分完成。")





