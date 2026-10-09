import os, zipfile

zip_name = "cats_and_dogs.zip"
with zipfile.ZipFile(zip_name,"r") as zip_ref:
    zip_ref.extractall(".")

train_dir = os.path.join("cats_and_dogs","train")
cat_path = os.path.join(train_dir,"cats")
dog_path = os.path.join(train_dir,"dogs")

val_dir = os.path.join("cats_and_dogs","validation")
val_cat_path = os.path.join(val_dir,"cats")
val_dog_path = os.path.join(val_dir,"dogs")

print("训练集猫：", len(os.listdir(cat_path)))
print("训练集狗：", len(os.listdir(dog_path)))
print("验证集猫：", len(os.listdir(val_cat_path)))
print("验证集狗：", len(os.listdir(val_dog_path)))

import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

base_dir = "cats_and_dogs"
train_cat_dir = os.path.join(base_dir, "train/cats")
train_dog_dir = os.path.join(base_dir, "train/dogs")

# 随机选3张猫、3张狗
cat_files = random.sample(os.listdir(train_cat_dir), 3)
dog_files = random.sample(os.listdir(train_dog_dir), 3)

plt.figure(figsize=(9,6))
for i, fname in enumerate(cat_files):
    img = load_img(os.path.join(train_cat_dir, fname))
    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title("cat")
    plt.axis("off")

for i, fname in enumerate(dog_files):
    img = load_img(os.path.join(train_dog_dir, fname))
    plt.subplot(2,3,i+4)
    plt.imshow(img)
    plt.title("dog")
    plt.axis("off")

plt.savefig("random_sample.png")
plt.show()
print("图片已保存 random_sample.png")
