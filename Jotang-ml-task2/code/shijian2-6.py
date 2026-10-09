import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# 构造数据增强模块
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),   # 随机水平翻转
    layers.RandomCrop(height=130, width=130), # 随机裁剪
    layers.Resizing(150,150) # 裁剪后缩回到原图尺寸
])

# 读取图片
img_path = "cats_and_dogs/train/cats/cat.0.jpg"
img = load_img(img_path, target_size=(150,150))
img_arr = img_to_array(img)
img_arr = tf.expand_dims(img_arr, axis=0)

plt.figure(figsize=(14,4))
# 原图
plt.subplot(1,4,1)
plt.imshow(img_arr[0]/255.)
plt.title("原图，无增强")
plt.axis("off")

# 生成3组不同随机增强图
for i in range(3):
    aug_img = data_augmentation(img_arr, training=True) # training=True强制启用随机
    plt.subplot(1,4,i+2)
    plt.imshow(aug_img[0]/255.)
    plt.title(f"增强样本{i+1}")
    plt.axis("off")

plt.savefig("aug_compare.png")
print(" 一次性生成原图+3个不同增强样本，保存 aug_compare.png")
