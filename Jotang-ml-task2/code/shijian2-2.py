import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

base_dir = "cats_and_dogs"
train_dir = os.path.join(base_dir, "train")

# 数据生成器：归一化像素到0~1
train_datagen = ImageDataGenerator(rescale=1./255)

# 构造batch，batch_size=20，每次读取20张图片作为一个批次
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),   # 统一缩放到150×150
    batch_size=20,
    class_mode="binary"
)

# 随机取一张原图，对比预处理前后
img_path = os.path.join(train_dir, "cats", "cat.0.jpg")
# 原始图片
img_original = load_img(img_path)
arr_original = img_to_array(img_original) / 255.0

# 预处理之后的图片（resize到150,150）
img_processed = load_img(img_path, target_size=(150,150))
arr_processed = img_to_array(img_processed) / 255.0

# 绘图保存
plt.figure(figsize=(8, 4))
plt.subplot(1,2,1)
plt.title("原图")
plt.imshow(arr_original)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("预处理后 (150×150)")
plt.imshow(arr_processed)
plt.axis("off")

plt.savefig("preprocess_compare.png")
print("对比图片保存完成 preprocess_compare.png")

# 读取一个batch的数据，打印batch形状
batch_imgs, batch_labels = next(train_generator)
print(f"一个batch图片张量shape：{batch_imgs.shape}")
print(f"一个batch标签shape：{batch_labels.shape}")
