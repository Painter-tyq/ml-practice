import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

data_dir = "cats_and_dogs"
img_height = 150
img_width = 150
batch_size = 32

# 和训练文件一模一样的网络结构
inputs = layers.Input(shape=(150,150,3))
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=inputs, outputs=outputs)

# 加载权重
model.load_weights("best_cats_dogs.weights.h5")
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics=['acc']
)

# 加载数据集
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir+"/train",
  image_size=(img_height, img_width),
  batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir+"/validation",
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names
print("类别：", class_names)

# ===================== 2-7 查找并可视化错分样本（修复下标bug） =====================
error_imgs = []
error_true_label = []
error_pred_label = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    pred_classes = (preds > 0.5).astype(int)
    
    for img, true_lb, pred_lb in zip(images, labels, pred_classes):
        # 加 .item() 取出单个数字
        true_val = true_lb.numpy().item()
        pred_val = pred_lb.item()
        if true_val != pred_val:
            error_imgs.append(img.numpy().astype("uint8"))
            error_true_label.append(true_val)
            error_pred_label.append(pred_val)

plt.figure(figsize=(12, 8))
for i in range(min(6, len(error_imgs))):
    plt.subplot(2,3,i+1)
    plt.imshow(error_imgs[i])
    plt.title(f"True:{class_names[error_true_label[i]]}, Pred:{class_names[error_pred_label[i]]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

print(f"找到预测错误样本总数：{len(error_imgs)}")
