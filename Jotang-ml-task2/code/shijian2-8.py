import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

img_height = 150
img_width = 150

# 重建2-7基础模型
inputs = layers.Input(shape=(150, 150, 3))
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

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics=['acc']
)
model.load_weights("best_cats_dogs.weights.h5")

# ========= 方式：直接从验证集拿一张图片推理 =========
data_dir = "cats_and_dogs"
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir+"/validation",
  image_size=(img_height, img_width),
  batch_size=1
)
# 取出第一张图片
for img_batch, label_batch in val_ds.take(1):
    img_np = img_batch.numpy()[0].astype("uint8")
    true_label = label_batch.numpy()[0]

class_names = ['cats','dogs']
# 推理
pred = model.predict(img_batch, verbose=0)
if pred < 0.5:
    result = class_names[0]
    confidence = 1 - pred[0][0]
else:
    result = class_names[1]
    confidence = pred[0][0]

print(f"真实标签：{class_names[true_label]}")
print(f"预测类别：{result}")
print(f"置信度：{confidence:.3f}")

# 绘图
plt.figure()
plt.imshow(img_np)
plt.title(f"True:{class_names[true_label]}, Pred:{result}, Conf:{confidence:.3f}")
plt.axis("off")
plt.savefig("infer_result.png", dpi=300, bbox_inches="tight")
# plt.show()
print(" 预测图片已保存 infer_result.png")

