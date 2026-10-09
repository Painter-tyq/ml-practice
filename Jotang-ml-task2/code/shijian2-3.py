import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# CNN模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ========== 修复点：先build模型，定义输入 ==========
model.build(input_shape=(None,150,150,3))

# ========== 第9题 模型结构图 ==========
from tensorflow.keras.utils import plot_model
plot_model(
    model,
    to_file="model_structure.png",
    show_shapes=True,
    show_layer_names=True,
    dpi=300
)
print(" 模型结构图已保存 model_structure.png")

# 提取前4层输出
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# 读取图片，预处理
img_path = "cats_and_dogs/train/cats/cat.0.jpg"
img = load_img(img_path, target_size=(150,150))
img_arr = img_to_array(img)/255.0
img_arr = tf.expand_dims(img_arr, axis=0)

activations = activation_model.predict(img_arr, verbose=0)

# 画出第一层卷积的4张特征图
plt.figure(figsize=(12,4))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(activations[0][0,:,:,i], cmap="gray")
    plt.axis("off")
plt.savefig("feature_map.png")
print(" 特征图保存 feature_map.png")
