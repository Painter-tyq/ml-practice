import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 函数式API搭建网络
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

# 跳过Input层，取【卷积、池化】的输出，从下标1开始取前3层
layer_outputs = [layer.output for layer in model.layers[1:4]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# 读取图片
img_path = "cats_and_dogs/train/cats/cat.0.jpg"
img = load_img(img_path, target_size=(150,150))
img_arr = img_to_array(img)/255.0
img_arr = tf.expand_dims(img_arr, axis=0)

activations = activation_model.predict(img_arr, verbose=0)

# 只画3张图，对应第一层卷积的前3个通道
plt.figure(figsize=(12,4))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(activations[0][0,:,:,i], cmap="gray")
    plt.axis("off")
plt.savefig("feature_map.png")
print(" 特征图 feature_map.png 保存成功！")
