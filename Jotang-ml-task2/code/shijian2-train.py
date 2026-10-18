import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 搭建CNN模型
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

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics=['acc']
)

# 数据加载（带数据增强）
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'cats_and_dogs/train',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
validation_generator = val_datagen.flow_from_directory(
    'cats_and_dogs/validation',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
test_generator = val_datagen.flow_from_directory(
    'cats_and_dogs/test',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
# 保存最优模型权重
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_cats_dogs.weights.h5",
        save_best_only=True,
        save_weights_only=True
    )
]

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=callbacks
)

print(" 训练结束，最优权重已保存 best_cats_dogs.weights.h5")

import matplotlib.pyplot as plt

# 1. 设置画布，画两张图（左边 Loss，右边 Accuracy）
plt.figure(figsize=(12, 4))

# 2. 画 Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='red')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 3. 画 Accuracy 曲线
plt.subplot(1, 2, 2)
acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'

plt.plot(history.history[acc_key], label='Train Acc', color='blue')
plt.plot(history.history[val_acc_key], label='Val Acc', color='red')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 4. 保存并显示
plt.tight_layout()
plt.savefig("training_curves.png")
print("曲线图已保存为 training_curves.png")
