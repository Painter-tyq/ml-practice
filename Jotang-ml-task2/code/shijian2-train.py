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

