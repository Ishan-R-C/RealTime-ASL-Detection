import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D
import matplotlib.pyplot as plt

#initialize parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "processed"

#augments the images to create more diversity in the database
datagen = ImageDataGenerator(
    validation_split=0.2, #splits 20% data for validation
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.8, 1.2]
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    color_mode = 'grayscale',
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    subset = 'training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    color_mode = 'grayscale',
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    subset = 'validation'
)

input_tensor = Input(shape = (IMG_SIZE, IMG_SIZE, 1)) #accepts grayscale image
x = Conv2D(3, (3,3), padding = 'same')(input_tensor) #converts into 3 channels for MobileNetV2

base_model = MobileNetV2(
    input_shape = (IMG_SIZE, IMG_SIZE, 3),
    include_top = False,
    weights = 'imagenet'
)
base_model.trainable = False

x = base_model(x)
x = GlobalAveragePooling2D()(x)
output = Dense(train_gen.num_classes, activation = 'softmax')(x)

model = Model(inputs = input_tensor, outputs = output)
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#trains for given epochs
history = model.fit(
    train_gen,
    epochs = EPOCHS,
    validation_data = val_gen
)

model.save("asl_model.h5")
print("Model Saved")

#visualizes training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label = 'Train Acc')
plt.plot(history.history['val_accuracy'], label = 'Val Acc')
plt.title("Training Accuracy")
plt.legend()
plt.show()