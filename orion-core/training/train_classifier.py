import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.orion_model import build_orion_model


# =====================
# CONFIG
# =====================
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 25


# =====================
# DATA GENERATORS
# =====================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.25,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

train_gen = datagen.flow_from_directory(
    "data/raw",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    "data/raw",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)


# =====================
# MODEL
# =====================
num_classes = train_gen.num_classes
model = build_orion_model(num_classes)


# =====================
# COMPILE
# =====================
# IMPORTANT:
# Model outputs = [class_output, embedding]
# Generator returns only ONE label (classification)
# So we use LOSS LIST (not dict)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=[
        "categorical_crossentropy",  # for class_output
        None                         # for embedding (no loss yet)
    ],
    metrics=[
        ["accuracy"],  # metrics for class_output
        []             # no metrics for embedding
    ]
)


# =====================
# TRAIN
# =====================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)


# =====================
# SAVE MODEL (NEW FORMAT)
# =====================
model.save("orion_model.keras")
print("✅ ORION model saved in .keras format")
