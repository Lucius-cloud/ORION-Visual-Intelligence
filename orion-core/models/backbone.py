import tensorflow as tf
from tensorflow.keras import layers, models

def build_backbone(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze backbone (IMPORTANT for small dataset)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    backbone = models.Model(inputs, x, name="orion_backbone")
    return backbone
