import tensorflow as tf

def get_loss():
    return {
        "class_output": tf.keras.losses.SparseCategoricalCrossentropy(),
        "embedding": None
    }
