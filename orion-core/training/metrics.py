import tensorflow as tf

def get_metrics():
    return {
        "class_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    }
