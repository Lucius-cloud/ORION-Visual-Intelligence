from tensorflow.keras import layers

# -------------------------
# Classification Head
# -------------------------
def classification_head(features, num_classes):
    x = layers.Dense(128, activation="relu")(features)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_output"
    )(x)
    return output


# -------------------------
# Embedding Head (TFLite-safe)
# -------------------------
def embedding_head(features, embedding_dim=128):
    x = layers.Dense(embedding_dim)(features)
    x = layers.BatchNormalization()(x)
    output = layers.Activation(
        "linear",
        name="embedding"
    )(x)
    return output
