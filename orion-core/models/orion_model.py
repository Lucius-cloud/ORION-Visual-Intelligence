import tensorflow as tf
from tensorflow.keras import models
from .backbone import build_backbone
from .heads import classification_head, embedding_head
def build_orion_model(num_classes):
    backbone = build_backbone()

    features = backbone.output

    class_out = classification_head(features, num_classes)
    embed_out = embedding_head(features)

    model = models.Model(
        inputs=backbone.input,
        outputs=[class_out, embed_out],
        name="ORION"
    )
    return model
