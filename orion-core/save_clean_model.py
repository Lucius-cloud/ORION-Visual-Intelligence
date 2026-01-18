import os
from models.orion_model import build_orion_model
import tensorflow as tf

# Where to save the clean model
OUT_PATH = "orion_model.keras"

# Number of classes in your training dataset (update if needed)
NUM_CLASSES = 5   # <-- CHANGE THIS

# Build Orion model
model = build_orion_model(NUM_CLASSES)

# Save clean model in Keras v3 format (zip archive)
model.save(OUT_PATH)

print("\n✅ Clean ORION model saved as:", OUT_PATH)
