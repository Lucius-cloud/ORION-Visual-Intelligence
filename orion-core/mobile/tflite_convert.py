import tensorflow as tf
import numpy as np
import os
import glob
from tensorflow.keras.preprocessing import image

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = r"C:\Users\navst\ORION-Visual-Intelligence\orion-core\models\orion_model.keras"
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
TFLITE_DIR = os.path.join(BASE_DIR, "mobile", "tflite")

IMG_SIZE = (224, 224)
os.makedirs(TFLITE_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL (NO custom_objects!)
# ============================================================
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

print("✅ ORION model loaded successfully")

# ============================================================
# FP16 CONVERSION
# ============================================================
fp16_converter = tf.lite.TFLiteConverter.from_keras_model(model)
fp16_converter.optimizations = [tf.lite.Optimize.DEFAULT]
fp16_converter.target_spec.supported_types = [tf.float16]

fp16_model = fp16_converter.convert()

with open(os.path.join(TFLITE_DIR, "orion_fp16.tflite"), "wb") as f:
    f.write(fp16_model)

print("✅ FP16 TFLite model generated")

# ============================================================
# INT8 CONVERSION (REAL DATA CALIBRATION)
# ============================================================
def representative_dataset():
    image_paths = glob.glob(os.path.join(DATA_DIR, "*", "*.jpg"))[:100]
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        yield [img]

int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
int8_converter.representative_dataset = representative_dataset
int8_converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
int8_converter.inference_input_type = tf.uint8
int8_converter.inference_output_type = tf.uint8

int8_model = int8_converter.convert()

with open(os.path.join(TFLITE_DIR, "orion_int8.tflite"), "wb") as f:
    f.write(int8_model)

print("✅ INT8 TFLite model generated")

print("\n🎉 STAGE 5 COMPLETE — MOBILE MODELS READY")
