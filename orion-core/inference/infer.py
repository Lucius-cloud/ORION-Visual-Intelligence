import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = "models/orion_model.keras"   
DATA_DIR = "data/raw"
EMBED_DIR = "data/embeddings"
IMG_SIZE = (224, 224)

if not os.path.isdir(EMBED_DIR):
    os.makedirs(EMBED_DIR)

print("📦 Loading ORION model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def build_embedding_db():
    print("🔨 Building embedding database...")

    for cls in os.listdir(DATA_DIR):
        cls_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_path):
            continue

        embeddings = {}

        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)

            img = preprocess_image(img_path)
            _, emb = model.predict(img, verbose=0)
            embeddings[img_name] = emb[0]

        save_path = os.path.join(EMBED_DIR, f"{cls}.npy")
        np.save(save_path, embeddings)
        print(f"💾 Saved embeddings for {cls}")

    print("✅ Embedding DB built successfully")

def run_single_inference(image_path):
    img = preprocess_image(image_path)
    class_probs, embedding = model.predict(img, verbose=0)

    class_probs = class_probs[0]
    embedding = embedding[0]

    predicted_class = np.argmax(class_probs)
    confidence = class_probs[predicted_class]

    print("✅ Inference Successful")
    print(f"Predicted class index: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Embedding shape: {embedding.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image for inference")
    parser.add_argument("--build-db", action="store_true", help="Build embedding database")

    args = parser.parse_args()

    if args.build_db:
        build_embedding_db()
    elif args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        run_single_inference(args.image)
    else:
        parser.error("Please provide --image or --build-db")
