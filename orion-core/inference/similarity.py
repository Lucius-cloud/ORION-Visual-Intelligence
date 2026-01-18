import os
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/orion_model.keras"
EMBED_DIR = "data/embeddings"
DATA_DIR = "data/raw"
IMG_SIZE = (224, 224)
TOP_K = 5

print("📦 Loading ORION model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def run_similarity(image_path):
    img = preprocess_image(image_path)
    _, query_emb = model.predict(img, verbose=0)
    query_emb = query_emb[0]

    scores = []

    for cls_file in os.listdir(EMBED_DIR):
        cls_path = os.path.join(EMBED_DIR, cls_file)
        class_name = cls_file.replace(".npy", "")
        embeddings = np.load(cls_path, allow_pickle=True).item()

        for img_name, emb in embeddings.items():
            score = cosine_similarity(query_emb, emb)
            scores.append((class_name, img_name, score))

    scores.sort(key=lambda x: x[2], reverse=True)

    print("\n🔍 Top Similar Images:")
    for cls, img, score in scores[:TOP_K]:
        print(f"{cls}/{img}  →  similarity: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Query image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError("Query image not found")

    run_similarity(args.image)
