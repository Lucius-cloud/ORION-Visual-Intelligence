import numpy as np
import json
import os

SRC = "data/embeddings"
DST = "../android_embeddings"

os.makedirs(DST, exist_ok=True)

for file in os.listdir(SRC):
    if not file.endswith(".npy"):
        continue

    print(f"Processing {file}...")

    data = np.load(
        os.path.join(SRC, file),
        allow_pickle=True
    ).item()

    out = {}
    for img_name, embedding in data.items():
        out[img_name] = embedding.tolist()

    with open(
        os.path.join(DST, file.replace(".npy", ".json")),
        "w"
    ) as f:
        json.dump(out, f)

print("\n✅ Embeddings converted to JSON successfully")
