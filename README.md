# ORION-Visual-Intelligence 🚀

**Privacy-first, fully on-device visual intelligence system** for real-time electronics component detection using YOLOv8 and TensorFlow Lite.

This project demonstrates an **end-to-end mobile AI pipeline** — from dataset creation and model training to optimized on-device inference on Android using GPU and NNAPI acceleration.

---

## 🧠 What It Does

Detects common electronic components in real time:
- Resistors
- Capacitors
- Transistors
- ICs
- PCBs

All inference runs **entirely on-device** (no cloud, no server).

---

## 🏗️ Project Structure

ORION-Visual-Intelligence/
├── android/ # Android app (CameraX + TFLite)
├── orion-core/ # ML training, inference, conversion pipeline
├── README.md # Project overview (this file)
└── .gitignore


---

## ⚙️ Tech Stack

**ML / CV**
- YOLOv8 (Ultralytics)
- PyTorch
- TensorFlow Lite (FP16)
- Google Colab

**Mobile**
- Android (Kotlin)
- CameraX
- TensorFlow Lite Interpreter
- GPU Delegate (primary)
- NNAPI Delegate (fallback)

**Tools**
- Roboflow (dataset versioning)
- OpenCV
- NumPy
- Matplotlib

---

## 📊 Model Performance

- **mAP@50:** ~68%
- **mAP@50–95:** ~58%
- Per-class precision, recall & confusion matrix analyzed
- Trained for 40 epochs on a custom dataset

---

## ⚡ On-Device Performance

Tested on **OnePlus Nord CE4 Lite**:
- **Latency:** ~150–220 ms
- **FPS:** ~4–7 FPS
- **Delegation:** GPU → NNAPI → CPU fallback
- **Model:** FP16 TensorFlow Lite

---

## 📱 Android App Features

- Live camera feed using CameraX
- Real-time bounding box rendering
- FPS & inference latency logging
- Automatic hardware delegate selection
- Fully offline inference

---

## 🔒 Privacy & Design Philosophy

- No internet required
- No image uploads
- No cloud inference
- Designed for edge deployment

---

## 📌 Highlights

✔ End-to-end ML + Mobile pipeline  
✔ Hardware-accelerated inference  
✔ Real-device benchmarking  
✔ Clean, production-style repo structure  

---

## 👤 Author

Built by **Naveen Ganesh**  
