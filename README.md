# ORION – Visual Intelligence 🚀

**Privacy-first, fully on-device real-time electronic component detection using YOLOv8 and TensorFlow Lite**

ORION – Visual Intelligence is an end-to-end mobile computer vision system that performs real-time electronic component detection entirely **on-device**, without relying on cloud services or network connectivity.

This project showcases a complete **production-style ML + Android pipeline**, covering dataset creation, model training, evaluation, TensorFlow Lite optimization, and hardware-accelerated inference on real Android devices.

---

## 🧠 Overview

ORION detects common electronic components directly from a live camera feed:

* Resistors
* Capacitors
* Transistors
* ICs (Integrated Circuits)
* PCBs (Printed Circuit Boards)

All inference is executed locally on the device, ensuring **low latency, offline functionality, and user privacy**.

---

## 🏗️ Project Structure

```
ORION-Visual-Intelligence/
├── android/        # Android application (CameraX + TensorFlow Lite)
├── orion-core/     # Model training, evaluation, and conversion pipeline
├── README.md       # Project documentation
└── .gitignore
```

---

## ⚙️ Technology Stack

### Machine Learning & Computer Vision

* YOLOv8 (Ultralytics)
* PyTorch
* TensorFlow Lite (FP32 → FP16 conversion)
* Google Colab

### Mobile & Edge Deployment

* Android (Kotlin)
* CameraX
* TensorFlow Lite Interpreter
* GPU Delegate (primary acceleration)
* NNAPI Delegate (fallback)

### Tooling & Utilities

* Roboflow (dataset annotation & versioning)
* OpenCV
* NumPy
* Matplotlib

---

## 📊 Model Training & Evaluation

* Custom dataset built from **150+ real-world images**
* Dataset annotated and versioned using Roboflow
* Model trained using Ultralytics YOLOv8 for **40 epochs**
* Evaluation metrics:

  * **mAP@50:** ~68%
  * **mAP@50–95:** ~58%
* Per-class precision, recall, and confusion matrix analysis performed during validation

---

## ⚡ On-Device Performance

**Tested on OnePlus Nord CE4 Lite**

* Inference latency: **~150–220 ms**
* Throughput: **~4–7 FPS**
* Hardware delegation: GPU → NNAPI → CPU fallback
* Model format: FP16 TensorFlow Lite

Performance metrics were measured using live CameraX input with real-time logging of inference latency and FPS.

---

## 📱 Android Application Features

* Live camera feed powered by CameraX
* Real-time bounding box rendering
* FPS and inference latency logging (Logcat)
* Automatic hardware delegate selection
* Fully offline inference with no network dependency

---

## 🧩 Key Engineering Challenges Addressed

* Resolved **TensorFlow Lite output tensor mismatch** by re-exporting the YOLOv8 FP16 model with consistent class configuration
* Implemented **class-aware Non-Max Suppression (NMS)** to eliminate duplicate detections
* Integrated GPU delegate with NNAPI fallback for robust hardware acceleration
* Validated model performance using **live real-world detection**, not just offline datasets

---

## 🔒 Privacy & Design Philosophy

* No internet connectivity required
* No image uploads or cloud inference
* Designed for edge deployment and privacy-preserving AI

---

## 📌 Highlights

* End-to-end ML + Android deployment pipeline
* Hardware-accelerated on-device inference
* Real-device benchmarking and validation
* Clean, modular, and production-style project structure

---

## 👤 Author

**Naveen Ganesh**

---

> This project demonstrates practical experience in deploying modern deep learning models to mobile devices, with a strong focus on performance optimization, reliability, and real-world usability.
