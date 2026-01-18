# ORION Core — ML Pipeline 🧠

This directory contains the **complete machine learning pipeline** used to train, evaluate, and convert the YOLOv8 model for on-device Android deployment.

---

## 📁 Directory Structure

orion-core/
├── training/ # YOLOv8 training scripts & configs
├── inference/ # Python inference & evaluation
├── mobile/ # TFLite testing utilities
├── models/ # Trained weights & exported models
├── notebooks/ # Experiments, analysis & visualizations
├── tools/ # Helper scripts
├── save_clean_model.py
└── README.md


---

## 📦 Dataset

- Custom dataset with 150+ real-world images
- Annotated and versioned using **Roboflow**
- Classes:
  - Resistor
  - Capacitor
  - Transistor
  - IC
  - PCB

---

## 🏋️ Training

- Framework: **Ultralytics YOLOv8**
- Training environment: Google Colab
- Epochs: 40
- Input size: 640×640

Evaluation includes:
- mAP@50
- mAP@50–95
- Per-class precision & recall
- Confusion matrix analysis

---

## 🔁 Model Conversion Pipeline

1. PyTorch YOLOv8 (`.pt`)
2. TensorFlow SavedModel
3. TensorFlow Lite FP32 (baseline)
4. **TensorFlow Lite FP16 (final mobile model)**

FP16 was selected for:
- Better mobile performance
- Lower latency
- GPU compatibility

---

## 📱 Mobile Deployment Target

- Android (Kotlin)
- TensorFlow Lite Interpreter
- GPU Delegate (preferred)
- NNAPI fallback
- CPU fallback (last resort)

---

## 🧪 Testing & Validation

- Python inference sanity checks
- Output tensor validation
- Bounding box alignment checks
- On-device benchmarking via Android logs

---

## 🎯 Design Goals

- Accuracy–performance balance
- Mobile-first optimization
- Clean model export
- Real-device validation (not emulators)

---

## 📝 Notes

Large raw datasets and temporary files are intentionally excluded from this repository to keep it lightweight and reproducible.
