# 🍎🍌🥭 Fruit Ripeness Detection — YOLOv8n-OBB

An object detection project that automatically classifies the **ripeness stage** of apples, bananas, and mangoes from images and video, using a fine-tuned **YOLOv8 nano (Oriented Bounding Box)** model.

**Group 1:** Nawaf · Riyad · Mohammed · Norah · Rahaf

---

## 📌 Problem

Manually inspecting fruit ripeness is:
- Time-consuming
- Inconsistent between inspectors
- Dependent on subjective human judgment

**Goal:** Automatically classify fruit ripeness from images using computer vision, so the process is fast, consistent, and scalable.

---

## 🎯 Objectives

- Detect fruit ripeness automatically using computer vision.
- Train a YOLOv8 Oriented Object Detection model.
- Deploy the model for real-time predictions via a Gradio web app.

---

## 📂 Repository Contents

| File / Folder | Description |
|---|---|
| `Fruit_Ripeness_Training-G1.ipynb` | Main Colab notebook: dataset download, preprocessing, training (3 hyperparameter runs), evaluation, testing on new images, and Gradio deployment |
| `Fruit_Ripeness_v1i_yolov8-obb.zip` | Exported dataset in YOLOv8-OBB format (images + normalized OBB labels + `data.yaml`) |
| `data.yaml` | Dataset config file (class names and paths) used by Ultralytics for training |
| `README_dataset.txt` | Original dataset attribution from Roboflow Universe |
| `README_roboflow.txt` | Export details (preprocessing & augmentation applied by Roboflow) |
| `CV_Pres.pdf` | Final project presentation slides |

---

## 🗂️ Dataset

- **Source:** [Roboflow Universe – Fruit Ripeness](https://universe.roboflow.com/-plwff/fruit-ripeness-ffuvb-e1mir)
- **License:** CC BY 4.0
- **Format:** YOLOv8 Oriented Bounding Box (OBB)
- **Total images:** 1,915
  - Train: 1,694
  - Validation: 148
  - Test: 73

### Classes (9)
| # | Class |
|---|---|
| 0 | apple-overripe |
| 1 | apple-ripe |
| 2 | apple-unripe |
| 3 | banana-overripe |
| 4 | banana-ripe |
| 5 | banana-unripe |
| 6 | mango-overripe |
| 7 | mango-ripe |
| 8 | mango-unripe |

### Preprocessing (applied on Roboflow)
- Auto-orientation of pixel data (EXIF stripping)
- Resize to 512×512 (stretch)

### Augmentation (applied on Roboflow)
- Horizontal flip (50% probability)
- Brightness adjustment (±15%)
- Gaussian blur (0–2.5 px)

---

## 🧠 Model & Training

- **Architecture:** YOLOv8 nano, Oriented Bounding Box head (`yolov8n-obb.pt`)
- **Framework:** [Ultralytics](https://github.com/ultralytics/ultralytics)
- Three training configurations were compared to study the effect of epochs and image size:

| Run | Epochs | Image size | Batch |
|---|---|---|---|
| Run 1 – Baseline | 25 | 512 | 16 |
| Run 2 – More epochs | 50 | 512 | 16 |
| Run 3 – Bigger image size | 25 | 640 | 8 |

The best-performing run (highest mAP50-95) was selected as the final model.

### Final Results (best run)

| Metric | Score |
|---|---|
| Precision | 0.998 |
| Recall | 0.982 |
| mAP50 | 0.99 |
| mAP50-95 | 0.932 |

**Classes detected well:** apple-ripe, apple-unripe, mango-unripe, banana-ripe (precision & recall ≈ 1.0)

**Weaker points:**
- `banana-unripe` had the lowest recall (~0.84), likely confused with `banana-ripe` under certain lighting/angles.
- Testing on a real out-of-distribution image (a severely rotten apple) exposed a generalization gap: the model predicted `apple-ripe` with only 66% confidence instead of `apple-overripe`, suggesting the training data lacked diverse severity levels of decay.

**Possible improvements:**
- Collect more images for underrepresented classes (e.g. mango-ripe).
- Add training images covering more extreme/varied ripeness and lighting conditions.

---

## 🚀 How to Run

1. Open `Fruit_Ripeness_Training-G1.ipynb` in **Google Colab**.
2. Set the runtime to GPU: `Runtime > Change runtime type > T4 GPU`.
3. Run the cells top to bottom:
   - Mounts Google Drive (dataset & weights are cached there — re-running the notebook won't re-download or re-train from scratch).
   - Downloads the dataset from Roboflow (or use the included `.zip` and point `data.yaml` to it).
   - Trains the 3 model configurations (or loads existing weights if already trained).
   - Evaluates the best model (mAP, precision, recall) on the validation and test sets.
   - Tests the model on new, real-world images uploaded by the user.
   - Launches a **Gradio** web app for interactive image/video testing.

### Using the included dataset export directly
```python
from ultralytics import YOLO

model = YOLO("yolov8n-obb.pt")
model.train(data="data.yaml", epochs=50, imgsz=512, batch=16)
```
> Unzip `Fruit_Ripeness_v1i_yolov8-obb.zip` first and make sure `data.yaml`'s `path` points to the extracted folder.

---

## 🖥️ Deployment

A **Gradio** web interface is included in the notebook, with two tabs:
- **Image tab:** upload an image → get annotated result + confidence table.
- **Video tab (bonus):** upload a video → get frame-by-frame annotated output.

---

## 📊 Presentation

See `CV_Pres.pdf` for the full project walkthrough: problem statement, dataset, model performance, and error analysis.

---

## 🙏 Acknowledgements

Dataset provided by a Roboflow user via [Roboflow Universe](https://universe.roboflow.com), licensed under CC BY 4.0. Exported and processed using [Roboflow](https://roboflow.com). Model built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
