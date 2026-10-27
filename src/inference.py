from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
YOLO_NAMES = ["Glioma", "Meningioma", "Pituitary"]
IMG_SIZE = 224


def load_classifier(path=ROOT / "models" / "mobilenet_finetuned.keras"):
    return tf.keras.models.load_model(path)


def load_detector(path=ROOT / "models" / "yolov8n_mri_best.pt"):
    return YOLO(str(path))


def predict_class(model, image):
    """image: PIL image. Returns the 4 probabilities."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    arr = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE)).numpy()   # same resize as training
    x = preprocess_input(arr)
    return model.predict(x[None], verbose=0)[0]


def detect(model, image, conf=0.25):
    """Returns a list of boxes: class, confidence, xyxy in pixels."""
    result = model.predict(image.convert("RGB"), conf=conf, verbose=False)[0]
    return [
        {"class": YOLO_NAMES[int(b.cls)], "conf": float(b.conf),
         "xyxy": [float(v) for v in b.xyxy[0]]}
        for b in result.boxes
    ]