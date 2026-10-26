import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
BATCH_SIZE = 32
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
CSV_PATH = "../data/inventory_final.csv"

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(10 / 360, fill_mode="constant", fill_value=-1.0),
    layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=-1.0),
    layers.RandomZoom(0.1, fill_mode="constant", fill_value=-1.0),
    layers.RandomContrast(0.1, value_range=(-1, 1)),
], name="data_augmentation")


def parse(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    return img, label


def make_dataset(split, augment=False, csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    d = df[df["split"] == split]
    paths = d["image_path"].values
    labels = d["folder_class"].map(CLASS_TO_ID).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if augment:
        ds = ds.shuffle(len(d), seed=42)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)