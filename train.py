"""
Crop Disease Detection — Training Script
Team: Sun & Nysa
Model: EfficientNetB0 with Transfer Learning
Dataset: PlantVillage (54,306 images, 38 classes)

Run locally in VS Code. See SETUP.md for environment setup instructions.
"""

import os, json, shutil, pathlib, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ── 1. !! SET THIS TO YOUR DATASET FOLDER !! ────────────────────────────────
#
# Point DATA_DIR to whichever folder contains the 38 class subfolders.
# Common layouts after unzipping the Kaggle download:
#
#   Layout A (most common):
#     plantdisease/
#       PlantVillage/
#         Apple___Apple_scab/
#         Apple___Black_rot/
#         ...
#   → set DATA_DIR = "plantdisease/PlantVillage"
#
#   Layout B (already flat):
#     plantdisease/
#       Apple___Apple_scab/
#       Apple___Black_rot/
#       ...
#   → set DATA_DIR = "plantdisease"
#
# Use a raw string (r"...") on Windows to avoid backslash issues.
# Examples:
#   Mac/Linux: DATA_DIR = "/Users/sun/Downloads/PlantVillage"
#   Relative : DATA_DIR = "data/PlantVillage"   ← if dataset is inside project folder

DATA_DIR = "/Users/ngl/Documents/PlantVillage"   # <-- CHANGE THIS

# ── 2. Constants ────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_HEAD = 10
EPOCHS_FINE = 15
LR_HEAD     = 1e-3
LR_FINE     = 1e-4
SEED        = 42
OUTPUT_DIR  = pathlib.Path("outputs")   # all saved files go here
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── 3. GPU check ─────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}")
    # Allow memory growth so TF doesn't grab the entire VRAM at once
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("⚠️  No GPU detected — training on CPU. This will be slow (~10× slower).")
    print("    Consider running on Google Colab (free T4 GPU) if training is too slow.")

# ── 4. Explore & understand class distribution ───────────────────────────────
def explore_dataset(data_dir):
    raw_path = pathlib.Path(data_dir)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset folder not found: {raw_path.resolve()}\n"
            f"   Open train.py and update the DATA_DIR variable at the top."
        )

    classes = sorted([d.name for d in raw_path.iterdir() if d.is_dir()])
    if len(classes) == 0:
        raise ValueError(f"No subfolders found in {raw_path}. Check your DATA_DIR path.")

    class_counts = {}
    for cls in classes:
        n = (len(list((raw_path / cls).glob("*.jpg")))
           + len(list((raw_path / cls).glob("*.JPG")))
           + len(list((raw_path / cls).glob("*.jpeg")))
           + len(list((raw_path / cls).glob("*.png"))))
        class_counts[cls] = n

    total = sum(class_counts.values())
    print(f"\n{'='*60}")
    print(f"Dataset path  : {raw_path.resolve()}")
    print(f"Total images  : {total:,}")
    print(f"Total classes : {len(classes)}")
    print(f"{'='*60}")
    print(f"{'Class':<45} {'Count':>7} {'%':>6}")
    print("-"*60)
    for cls, n in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"{cls:<45} {n:>7,} {100*n/total:>5.1f}%")

    counts = np.array(list(class_counts.values()))
    ratio  = counts.max() / (counts.min() or 1)
    print(f"\nMax/Min ratio: {ratio:.1f}x  — {'imbalanced' if ratio > 3 else 'roughly balanced'}")
    return classes, class_counts, raw_path

classes, class_counts, data_root = explore_dataset(DATA_DIR)
NUM_CLASSES = len(classes)


# ── 5. Split into train / val / test ─────────────────────────────────────────
def build_split_dirs(data_root, dest="split_data", train_frac=0.70, val_frac=0.15):
    dest = pathlib.Path(dest)
    if dest.exists():
        print(f"\nSplit folder already exists at '{dest}' — skipping copy.")
        print("   Delete 'split_data/' and re-run if you want a fresh split.")
        return dest

    print(f"\nCreating train/val/test split in '{dest}/' ...")
    for cls in classes:
        images = (list((data_root / cls).glob("*.jpg"))
                + list((data_root / cls).glob("*.JPG"))
                + list((data_root / cls).glob("*.jpeg"))
                + list((data_root / cls).glob("*.png")))
        random.shuffle(images)
        n       = len(images)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        splits  = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }
        for split, imgs in splits.items():
            target = dest / split / cls
            target.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, target / img.name)
    print("Split complete ✓")
    return dest

split_dir = build_split_dirs(data_root)


# ── 6. Data generators ───────────────────────────────────────────────────────
#
# Preprocessing applied to EVERY image (train + val + test):
#   • Resize to 224×224 pixels
#   • Normalize pixel values from [0, 255] → [0.0, 1.0]
#
# Additional augmentation on TRAINING images only:
#   • Random horizontal flip   — disease patterns are not directional
#   • Rotation ±15°            — simulates tilted leaf photos
#   • Zoom 0–20%               — simulates different camera distances
#   • Brightness ±10%          — simulates different lighting
#
train_datagen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    horizontal_flip  = True,
    rotation_range   = 15,
    zoom_range       = 0.20,
    brightness_range = [0.9, 1.1],
    fill_mode        = "nearest",
)
eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

def make_gen(datagen, split, shuffle=True):
    return datagen.flow_from_directory(
        split_dir / split,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = "categorical",
        shuffle     = shuffle,
        seed        = SEED,
    )

train_gen = make_gen(train_datagen, "train", shuffle=True)
val_gen   = make_gen(eval_datagen,  "val",   shuffle=False)
test_gen  = make_gen(eval_datagen,  "test",  shuffle=False)

# Save class index map — needed by app.py
class_index_path = OUTPUT_DIR / "class_indices.json"
with open(class_index_path, "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print(f"Saved {class_index_path}")


# ── 7. Class-weight computation ───────────────────────────────────────────────
y_train = train_gen.classes
cw_values = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(cw_values))


# ── 8. Build model ────────────────────────────────────────────────────────────
def build_model():
    base = EfficientNetB0(
        include_top = False,
        weights     = "imagenet",
        input_shape = (IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False  # freeze for Phase 1

    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs, outputs), base

model, base_model = build_model()
model.summary()
print(f"\nTotal parameters      : {model.count_params():,}")
print(f"Trainable (Phase 1)   : {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")


# ── 9. Phase 1 — Head only ────────────────────────────────────────────────────
model.compile(
    optimizer = keras.optimizers.Adam(LR_HEAD),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy",
                 keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
)

cb_p1 = [
    keras.callbacks.ModelCheckpoint(
        str(OUTPUT_DIR / "best_phase1.keras"),
        save_best_only=True, monitor="val_accuracy"),
    keras.callbacks.EarlyStopping(
        patience=4, restore_best_weights=True, monitor="val_accuracy"),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=2, monitor="val_loss"),
    keras.callbacks.CSVLogger(str(OUTPUT_DIR / "history_phase1.csv")),
]

print("\n─── Phase 1: Training classification head ───")
history1 = model.fit(
    train_gen,
    epochs          = EPOCHS_HEAD,
    validation_data = val_gen,
    class_weight    = class_weights,
    callbacks       = cb_p1,
)


# ── 10. Phase 2 — Fine-tune top 30 base layers ───────────────────────────────
for layer in base_model.layers[-30:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

print(f"Trainable (Phase 2)   : {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")

model.compile(
    optimizer = keras.optimizers.Adam(LR_FINE),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy",
                 keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
)

cb_p2 = [
    keras.callbacks.ModelCheckpoint(
        str(OUTPUT_DIR / "best_model.keras"),
        save_best_only=True, monitor="val_accuracy"),
    keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, monitor="val_accuracy"),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.3, patience=3, monitor="val_loss"),
    keras.callbacks.CSVLogger(str(OUTPUT_DIR / "history_phase2.csv")),
]

print("\n─── Phase 2: Fine-tuning top 30 EfficientNet layers ───")
history2 = model.fit(
    train_gen,
    epochs          = EPOCHS_HEAD + EPOCHS_FINE,
    validation_data = val_gen,
    class_weight    = class_weights,
    callbacks       = cb_p2,
    initial_epoch   = len(history1.history["loss"]),
)


# ── 11. Evaluate on test set ─────────────────────────────────────────────────
print("\n─── Test Set Evaluation ───")
model.load_weights(str(OUTPUT_DIR / "best_model.keras"))

loss, acc, top3 = model.evaluate(test_gen, verbose=1)
print(f"\nTest Loss      : {loss:.4f}")
print(f"Test Accuracy  : {acc*100:.2f}%")
print(f"Top-3 Accuracy : {top3*100:.2f}%")

test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred    = np.argmax(y_pred_probs, axis=1)
y_true    = test_gen.classes
idx2class = {v: k for k, v in train_gen.class_indices.items()}
labels    = [idx2class[i] for i in range(NUM_CLASSES)]

report_str = classification_report(y_true, y_pred, target_names=labels, digits=3)
print("\nPer-class Report:")
print(report_str)

with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(f"Test Loss      : {loss:.4f}\n")
    f.write(f"Test Accuracy  : {acc*100:.2f}%\n")
    f.write(f"Top-3 Accuracy : {top3*100:.2f}%\n\n")
    f.write(report_str)
print(f"Saved {OUTPUT_DIR / 'classification_report.txt'}")


# ── 12. Training curves ───────────────────────────────────────────────────────
def merge(h1, h2):
    return {k: h1.history[k] + h2.history[k] for k in h1.history}

hist  = merge(history1, history2)
epochs_range = range(1, len(hist["accuracy"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs_range, hist["accuracy"],     label="Train Acc")
axes[0].plot(epochs_range, hist["val_accuracy"], label="Val Acc")
axes[0].axvline(EPOCHS_HEAD, color="gray", ls="--", alpha=0.5, label="Fine-tune start")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")

axes[1].plot(epochs_range, hist["loss"],     label="Train Loss")
axes[1].plot(epochs_range, hist["val_loss"], label="Val Loss")
axes[1].axvline(EPOCHS_HEAD, color="gray", ls="--", alpha=0.5, label="Fine-tune start")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].set_xlabel("Epoch")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.show()
print(f"Saved {OUTPUT_DIR / 'training_curves.png'}")


# ── 13. Confusion matrix (top 15 classes) ────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
top15_idx = sorted(range(NUM_CLASSES),
                   key=lambda i: class_counts.get(labels[i], 0), reverse=True)[:15]
cm_top    = cm[np.ix_(top15_idx, top15_idx)]
top15_lbl = [labels[i].replace("___", "\n") for i in top15_idx]

plt.figure(figsize=(14, 12))
sns.heatmap(cm_top, annot=True, fmt="d", cmap="Blues",
            xticklabels=top15_lbl, yticklabels=top15_lbl)
plt.title("Confusion Matrix (Top 15 Classes)")
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.show()
print(f"Saved {OUTPUT_DIR / 'confusion_matrix.png'}")


# ── 14. Save final model ──────────────────────────────────────────────────────
model.save(OUTPUT_DIR / "crop_disease_model")
print(f"\n✅ Training complete! Model saved to {OUTPUT_DIR / 'crop_disease_model'}")
print("\nNext step: copy these to your Hugging Face Space:")
print(f"  {OUTPUT_DIR / 'crop_disease_model'}/")
print(f"  {OUTPUT_DIR / 'class_indices.json'}")
