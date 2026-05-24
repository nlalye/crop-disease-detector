import json, pathlib
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

# ── Load model ───────────────────────────────────────────────────────────────
# Works whether you run locally or on Hugging Face Spaces
BASE = pathlib.Path(__file__).parent

MODEL_PATH = BASE / "crop_disease_model"
if not MODEL_PATH.exists():
    # fallback: look in outputs/ (if running from project root after training)
    MODEL_PATH = BASE / "outputs" / "crop_disease_model"

CLASS_INDEX_PATH = BASE / "class_indices.json"
if not CLASS_INDEX_PATH.exists():
    CLASS_INDEX_PATH = BASE / "outputs" / "class_indices.json"

print(f"Loading model from {MODEL_PATH} ...")
model = tf.saved_model.load(str(MODEL_PATH))
infer = model.signatures["serving_default"]
print("Model loaded ✓")

with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

# ── Disease info ──────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "healthy": {
        "description": "This leaf appears healthy — no signs of disease detected.",
        "treatment": "Continue regular watering, fertilization, and pest monitoring.",
        "severity": "None",
    },
    "Bacterial_spot": {
        "description": "Bacterial spot causes water-soaked lesions that turn brown with yellow halos.",
        "treatment": "Use copper-based bactericides. Avoid overhead irrigation. Remove infected leaves.",
        "severity": "Moderate",
    },
    "Early_blight": {
        "description": "Early blight (Alternaria) shows dark concentric rings forming a bullseye pattern.",
        "treatment": "Apply fungicide (chlorothalonil or mancozeb). Improve air circulation.",
        "severity": "Moderate",
    },
    "Late_blight": {
        "description": "Late blight (Phytophthora) causes dark water-soaked lesions with white mold on the underside.",
        "treatment": "Apply fungicide immediately. Remove infected plants. Spreads very fast.",
        "severity": "Severe",
    },
    "Leaf_scorch": {
        "description": "Leaf scorch appears as brown edges and tips from water stress or fungal infection.",
        "treatment": "Improve irrigation. Apply appropriate fungicide if fungal origin confirmed.",
        "severity": "Mild–Moderate",
    },
    "Powdery_mildew": {
        "description": "Powdery mildew coats leaves in white powder, stunting growth.",
        "treatment": "Apply sulfur-based fungicides or neem oil. Improve air circulation.",
        "severity": "Moderate",
    },
    "Black_rot": {
        "description": "Black rot creates V-shaped yellow lesions along leaf margins, eventually turning brown.",
        "treatment": "Remove infected leaves. Apply copper-based fungicide. Avoid wetting foliage.",
        "severity": "Moderate–Severe",
    },
    "Common_rust": {
        "description": "Common rust shows small, circular, powdery orange-brown pustules on both leaf surfaces.",
        "treatment": "Apply fungicide early. Plant resistant varieties when possible.",
        "severity": "Moderate",
    },
    "Northern_Leaf_Blight": {
        "description": "Northern Leaf Blight causes long, cigar-shaped gray-green lesions on corn leaves.",
        "treatment": "Apply fungicide at tasseling. Use resistant hybrids.",
        "severity": "Moderate–Severe",
    },
}

def get_disease_info(class_name: str) -> dict:
    name_lower = class_name.lower()
    if "healthy" in name_lower:
        return DISEASE_INFO["healthy"]
    for key, val in DISEASE_INFO.items():
        if key.lower().replace("_", " ") in name_lower.replace("_", " "):
            return val
    return {
        "description": "No additional information available for this condition.",
        "treatment": "Consult an agricultural extension service for advice.",
        "severity": "Unknown",
    }

def format_class_name(raw: str) -> str:
    """'Tomato___Late_blight' → 'Tomato — Late Blight'"""
    return raw.replace("___", " — ").replace("_", " ").title()

# ── Preprocessing ──────────────────────────────────────────────────────────────
IMG_SIZE = 224

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(image: Image.Image):
    if image is None:
        return None, "⬅️ Please upload a leaf photo to get started."

    x     = preprocess(image)
    input_tensor = tf.constant(x, dtype=tf.float32)
    output = infer(input_tensor)
    preds = list(output.values())[0].numpy()[0] 

    top3_idx    = preds.argsort()[-3:][::-1]
    top3_labels = [format_class_name(idx_to_class[i]) for i in top3_idx]
    top3_confs  = [float(preds[i]) for i in top3_idx]
    label_dict  = {label: conf for label, conf in zip(top3_labels, top3_confs)}

    info = get_disease_info(idx_to_class[top3_idx[0]])
    severity_emoji = {
        "None": "✅", "Mild–Moderate": "🟡", "Moderate": "🟠",
        "Severe": "🔴", "Unknown": "❓"
    }
    emoji = severity_emoji.get(info["severity"], "❓")

    info_md = (
        f"## {emoji} {top3_labels[0]}\n\n"
        f"**Confidence:** {top3_confs[0]*100:.1f}%  \n"
        f"**Severity:** {info['severity']}\n\n"
        f"**What it is:** {info['description']}\n\n"
        f"**Recommended action:** {info['treatment']}\n\n"
        f"---\n"
        f"*⚠️ This tool is for educational purposes only. "
        f"Always consult an agronomist before taking action.*"
    )

    return label_dict, info_md

# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🌿 Crop Disease Detector",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:

    gr.Markdown("""
    # 🌿 Crop Disease Detector
    Upload a photo of a plant leaf to identify diseases and get treatment advice.

    **Supported crops:** Tomato · Potato · Corn · Grape · Apple · Pepper · Strawberry · and more  
    **Model:** EfficientNetB0 fine-tuned on PlantVillage (54,306 images, 38 classes)  
    **Team:** Sun & Nysa — AI Final Project 2026
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 Upload Leaf Photo", height=300)
            submit_btn  = gr.Button("🔍 Detect Disease", variant="primary", size="lg")
            gr.Markdown("""
            **Tips for best results:**
            - Clear, well-lit photo
            - Single leaf in frame
            - Avoid heavy shadows or blur
            """)

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=3, label="Top 3 Predictions")
            info_output  = gr.Markdown()

    submit_btn.click(fn=predict, inputs=[image_input], outputs=[label_output, info_output])
    image_input.change(fn=predict, inputs=[image_input], outputs=[label_output, info_output])

    gr.Markdown("""
    ---
    ### How it works
    1. Your image is resized to 224×224 and pixel values are normalized to [0, 1]
    2. The image passes through EfficientNetB0 (pretrained on ImageNet, fine-tuned on PlantVillage)
    3. The model outputs a probability for each of the 38 plant/disease classes
    4. The top 3 predictions are shown with confidence percentages

    **Dataset:** [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)  
    **GitHub:** [github.com/nlalye/crop-disease-detector](https://github.com)
    """)

# Run locally: python app.py → opens at http://127.0.0.1:7860
# On Hugging Face: Gradio detects the Space environment automatically
if __name__ == "__main__":
    demo.launch()
