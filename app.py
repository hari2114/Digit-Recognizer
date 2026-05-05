"""
Handwritten Digit Recognizer
==============================
CNN trained on MNIST (99%+ accuracy).
Flask web app with HTML5 canvas for drawing digits and getting predictions.

Install:
    pip install flask tensorflow numpy pillow

Usage:
    python app.py
    Open: http://127.0.0.1:5000
"""

import os
import io
import base64
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template_string

# TensorFlow import with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not installed. Run: pip install tensorflow")


app = Flask(__name__)
MODEL_PATH = "digit_model.h5"


# ── Model ─────────────────────────────────────────────────────────────────
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_save():
    print("[TRAIN] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis] / 255.0
    x_test  = x_test[..., np.newaxis]  / 255.0

    model = build_model()
    print("[TRAIN] Training CNN... (~2-3 min on CPU)")
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[TRAIN] Test Accuracy: {acc:.4f}")
    model.save(MODEL_PATH)
    print(f"[TRAIN] Model saved to {MODEL_PATH}")
    return model


def load_or_train():
    if not TF_AVAILABLE:
        return None
    if os.path.exists(MODEL_PATH):
        print("[INFO] Loading saved model...")
        return keras.models.load_model(MODEL_PATH)
    return train_and_save()


model = load_or_train()


# ── Image Preprocessing ───────────────────────────────────────────────────
def preprocess_canvas(image_data_url: str) -> np.ndarray:
    """Convert base64 canvas image to 28x28 MNIST-compatible array."""
    header, data = image_data_url.split(",", 1)
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")      # grayscale
    img = ImageOps.invert(img)                                 # white digit on black
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ..., np.newaxis]                    # (1,28,28,1)


# ── HTML Template ─────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Digit Recognizer</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',sans-serif;background:#0a0a0a;color:#f0f0f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
  .card{background:#111;border:1px solid #222;border-radius:16px;padding:40px;max-width:460px;width:100%;text-align:center}
  h1{font-size:26px;font-weight:700;margin-bottom:8px}
  p.sub{color:#666;font-size:13px;margin-bottom:28px}
  canvas{border:2px solid #2a2a2a;border-radius:12px;cursor:crosshair;background:#000;touch-action:none}
  .btns{display:flex;gap:10px;margin-top:16px}
  button{flex:1;padding:12px;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
  #predictBtn{background:#fff;color:#000}#predictBtn:hover{background:#ddd}
  #clearBtn{background:#181818;color:#888;border:1px solid #2a2a2a}#clearBtn:hover{color:#fff}
  .result{margin-top:24px;padding:20px;background:#181818;border-radius:10px;display:none}
  .result-digit{font-size:64px;font-weight:800;line-height:1;color:#fff}
  .result-conf{font-size:13px;color:#666;margin-top:6px}
  .bars{margin-top:16px;display:flex;flex-direction:column;gap:4px}
  .bar-row{display:flex;align-items:center;gap:8px;font-size:12px}
  .bar-label{width:14px;text-align:right;color:#888}
  .bar-track{flex:1;background:#222;border-radius:4px;height:8px;overflow:hidden}
  .bar-fill{height:100%;border-radius:4px;background:#4f8ef7;transition:width .4s}
  .bar-fill.top{background:#fff}
  .bar-pct{width:36px;text-align:right;color:#555;font-size:11px}
</style>
</head>
<body>
<div class="card">
  <h1>✏️ Digit Recognizer</h1>
  <p class="sub">Draw a digit (0–9) below — CNN predicts in real time.</p>
  <canvas id="canvas" width="280" height="280"></canvas>
  <div class="btns">
    <button id="predictBtn" onclick="predict()">Predict</button>
    <button id="clearBtn" onclick="clearCanvas()">Clear</button>
  </div>
  <div class="result" id="result">
    <div class="result-digit" id="digit">?</div>
    <div class="result-conf" id="conf"></div>
    <div class="bars" id="bars"></div>
  </div>
</div>
<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;
ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.strokeStyle = '#fff';

function pos(e) {
  const r = canvas.getBoundingClientRect();
  const src = e.touches ? e.touches[0] : e;
  return {x: src.clientX - r.left, y: src.clientY - r.top};
}
canvas.addEventListener('mousedown', e => { drawing=true; ctx.beginPath(); const p=pos(e); ctx.moveTo(p.x,p.y); });
canvas.addEventListener('mousemove', e => { if(!drawing)return; const p=pos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); });
canvas.addEventListener('mouseup', () => drawing=false);
canvas.addEventListener('touchstart', e => { e.preventDefault(); drawing=true; ctx.beginPath(); const p=pos(e); ctx.moveTo(p.x,p.y); });
canvas.addEventListener('touchmove', e => { e.preventDefault(); if(!drawing)return; const p=pos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); });
canvas.addEventListener('touchend', () => drawing=false);

function clearCanvas() { ctx.clearRect(0,0,280,280); document.getElementById('result').style.display='none'; }

async function predict() {
  const dataURL = canvas.toDataURL('image/png');
  const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({image:dataURL})});
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  document.getElementById('result').style.display='block';
  document.getElementById('digit').textContent = data.digit;
  document.getElementById('conf').textContent = `Confidence: ${data.confidence}`;
  const bars = document.getElementById('bars');
  bars.innerHTML = data.probabilities.map((p,i) => `
    <div class="bar-row">
      <span class="bar-label">${i}</span>
      <div class="bar-track"><div class="bar-fill ${i==data.digit?'top':''}" style="width:${(p*100).toFixed(1)}%"></div></div>
      <span class="bar-pct">${(p*100).toFixed(1)}%</span>
    </div>`).join('');
}
</script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "TensorFlow not available. Install with: pip install tensorflow"}), 500

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        arr = preprocess_canvas(data["image"])
        proba = model.predict(arr, verbose=0)[0]
        digit = int(np.argmax(proba))
        return jsonify({
            "digit": digit,
            "confidence": f"{proba[digit]:.1%}",
            "probabilities": [round(float(p), 4) for p in proba],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("[INFO] Digit Recognizer running at http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
