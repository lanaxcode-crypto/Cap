from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
from PIL import Image
import json, io

app = FastAPI(title="Captcha Solver API")

# Muat model ONNX & labels
session = ort.InferenceSession("captcha_model.onnx")
with open("labels.json") as f:
    classes = json.load(f)

def preprocess(img: Image.Image):
    img = img.resize((48, 48)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # Normalisasi sama seperti saat training
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)   # Tambah batch dimension
    return arr

@app.get("/")
def root():
    return {"status": "running", "classes": classes}

@app.post("/predict")
async def predict_icon(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    input_data = preprocess(img)
    outputs = session.run(None, {"input": input_data})
    pred_idx = int(np.argmax(outputs[0]))
    
    return {
        "prediction": classes[pred_idx],
        "confidence": float(np.max(outputs[0]))
    }

# Endpoint lengkap untuk selesaikan gambar captcha utuh (340x300)
@app.post("/solve")
async def solve_captcha(file: UploadFile = File(...)):
    contents = await file.read()
    full_img = Image.open(io.BytesIO(contents)).convert("RGB")
    w, h = full_img.size
    
    arr = np.array(full_img)
    bg = np.array([13, 13, 21])
    diff = np.abs(arr - bg).sum(axis=2)
    binary = (diff > 60).astype(np.uint8)

    # Denoise
    from scipy.ndimage import label, center_of_mass
    labeled, num_features = label(binary)
    
    crops = []
    centers = []
    CROP_SIZE = 48
    
    for i in range(1, num_features + 1):
        points = np.argwhere(labeled == i)
        if len(points) > 100:  # Abaikan sisa noise
            y_min, x_min = points.min(axis=0)
            y_max, x_max = points.max(axis=0)
            cx, cy = int((x_min + x_max) // 2), int((y_min + y_max) // 2)
            
            crop_x = max(0, min(w - CROP_SIZE, cx - CROP_SIZE // 2))
            crop_y = max(0, min(h - CROP_SIZE, cy - CROP_SIZE // 2))
            crop = full_img.crop((crop_x, crop_y, crop_x + CROP_SIZE, crop_y + CROP_SIZE))
            
            crops.append(crop)
            centers.append((cx, cy))

    # Prediksi semua ikon
    results = []
    upside_down_target = None
    icon_x_target = None

    for idx, crop in enumerate(crops):
        inp = preprocess(crop)
        out = session.run(None, {"input": inp})
        pred = classes[int(np.argmax(out[0]))]
        cx, cy = centers[idx]
        
        info = {"index": idx, "prediction": pred, "x": cx, "y": cy}
        results.append(info)
        
        if pred == "upside_down" and upside_down_target is None:
            upside_down_target = info
        elif pred == "icon_x" and icon_x_target is None:
            icon_x_target = info

    # Logika aturan captcha:
    # "suruh pilih upside down kalo tidak ada baru pilih gambar X"
    if upside_down_target is not None:
        target = upside_down_target
        reason = "upside_down"
    elif icon_x_target is not None:
        target = icon_x_target
        reason = "icon_x"
    else:
        target = results[0] if results else None
        reason = "fallback"

    return {
        "target": target,
        "selected_by": reason,
        "all_icons": results
    }
