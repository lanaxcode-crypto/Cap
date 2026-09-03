from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_erosion
import json, io

app = FastAPI(title="Captcha Solver API")

# Muat model ONNX & labels
session = ort.InferenceSession("captcha_model.onnx")
with open("labels.json") as f:
    classes = json.load(f)

def preprocess(img: Image.Image):
    img = img.resize((48, 48)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
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

@app.post("/solve")
async def solve_captcha(file: UploadFile = File(...)):
    contents = await file.read()
    full_img = Image.open(io.BytesIO(contents)).convert("RGB")
    w, h = full_img.size
    
    arr = np.array(full_img)
    bg = np.array([13, 13, 21])
    diff = np.abs(arr - bg).sum(axis=2)
    binary = (diff > 60).astype(bool)

    # Denoise: buang garis tipis & titik noise < 2 pixel
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    denoised = binary_erosion(binary, structure=struct)

    labeled, num_features = label(denoised)
    
    comps = []
    CROP_SIZE = 48
    
    for i in range(1, num_features + 1):
        points = np.argwhere(labeled == i)
        # Objek ikon asli memiliki area minimal 120 pixel
        if len(points) > 120:
            y_min, x_min = points.min(axis=0)
            y_max, x_max = points.max(axis=0)
            # Pastikan bukan garis pinggir/border noise
            bw = x_max - x_min + 1
            bh = y_max - y_min + 1
            if bw >= 15 and bh >= 15:
                cx, cy = int((x_min + x_max) // 2), int((y_min + y_max) // 2)
                comps.append((len(points), cx, cy))

    # Urutkan berdasarkan ukuran area objek terbesar
    comps.sort(key=lambda x: x[0], reverse=True)
    # Ambil tepat 6 ikon utama
    top_6 = comps[:6]
    # Urutkan posisi dari kiri ke kanan
    top_6.sort(key=lambda x: x[1])

    results = []
    upside_down_target = None
    icon_x_target = None

    for idx, (_, cx, cy) in enumerate(top_6):
        crop_x = max(0, min(w - CROP_SIZE, cx - CROP_SIZE // 2))
        crop_y = max(0, min(h - CROP_SIZE, cy - CROP_SIZE // 2))
        crop = full_img.crop((crop_x, crop_y, crop_x + CROP_SIZE, crop_y + CROP_SIZE))

        inp = preprocess(crop)
        out = session.run(None, {"input": inp})
        pred = classes[int(np.argmax(out[0]))]
        
        info = {"index": idx, "prediction": pred, "x": cx, "y": cy}
        results.append(info)
        
        if pred == "upside_down" and upside_down_target is None:
            upside_down_target = info
        elif pred == "icon_x" and icon_x_target is None:
            icon_x_target = info

    # Aturan: pilih upside_down dulu, jika tidak ada baru icon_x
    if upside_down_target is not None:
        target = upside_down_target
        reason = "upside_down"
    elif icon_x_target is not None:
        target = icon_x_target
        reason = "icon_x"
    else:
        target = results[0] if results else {"x": 170, "y": 150}
        reason = "fallback"

    return {
        "target": target,
        "selected_by": reason,
        "all_icons": results
    }
