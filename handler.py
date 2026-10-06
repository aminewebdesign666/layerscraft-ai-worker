import base64, io, os
import numpy as np
import cv2
from PIL import Image
import torch
import runpod
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("SAM_CHECKPOINT", "/models/sam_vit_b.pth")
sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
sam.to(device=DEVICE)
generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=24,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    min_mask_region_area=900,
)

def decode_image(value):
    if not value:
        raise ValueError("input.image is required")
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)

def png_b64(arr):
    im = Image.fromarray(arr)
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return base64.b64encode(out.getvalue()).decode("ascii")

def handler(job):
    inp = job.get("input", {})
    image = decode_image(inp.get("image"))
    h, w = image.shape[:2]
    max_layers = int(inp.get("max_layers", 18))
    masks = generator.generate(image)
    masks.sort(key=lambda m: m["area"], reverse=True)

    layers = []
    accepted = []
    total = float(h * w)
    for idx, m in enumerate(masks):
        if len(layers) >= max_layers:
            break
        mask = m["segmentation"]
        area = float(m["area"])
        ratio = area / total
        if ratio > 0.94 or ratio < 0.002:
            continue
        # Skip near-duplicate masks.
        duplicate = False
        for prev in accepted:
            inter = np.logical_and(mask, prev).sum()
            union = np.logical_or(mask, prev).sum()
            if union and inter / union > 0.90:
                duplicate = True
                break
        if duplicate:
            continue
        accepted.append(mask)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = mask.astype(np.uint8) * 255
        x, y, bw, bh = [int(v) for v in m["bbox"]]
        layers.append({
            "id": f"layer_{len(layers)+1}",
            "name": f"Object {len(layers)+1}",
            "type": "object",
            "bbox": {"x": x, "y": y, "width": bw, "height": bh},
            "area_ratio": round(ratio, 6),
            "png_base64": png_b64(rgba),
        })

    return {
        "version": "0.1.0",
        "width": w,
        "height": h,
        "model": "sam-vit-b",
        "layers": layers,
        "note": "MVP automatic segmentation. Semantic naming, OCR and background reconstruction are added in the next pipeline stage."
    }

runpod.serverless.start({"handler": handler})
