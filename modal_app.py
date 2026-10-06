import base64
import io

import modal

app = modal.App("layerscraft-ai-worker")

MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_PATH = "/models/sam_vit_b.pth"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "libglib2.0-0", "libgl1")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "opencv-python-headless>=4.9.0",
        "numpy>=1.26.0",
        "Pillow>=10.0.0",
        "fastapi[standard]>=0.115.0",
    )
    .pip_install("git+https://github.com/facebookresearch/segment-anything.git")
    .run_commands(
        "mkdir -p /models",
        f"wget -q -O {MODEL_PATH} {MODEL_URL}",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=60,
    timeout=600,
)
class LayersCraftSAM:
    @modal.enter()
    def load_model(self):
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
        sam.to(device=device)

        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=24,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            min_mask_region_area=900,
        )

    @modal.method()
    def segment(self, image_base64: str, max_layers: int = 18):
        import numpy as np
        from PIL import Image

        if not image_base64:
            raise ValueError("image is required")

        if image_base64.startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]

        raw = base64.b64decode(image_base64)
        source = Image.open(io.BytesIO(raw)).convert("RGB")
        image_np = np.array(source)
        height, width = image_np.shape[:2]

        max_layers = max(1, min(int(max_layers), 30))
        masks = self.generator.generate(image_np)
        masks.sort(key=lambda item: item["area"], reverse=True)

        layers = []
        accepted_masks = []
        total_area = float(height * width)

        for item in masks:
            if len(layers) >= max_layers:
                break

            mask = item["segmentation"]
            area_ratio = float(item["area"]) / total_area

            if area_ratio > 0.94 or area_ratio < 0.002:
                continue

            duplicate = False
            for previous in accepted_masks:
                intersection = np.logical_and(mask, previous).sum()
                union = np.logical_or(mask, previous).sum()
                if union and (intersection / union) > 0.90:
                    duplicate = True
                    break

            if duplicate:
                continue

            accepted_masks.append(mask)

            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[:, :, :3] = image_np
            rgba[:, :, 3] = mask.astype(np.uint8) * 255

            output = io.BytesIO()
            Image.fromarray(rgba).save(output, format="PNG", optimize=True)
            png_base64 = base64.b64encode(output.getvalue()).decode("ascii")

            x, y, box_width, box_height = [int(value) for value in item["bbox"]]
            layers.append(
                {
                    "id": f"layer_{len(layers) + 1}",
                    "name": f"Object {len(layers) + 1}",
                    "type": "object",
                    "bbox": {
                        "x": x,
                        "y": y,
                        "width": box_width,
                        "height": box_height,
                    },
                    "area_ratio": round(area_ratio, 6),
                    "png_base64": png_base64,
                }
            )

        return {
            "version": "0.2.0",
            "width": width,
            "height": height,
            "model": "sam-vit-b",
            "layers": layers,
        }


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def segment(payload: dict):
    image_base64 = payload.get("image")
    max_layers = payload.get("max_layers", 18)

    if not image_base64:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="image is required")

    return LayersCraftSAM().segment.remote(image_base64, max_layers)


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    return {"ok": True, "service": "layerscraft-ai-worker", "version": "0.2.0"}
