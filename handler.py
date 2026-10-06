import base64
import io
import os

import numpy as np
from PIL import Image
import torch
import runpod

from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = os.getenv(
    "SAM_CHECKPOINT",
    "/models/sam_vit_b.pth"
)

sam = sam_model_registry["vit_b"](
    checkpoint=MODEL_PATH
)

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

    image = Image.open(
        io.BytesIO(raw)
    ).convert("RGB")

    return np.array(image)


def png_base64(array):

    output = io.BytesIO()

    Image.fromarray(array).save(
        output,
        format="PNG",
        optimize=True
    )

    return base64.b64encode(
        output.getvalue()
    ).decode("ascii")


def handler(job):

    data = job.get("input", {})

    image = decode_image(
        data.get("image")
    )

    height, width = image.shape[:2]

    max_layers = int(
        data.get("max_layers", 18)
    )

    masks = generator.generate(image)

    masks.sort(
        key=lambda item: item["area"],
        reverse=True
    )

    layers = []
    accepted_masks = []

    total_area = float(
        height * width
    )

    for item in masks:

        if len(layers) >= max_layers:
            break

        mask = item["segmentation"]

        area_ratio = (
            float(item["area"])
            / total_area
        )

        if area_ratio > 0.94:
            continue

        if area_ratio < 0.002:
            continue

        duplicate = False

        for previous in accepted_masks:

            intersection = np.logical_and(
                mask,
                previous
            ).sum()

            union = np.logical_or(
                mask,
                previous
            ).sum()

            if union > 0:

                iou = intersection / union

                if iou > 0.90:
                    duplicate = True
                    break

        if duplicate:
            continue

        accepted_masks.append(mask)

        rgba = np.zeros(
            (height, width, 4),
            dtype=np.uint8
        )

        rgba[:, :, :3] = image

        rgba[:, :, 3] = (
            mask.astype(np.uint8) * 255
        )

        x, y, box_width, box_height = [
            int(value)
            for value in item["bbox"]
        ]

        layers.append({
            "id": f"layer_{len(layers)+1}",

            "name":
                f"Object {len(layers)+1}",

            "type": "object",

            "bbox": {
                "x": x,
                "y": y,
                "width": box_width,
                "height": box_height
            },

            "area_ratio":
                round(area_ratio, 6),

            "png_base64":
                png_base64(rgba)
        })

    return {
        "version": "0.1.0",
        "width": width,
        "height": height,
        "model": "sam-vit-b",
        "layers": layers
    }


runpod.serverless.start({
    "handler": handler
})
