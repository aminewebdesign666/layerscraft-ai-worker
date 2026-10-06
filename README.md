# LayersCraft AI Worker

GPU segmentation backend for LayersCraft, migrated from RunPod to Modal.

## Deploy

Install and authenticate Modal:

```bash
pip3 install modal
python3 -m modal setup
```

Then clone this repository, switch to the Modal branch, and deploy:

```bash
git clone https://github.com/aminewebdesign666/layerscraft-ai-worker.git
cd layerscraft-ai-worker
git checkout modal-migration
modal deploy modal_app.py
```

Modal will print public URLs for the `segment` and `health` web endpoints.

## Segment API

Send a POST request to the generated `segment` endpoint:

```json
{
  "image": "<base64 image or data URL>",
  "max_layers": 18
}
```

The response preserves the LayersCraft layer format and returns transparent PNG layers as base64.

## Cost controls

The GPU worker uses a T4, scales to zero when idle, and keeps a 60-second warm window. Start with small test images while using trial credits.
