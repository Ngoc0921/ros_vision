# YOLO / vision model assets

Place **exported inference artifacts** here (or in a subfolder). **Do not** point runtime config at an entire Ultralytics *training run directory* as if it were a single model file.

## Resolution order (runtime)

When configuring `model_path` (see `config/yolo_model.yaml`), choose the first path that exists on disk:

1. **`*.onnx`** — preferred for portable inference (e.g. `weights/best.onnx` or `box_yolo.onnx`).
2. **`weights/best.pt`** — Ultralytics PyTorch export.
3. **`weights/last.pt`** — fallback only if `best.pt` is missing.

## Suggested layout (example)

```text
models/
  README.md
  box_detector/              # optional grouping folder
    weights/
      best.pt
      best.onnx              # preferred once exported
    args.yaml                # training metadata (not loaded as weights)
```

## Current workspace status

If no `*.onnx` or `weights/*.pt` is present yet, inference nodes remain **TODO**; `config/yolo_model.yaml` still documents the **expected** canonical relative path pattern.
