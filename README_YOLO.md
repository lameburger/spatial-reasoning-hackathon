# Phase 2: YOLO Perception Baseline

This directory contains scripts for training and evaluating YOLOv8 object detectors on the ConstructionSite dataset.

## Object Classes

The detector is trained on the following classes:
- `person` - Workers/persons in the construction site
- `hard_hat` - Safety hard hats
- `harness` - Safety harnesses (or harness straps/D-rings)
- `excavator` - Excavator machinery
- `ladder` - Ladders
- `guardrail` - Safety guardrails

## Setup

1. **Install dependencies:**
```bash
pip install ultralytics datasets huggingface_hub pillow
```

2. **Set Hugging Face token:**
```bash
export HF_TOKEN=your_token_here
# Or run: huggingface-cli login
```

## Workflow

### Step 1: Convert Dataset to YOLO Format

Convert the ConstructionSite dataset from Hugging Face format to YOLO format:

```bash
python convert_to_yolo.py
```

This will:
- Download the dataset (if not cached)
- Extract bounding boxes from dataset fields (`excavator`, `worker_with_white_hard_hat`, violation bounding boxes)
- Convert to YOLO format (normalized center_x, center_y, width, height)
- Save images and labels to `yolo_dataset/` directory
- Generate statistics for train and test splits

**Output structure:**
```
yolo_dataset/
├── images/
│   ├── train/
│   └── test/
├── labels/
│   ├── train/
│   └── test/
├── train_stats.json
└── test_stats.json
```

### Step 2: Train YOLO Model

Train a YOLOv8 model (nano/small recommended for speed):

```bash
# Train nano model (fastest, smallest)
python train_yolo.py --model n --epochs 50 --batch 16

# Train small model (better accuracy)
python train_yolo.py --model s --epochs 50 --batch 16

# Custom training
python train_yolo.py --model n --epochs 100 --batch 32 --img-size 640 --device 0
```

**Arguments:**
- `--model`: Model size (`n`=nano, `s`=small, `m`=medium, `l`=large, `x`=xlarge)
- `--epochs`: Number of training epochs (default: 50)
- `--batch`: Batch size (default: 16)
- `--img-size`: Image size for training (default: 640)
- `--device`: Device to use (`0` for GPU, `cpu` for CPU, `None` for auto)
- `--name`: Run name (default: `construction_yolo`)

**Output:**
- Best model: `runs/detect/construction_yolo/weights/best.pt`
- Last checkpoint: `runs/detect/construction_yolo/weights/last.pt`
- Training plots and metrics in the run directory

### Step 3: Evaluate Model

Evaluate the trained model and compute mAP, precision, recall:

```bash
python evaluate_yolo.py --model runs/detect/construction_yolo/weights/best.pt
```

**Arguments:**
- `--model`: Path to trained model weights (required)
- `--conf`: Confidence threshold (default: 0.5)
- `--iou`: IoU threshold for mAP (default: 0.5)
- `--output`: Output directory for results (default: `evaluation_results`)

**Output:**
- `evaluation_results/metrics.json` - JSON file with all metrics
- `evaluation_results/` - Confusion matrix, PR curves, etc.
- Hard negatives (misses) for retraining

### Step 4: Export for Edge Inference

Export the model to ONNX, TorchScript, or TFLite:

```bash
# Export to ONNX (recommended for most edge devices)
python export_yolo.py --model runs/detect/construction_yolo/weights/best.pt --format onnx

# Export to TorchScript
python export_yolo.py --model runs/detect/construction_yolo/weights/best.pt --format torchscript

# Export to TFLite (quantized int8 for mobile/edge)
python export_yolo.py --model runs/detect/construction_yolo/weights/best.pt --format tflite

# Export to CoreML (for Apple devices)
python export_yolo.py --model runs/detect/construction_yolo/weights/best.pt --format coreml
```

**Arguments:**
- `--model`: Path to trained model weights (required)
- `--format`: Export format (`onnx`, `torchscript`, `tflite`, `coreml`, etc.)
- `--img-size`: Image size for export (default: 640)
- `--no-simplify`: Don't simplify ONNX model
- `--no-optimize`: Don't optimize for inference

## Quick Start Example

```bash
# 1. Convert dataset
export HF_TOKEN=your_token_here
python convert_to_yolo.py

# 2. Train model
python train_yolo.py --model n --epochs 50 --batch 16

# 3. Evaluate
python evaluate_yolo.py --model runs/detect/construction_yolo/weights/best.pt

# 4. Export
python export_yolo.py --model runs/detect/construction_yolo/weights/best.pt --format onnx
```

## Notes

- **Sparse annotations**: If bounding boxes are sparse for certain classes (especially `harness`, `ladder`, `guardrail`), you may need to annotate additional images using tools like CVAT or Label Studio.

- **Hard negatives**: After evaluation, review hard negatives (misses) and consider retraining with these examples to improve performance.

- **Model selection**: 
  - `yolov8n.pt` (nano) - Fastest, smallest, good for edge devices
  - `yolov8s.pt` (small) - Better accuracy, still fast
  - `yolov8m.pt` (medium) - Best accuracy, slower inference

- **Data augmentation**: YOLOv8 applies automatic data augmentation during training (mosaic, mixup, etc.)

## Expected Deliverables

✅ Fast object detector producing bounding boxes + confidences for:
- person
- hard_hat
- harness
- excavator
- ladder
- guardrail

✅ Evaluation metrics: mAP@0.5 and per-class precision/recall

✅ Exported models ready for edge inference (ONNX/TorchScript/TFLite)

