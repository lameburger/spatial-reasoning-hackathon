# Construction Hazard Detection

Fast construction hazard detection using **YOLO11 Construction Hazard Detection model**. Detects person, machinery, and vehicles with hazard proximity analysis!

## Quick Start

```bash
# 1. Download the Construction Hazard Detection model (one-time)
python download_construction_model.py n

# 2. Detect hazards (auto-detects the model)
python detect_hazards.py --video vlm_fine_tuned/site.mp4
```

## What It Detects

### ✅ Using Construction Hazard Detection Model (YOLO11)

**Detected Classes:**
- ✅ **person** - Class 5: Person
- ✅ **excavator** - Class 8: Machinery (mapped to excavator - covers excavator, crane, forklift)
- ✅ **dump_truck** - Class 10: Vehicle (mapped to dump_truck)

**Full Model Classes (11 total):**
- Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest
- Person, Safety Cone, Safety Vest
- **Machinery** (excavator/crane/forklift)
- **Vehicle** (dump truck)
- Utility Pole

## Hazard Detection Rules

The system automatically detects:
- **Heavy equipment proximity**: Person + Machinery/Vehicle (high severity)
- **Suspended load hazard**: Person + Crane (if crane class available)
- **Person on ladder**: Person + Ladder (if ladder class available)
- **Person on scaffold**: Person + Scaffold (if scaffold class available)

## Model Information

**Source:** [HuggingFace - Construction-Hazard-Detection-YOLO11](https://huggingface.co/yihong1120/Construction-Hazard-Detection-YOLO11)

**GitHub:** [yihong1120/Construction-Hazard-Detection](https://github.com/yihong1120/Construction-Hazard-Detection)

**Model Performance:**
- YOLO11n: mAP@0.5 = 58.0, mAP@0.5-0.95 = 34.2
- YOLO11s: mAP@0.5 = 70.1, mAP@0.5-0.95 = 44.8
- YOLO11m: mAP@0.5 = 73.3, mAP@0.5-0.95 = 42.6
- YOLO11l: mAP@0.5 = 77.3, mAP@0.5-0.95 = 54.6
- YOLO11x: mAP@0.5 = 82.0, mAP@0.5-0.95 = 61.7

## Usage

```bash
# Basic detection
python detect_hazards.py --video vlm_fine_tuned/site.mp4

# With custom confidence threshold
python detect_hazards.py --video vlm_fine_tuned/site.mp4 --conf 0.3

# Specify model explicitly
python detect_hazards.py --model models/models/pt/best_yolo11n.pt --video vlm_fine_tuned/site.mp4

# Download different model sizes
python download_construction_model.py s  # small (better accuracy)
python download_construction_model.py m  # medium
python download_construction_model.py l  # large (best accuracy)
```

## Output

- **Annotated video** with bounding boxes and hazard warnings
- **JSON summary** with detection statistics
- Saved to: `hazard_detection_results/`

## Files

- `detect_hazards.py` - Main detection script
- `download_construction_model.py` - Download model from HuggingFace
- `quick_train.py` - Fast training script (for additional classes)
- `GET_CONSTRUCTION_MODEL.md` - Alternative download methods

## Results

The model successfully detects:
- ✅ **Person** - Workers on construction site
- ✅ **Machinery** - Excavators, cranes, forklifts (mapped to excavator)
- ✅ **Vehicle** - Dump trucks, vehicles (mapped to dump_truck)

Hazard detection works by analyzing proximity between person and machinery/vehicles!
