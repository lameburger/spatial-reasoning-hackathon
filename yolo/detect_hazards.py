"""
Construction Hazard Detection
Uses pre-trained YOLOv8 or construction-specific model
"""
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse
import json
import os


# Map COCO classes to our construction hazard classes
COCO_TO_HAZARD = {
    'person': 'person',
    'truck': 'dump_truck',
    'car': 'dump_truck',
    'bus': 'dump_truck',
}

# Construction Hazard Detection model classes (from yihong1120/Construction-Hazard-Detection-YOLO11)
# Full class list: 0:Hardhat, 1:Mask, 2:NO-Hardhat, 3:NO-Mask, 4:NO-Safety Vest, 
#                 5:Person, 6:Safety Cone, 7:Safety Vest, 8:Machinery, 9:Vehicle
# Note: Class 9 is Vehicle (not Utility Pole in the actual model)
CONSTRUCTION_HAZARD_CLASSES = {
    # Core hazard classes
    'person': 'person',           # Class 5
    'machinery': 'excavator',     # Class 8 - map to excavator (covers excavator, crane, forklift)
    'vehicle': 'dump_truck',      # Class 9 (Vehicle)
    
    # PPE classes - map to hazard detection
    'hardhat': 'hardhat',         # Class 0 - Safety equipment present
    'mask': 'mask',               # Class 1 - Safety equipment present
    'no-hardhat': 'no_hardhat',   # Class 2 - PPE violation
    'no-mask': 'no_mask',         # Class 3 - PPE violation
    'no-safety vest': 'no_safety_vest',  # Class 4 - PPE violation
    'safety cone': 'safety_cone', # Class 6 - Safety equipment
    'safety vest': 'safety_vest', # Class 7 - Safety equipment present
}

# Generic construction classes mapping
CONSTRUCTION_CLASSES = {
    'person': 'person',
    'excavator': 'excavator',
    'crane': 'crane',
    'ladder': 'ladder',
    'scaffold': 'scaffold',
    'dump_truck': 'dump_truck',
    'forklift': 'forklift',
    'truck': 'dump_truck',
    'car': 'dump_truck',
    'machinery': 'excavator',  # Map machinery to excavator
    'vehicle': 'dump_truck',
}

# Hazard detection rules
HAZARD_RULES = {
    "suspended_load_hazard": {
        "description": "Suspended load hazard - crane/lifting equipment with person nearby",
        "requires": [["crane"], "person"],
        "severity": "high"
    },
    "heavy_equipment_proximity": {
        "description": "Heavy equipment proximity - person near heavy machinery",
        "requires": [["excavator", "dump_truck", "forklift"], "person"],
        "severity": "high"
    },
    "person_on_ladder": {
        "description": "Person on ladder - potential fall hazard",
        "requires": [["ladder"], "person"],
        "severity": "medium"
    },
    "person_on_scaffold": {
        "description": "Person on scaffold - height safety concern",
        "requires": [["scaffold"], "person"],
        "severity": "medium"
    },
    "ppe_violation_no_hardhat": {
        "description": "PPE violation - person without hardhat",
        "requires": [["no_hardhat"], "person"],
        "severity": "high"
    },
    "ppe_violation_no_safety_vest": {
        "description": "PPE violation - person without safety vest",
        "requires": ["no_safety_vest"],  # Person is implied if no_safety_vest detected
        "severity": "high"
    },
    "ppe_violation_no_mask": {
        "description": "PPE violation - person without mask",
        "requires": [["no_mask"], "person"],
        "severity": "medium"
    }
}


def calculate_proximity(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate distance between two bounding boxes (normalized)."""
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


def detect_hazards(detections: List[Dict], proximity_threshold: float = 0.3) -> List[Dict]:
    """Detect hazards based on object combinations."""
    hazards = []
    by_class = {}
    
    for det in detections:
        class_name = det["class_name"]
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append(det)
    
    # Check each hazard rule
    for hazard_name, rule in HAZARD_RULES.items():
        required = rule["requires"]
        
        if len(required) == 2:
            req1, req2 = required
            
            # Handle list of classes (any of these)
            if isinstance(req1, list):
                req1_detections = []
                for class_name in req1:
                    if class_name in by_class:
                        req1_detections.extend(by_class[class_name])
            else:
                req1_detections = by_class.get(req1, [])
            
            req2_detections = by_class.get(req2, []) if isinstance(req2, str) else []
            
            # Check proximity for spatial hazards
            if req1_detections and req2_detections:
                for det1 in req1_detections:
                    for det2 in req2_detections:
                        distance = calculate_proximity(det1["bbox"], det2["bbox"])
                        if distance < proximity_threshold:
                            hazards.append({
                                "hazard_type": hazard_name,
                                "description": rule["description"],
                                "severity": rule["severity"],
                                "objects": [
                                    {"class": det1["class_name"], "confidence": det1["confidence"]},
                                    {"class": det2["class_name"], "confidence": det2["confidence"]}
                                ],
                                "proximity": float(distance)
                            })
        
        elif len(required) == 1:
            # Single requirement (e.g., PPE violations - no_safety_vest implies person)
            req = required[0]
            if isinstance(req, list):
                # Check if any of these classes are present
                for class_name in req:
                    if class_name in by_class and by_class[class_name]:
                        for det in by_class[class_name]:
                            hazards.append({
                                "hazard_type": hazard_name,
                                "description": rule["description"],
                                "severity": rule["severity"],
                                "objects": [
                                    {"class": det["class_name"], "confidence": det["confidence"]}
                                ]
                            })
                        break
            else:
                if req in by_class and by_class[req]:
                    for det in by_class[req]:
                        hazards.append({
                            "hazard_type": hazard_name,
                            "description": rule["description"],
                            "severity": rule["severity"],
                            "objects": [
                                {"class": det["class_name"], "confidence": det["confidence"]}
                            ]
                        })
    
    return hazards


def map_class_name(class_name: str, model_names: dict) -> str:
    """Map detected class to our hazard class - maps ALL detected classes."""
    class_name_lower = class_name.lower()
    
    # Check Construction Hazard Detection model classes first (maps everything)
    if class_name_lower in CONSTRUCTION_HAZARD_CLASSES:
        return CONSTRUCTION_HAZARD_CLASSES[class_name_lower]
    
    # Handle variations and exact matches
    # PPE classes
    if class_name_lower in ['hardhat', 'helmet']:
        return 'hardhat'
    if class_name_lower in ['no-hardhat', 'no hardhat', 'no_hardhat']:
        return 'no_hardhat'
    if class_name_lower in ['mask', 'face mask']:
        return 'mask'
    if class_name_lower in ['no-mask', 'no mask', 'no_mask']:
        return 'no_mask'
    if class_name_lower in ['safety vest', 'vest', 'safety_vest']:
        return 'safety_vest'
    if class_name_lower in ['no-safety vest', 'no safety vest', 'no_safety_vest', 'no-safety-vest']:
        return 'no_safety_vest'
    if class_name_lower in ['safety cone', 'cone', 'safety_cone']:
        return 'safety_cone'
    
    # Core classes
    if class_name_lower in ['person', 'people', 'worker', 'worker']:
        return 'person'
    if class_name_lower in ['machinery', 'machine', 'equipment']:
        return 'excavator'  # Map machinery to excavator
    if class_name_lower in ['vehicle', 'truck', 'car', 'dump truck', 'dump_truck']:
        return 'dump_truck'
    
    # Check generic construction classes
    if class_name_lower in CONSTRUCTION_CLASSES:
        return CONSTRUCTION_CLASSES[class_name_lower]
    
    # Check COCO mapping
    if class_name_lower in COCO_TO_HAZARD:
        return COCO_TO_HAZARD[class_name_lower]
    
    # Try fuzzy matching
    if 'excavator' in class_name_lower or 'digger' in class_name_lower:
        return 'excavator'
    if 'crane' in class_name_lower or 'hook' in class_name_lower or 'load' in class_name_lower:
        return 'crane'
    if 'ladder' in class_name_lower:
        return 'ladder'
    if 'scaffold' in class_name_lower:
        return 'scaffold'
    if 'forklift' in class_name_lower or 'fork' in class_name_lower:
        return 'forklift'
    if 'truck' in class_name_lower or 'dump' in class_name_lower:
        return 'dump_truck'
    if 'machinery' in class_name_lower or 'machine' in class_name_lower:
        return 'excavator'  # Map machinery to excavator as catch-all
    
    return None  # Unknown class


def detect_video(
    model_path: str = None,
    video_path: str = "vlm_fine_tuned/site.mp4",
    conf_threshold: float = 0.25,
    output_dir: str = "hazard_detection_results"
):
    """Detect hazards in video."""
    
    # Auto-detect model
    if model_path is None:
        # Try Construction Hazard Detection model first
        possible_paths = [
            Path("models/models/pt/best_yolo11n.pt"),
            Path("models/models/pt/best_yolo11s.pt"),
            Path("models/models/pt/best_yolo11m.pt"),
            Path("models/models/pt/best_yolo11l.pt"),
            Path("models/best.pt"),
            Path("models/yolo11n.pt"),
        ]
        
        for path in possible_paths:
            if path.exists():
                model_path = str(path)
                print(f"✅ Using Construction Hazard Detection model: {model_path}")
                break
        
        if model_path is None:
            model_path = "yolov8n.pt"
            print(f"Using standard YOLOv8 model: {model_path}")
    
    print("=" * 60)
    print("CONSTRUCTION HAZARD DETECTION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print()
    
    # Load model
    model = YOLO(model_path)
    
    # Get model class names
    model_names = model.names
    print(f"Model has {len(model_names)} classes")
    print(f"Sample classes: {list(model_names.values())[:10]}")
    print()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem
    output_video_path = output_path / f"{video_name}_hazards.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Process video
    frame_count = 0
    total_detections = 0
    total_hazards = 0
    detected_classes = set()
    all_detections = []  # Store all detections for summary
    
    print(f"Processing {total_frames} frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)
        result = results[0]
        
        # Extract and map detections - MAP ALL CLASSES
        detections = []
        all_detected_original = set()  # Track original model classes
        if result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                detected_class = model_names[class_id]
                all_detected_original.add(detected_class)
                
                # Map to hazard class - map ALL classes
                hazard_class = map_class_name(detected_class, model_names)
                if hazard_class:  # Only skip if mapping returns None
                    detections.append({
                        "class_name": hazard_class,
                        "original_class": detected_class,
                        "bbox": bbox.tolist(),
                        "confidence": conf
                    })
                    detected_classes.add(hazard_class)
                    all_detections.append(hazard_class)  # Track for summary
        
        # Detect hazards
        hazards = detect_hazards(detections)
        
        total_detections += len(detections)
        total_hazards += len(hazards)
        
        # Draw annotations
        annotated_frame = result.plot()
        
        # Add hazard overlay
        if hazards:
            y_offset = 30
            for hazard in hazards[:3]:
                text = f"⚠️ {hazard['hazard_type']} ({hazard['severity']})"
                cv2.putText(annotated_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
        
        # Add detection count
        cv2.putText(annotated_frame, 
                   f"Detections: {len(detections)} | Hazards: {len(hazards)}",
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames... "
                  f"Detections: {total_detections}, Hazards: {total_hazards}")
    
    cap.release()
    out.release()
    
    # Summary
    summary = {
        "video_path": str(video_path),
        "model_path": model_path,
        "total_frames": frame_count,
        "total_detections": total_detections,
        "total_hazards": total_hazards,
        "avg_detections_per_frame": total_detections / max(frame_count, 1),
        "avg_hazards_per_frame": total_hazards / max(frame_count, 1),
        "output_video": str(output_video_path),
        "detected_classes": sorted(list(detected_classes)),
        "all_model_classes": {str(k): v for k, v in model_names.items()},
        "detections_by_class": {}
    }
    
    # Count detections per class (from all frames)
    class_counts = {}
    for class_name in all_detections:
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    summary["detections_by_class"] = class_counts
    
    # Save summary
    json_path = output_path / f"{video_name}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DETECTION COMPLETE")
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Total hazards: {total_hazards}")
    print(f"\nDetected classes ({len(detected_classes)}):")
    for class_name in sorted(detected_classes):
        count = class_counts.get(class_name, 0)
        print(f"  - {class_name}: {count} detections")
    print(f"\nOutput video: {output_video_path}")
    print(f"Summary: {json_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Construction Hazard Detection")
    parser.add_argument("--video", type=str, default="vlm_fine_tuned/site.mp4",
                       help="Path to video file")
    parser.add_argument("--model", type=str, default=None,
                       help="YOLO model path (auto-detects if not provided)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--output", type=str, default="hazard_detection_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    detect_video(
        model_path=args.model,
        video_path=args.video,
        conf_threshold=args.conf,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
