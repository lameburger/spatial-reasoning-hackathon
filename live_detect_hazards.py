"""
Live construction hazard detection using webcam or video stream.
Uses YOLO11 Construction Hazard Detection model.
"""
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse
import json


# Map all Construction Hazard Detection model classes
CONSTRUCTION_HAZARD_CLASSES = {
    'person': 'person',
    'machinery': 'excavator',
    'vehicle': 'dump_truck',
    'hardhat': 'hardhat',
    'mask': 'mask',
    'no-hardhat': 'no_hardhat',
    'no-mask': 'no_mask',
    'no-safety vest': 'no_safety_vest',
    'safety cone': 'safety_cone',
    'safety vest': 'safety_vest',
}

COCO_TO_HAZARD = {
    'person': 'person',
    'truck': 'dump_truck',
    'car': 'dump_truck',
    'bus': 'dump_truck',
}

# Hazard detection rules
HAZARD_RULES = {
    "heavy_equipment_proximity": {
        "description": "Heavy equipment proximity - person near heavy machinery",
        "requires": [["excavator", "dump_truck"], "person"],
        "severity": "high"
    },
    "ppe_violation_no_hardhat": {
        "description": "PPE violation - person without hardhat",
        "requires": [["no_hardhat"], "person"],
        "severity": "high"
    },
    "ppe_violation_no_safety_vest": {
        "description": "PPE violation - person without safety vest",
        "requires": ["no_safety_vest"],
        "severity": "high"
    },
}


def calculate_proximity(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate distance between two bounding boxes."""
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
            if isinstance(req1, list):
                req1_detections = []
                for class_name in req1:
                    if class_name in by_class:
                        req1_detections.extend(by_class[class_name])
            else:
                req1_detections = by_class.get(req1, [])
            
            req2_detections = by_class.get(req2, []) if isinstance(req2, str) else []
            
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
            req = required[0]
            if isinstance(req, list):
                for class_name in req:
                    if class_name in by_class and by_class[class_name]:
                        for det in by_class[class_name]:
                            hazards.append({
                                "hazard_type": hazard_name,
                                "description": rule["description"],
                                "severity": rule["severity"],
                                "objects": [{"class": det["class_name"], "confidence": det["confidence"]}]
                            })
                        break
            else:
                if req in by_class and by_class[req]:
                    for det in by_class[req]:
                        hazards.append({
                            "hazard_type": hazard_name,
                            "description": rule["description"],
                            "severity": rule["severity"],
                            "objects": [{"class": det["class_name"], "confidence": det["confidence"]}]
                        })
    
    return hazards


def map_class_name(class_name: str, model_names: dict) -> str:
    """Map detected class to our hazard class."""
    class_name_lower = class_name.lower()
    
    if class_name_lower in CONSTRUCTION_HAZARD_CLASSES:
        return CONSTRUCTION_HAZARD_CLASSES[class_name_lower]
    if class_name_lower in COCO_TO_HAZARD:
        return COCO_TO_HAZARD[class_name_lower]
    
    # Fuzzy matching
    if 'person' in class_name_lower:
        return 'person'
    if 'machinery' in class_name_lower or 'machine' in class_name_lower:
        return 'excavator'
    if 'vehicle' in class_name_lower or 'truck' in class_name_lower:
        return 'dump_truck'
    if 'hardhat' in class_name_lower or 'helmet' in class_name_lower:
        return 'hardhat' if 'no' not in class_name_lower else 'no_hardhat'
    if 'safety vest' in class_name_lower or 'vest' in class_name_lower:
        return 'safety_vest' if 'no' not in class_name_lower else 'no_safety_vest'
    if 'mask' in class_name_lower:
        return 'mask' if 'no' not in class_name_lower else 'no_mask'
    if 'cone' in class_name_lower:
        return 'safety_cone'
    
    return None


def live_detection(
    model_path: str = None,
    source: int = 0,  # 0 for webcam
    conf_threshold: float = 0.25,
):
    """Live hazard detection from webcam or video stream."""
    
    # Auto-detect model
    if model_path is None:
        possible_paths = [
            Path("models/models/pt/best_yolo11n.pt"),
            Path("models/models/pt/best_yolo11s.pt"),
            Path("models/models/pt/best_yolo11m.pt"),
            Path("models/models/pt/best_yolo11l.pt"),
        ]
        for path in possible_paths:
            if path.exists():
                model_path = str(path)
                break
        if model_path is None:
            model_path = "yolov8n.pt"
    
    print("=" * 60)
    print("LIVE CONSTRUCTION HAZARD DETECTION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Source: {'Webcam' if source == 0 else source}")
    print("Press 'q' to quit")
    print()
    
    # Load model
    model = YOLO(model_path)
    model_names = model.names
    
    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)
        result = results[0]
        
        # Extract and map detections
        detections = []
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                detected_class = model_names[class_id]
                
                hazard_class = map_class_name(detected_class, model_names)
                if hazard_class:
                    detections.append({
                        "class_name": hazard_class,
                        "original_class": detected_class,
                        "bbox": bbox.tolist(),
                        "confidence": conf
                    })
        
        # Detect hazards
        hazards = detect_hazards(detections)
        
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
        h, w = annotated_frame.shape[:2]
        cv2.putText(annotated_frame, 
                   f"Detections: {len(detections)} | Hazards: {len(hazards)}",
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Construction Hazard Detection', annotated_frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description="Live Construction Hazard Detection")
    parser.add_argument("--model", type=str, default=None,
                       help="YOLO model path (auto-detects if not provided)")
    parser.add_argument("--source", type=int, default=0,
                       help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    
    args = parser.parse_args()
    
    live_detection(
        model_path=args.model,
        source=args.source,
        conf_threshold=args.conf
    )


if __name__ == "__main__":
    main()

