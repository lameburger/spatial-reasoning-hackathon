"""
Convert ConstructionSite dataset to YOLO format.
Maps dataset bounding boxes to YOLO TXT format per image.
"""
from datasets import load_dataset
from huggingface_hub import login, HfFolder
from PIL import Image
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import json


# Class mapping: YOLO class_id -> class_name
CLASSES = {
    0: "person",
    1: "hard_hat",
    2: "harness",
    3: "excavator",
    4: "ladder",
    5: "guardrail",
}

# Reverse mapping for lookup
CLASS_NAME_TO_ID = {v: k for k, v in CLASSES.items()}

# Dataset field to class mapping
DATASET_FIELD_MAPPING = {
    "worker_with_white_hard_hat": "person",  # Workers are persons
    "excavator": "excavator",
    "rebar": None,  # Not in our class list, skip
    # Violations may contain person boxes - we'll extract from rule violations
}


def authenticate_hf():
    """Authenticate with Hugging Face."""
    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if token:
        login(token=token)
        return True
    return False


def normalize_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert normalized [x_min, y_min, x_max, y_max] to YOLO format [center_x, center_y, width, height].
    
    Args:
        bbox: Normalized bounding box [x_min, y_min, x_max, y_max] in 0-1 range
        img_width: Image width (for validation)
        img_height: Image height (for validation)
    
    Returns:
        Tuple of (center_x, center_y, width, height) all normalized 0-1
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Ensure values are in valid range
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))
    
    # Calculate center and dimensions
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    return center_x, center_y, width, height


def extract_annotations_from_sample(sample: Dict) -> List[Tuple[int, List[float]]]:
    """
    Extract all bounding box annotations from a dataset sample.
    
    Returns:
        List of (class_id, bbox) tuples where bbox is [x_min, y_min, x_max, y_max] normalized
    """
    annotations = []
    
    # Extract from direct object fields
    for field_name, class_name in DATASET_FIELD_MAPPING.items():
        if class_name is None:
            continue
        if field_name in sample and sample[field_name]:
            class_id = CLASS_NAME_TO_ID.get(class_name)
            if class_id is not None:
                for bbox in sample[field_name]:
                    if len(bbox) == 4:
                        annotations.append((class_id, bbox))
    
    # Extract from violation bounding boxes (usually persons)
    for rule_num in range(1, 5):
        rule_key = f"rule_{rule_num}_violation"
        if sample.get(rule_key) and isinstance(sample[rule_key], dict):
            violation = sample[rule_key]
            if "bounding_box" in violation:
                for bbox in violation["bounding_box"]:
                    if len(bbox) == 4:
                        # Violations typically involve persons
                        annotations.append((CLASS_NAME_TO_ID["person"], bbox))
    
    return annotations


def convert_dataset_to_yolo(
    dataset_name: str = "LouisChen15/ConstructionSite",
    output_dir: str = "yolo_dataset",
    split: str = "train"
):
    """
    Convert dataset to YOLO format.
    
    Args:
        dataset_name: Hugging Face dataset name
        output_dir: Output directory for YOLO dataset
        split: Dataset split to convert ('train' or 'test')
    """
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split)
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images" / split
    labels_dir = output_path / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_images": 0,
        "images_with_annotations": 0,
        "total_annotations": 0,
        "annotations_per_class": {name: 0 for name in CLASSES.values()}
    }
    
    print(f"Converting {len(dataset)} images...")
    for idx, sample in enumerate(dataset):
        stats["total_images"] += 1
        image = sample["image"]
        image_id = sample.get("image_id", f"{idx:07d}")
        
        # Save image
        image_path = images_dir / f"{image_id}.jpg"
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(image_path)
        
        # Extract annotations
        annotations = extract_annotations_from_sample(sample)
        
        # Save YOLO format labels
        label_path = labels_dir / f"{image_id}.txt"
        img_width, img_height = image.size
        
        with open(label_path, "w") as f:
            for class_id, bbox in annotations:
                center_x, center_y, width, height = normalize_bbox_to_yolo(bbox, img_width, img_height)
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                stats["total_annotations"] += 1
                stats["annotations_per_class"][CLASSES[class_id]] += 1
        
        if annotations:
            stats["images_with_annotations"] += 1
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} images...")
    
    # Save statistics
    stats_path = output_path / f"{split}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total images: {stats['total_images']}")
    print(f"Images with annotations: {stats['images_with_annotations']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Annotations per class:")
    for class_name, count in stats["annotations_per_class"].items():
        print(f"  {class_name}: {count}")
    
    return stats


def main():
    """Main conversion function."""
    if not authenticate_hf():
        print("Error: Authentication failed")
        return
    
    # Convert train and test splits
    print("=" * 60)
    print("Converting TRAIN split...")
    print("=" * 60)
    train_stats = convert_dataset_to_yolo(split="train")
    
    print("\n" + "=" * 60)
    print("Converting TEST split...")
    print("=" * 60)
    test_stats = convert_dataset_to_yolo(split="test")
    
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Train: {train_stats['images_with_annotations']}/{train_stats['total_images']} images annotated")
    print(f"Test: {test_stats['images_with_annotations']}/{test_stats['total_images']} images annotated")


if __name__ == "__main__":
    main()

