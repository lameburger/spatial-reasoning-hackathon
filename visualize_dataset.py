"""
Visualize YOLO dataset with bounding boxes overlaid on images.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import argparse


# Class colors (RGB)
CLASS_COLORS = {
    0: (255, 0, 0),      # person - red
    1: (0, 255, 0),      # hard_hat - green
    2: (0, 0, 255),      # harness - blue
    3: (255, 165, 0),    # excavator - orange
    4: (255, 0, 255),    # ladder - magenta
    5: (0, 255, 255),    # guardrail - cyan
}

CLASS_NAMES = {
    0: "person",
    1: "hard_hat",
    2: "harness",
    3: "excavator",
    4: "ladder",
    5: "guardrail",
}


def yolo_to_bbox(center_x, center_y, width, height, img_width, img_height):
    """Convert YOLO format to pixel coordinates."""
    x_center = center_x * img_width
    y_center = center_y * img_height
    w = width * img_width
    h = height * img_height
    
    x_min = int(x_center - w / 2)
    y_min = int(y_center - h / 2)
    x_max = int(x_center + w / 2)
    y_max = int(y_center + h / 2)
    
    return x_min, y_min, x_max, y_max


def draw_bboxes(image, label_path, class_names, class_colors):
    """Draw bounding boxes on image from YOLO label file."""
    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    if not label_path.exists():
        return image
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                x_min, y_min, x_max, y_max = yolo_to_bbox(
                    center_x, center_y, width, height, img_width, img_height
                )
                
                # Get color and class name
                color = class_colors.get(class_id, (255, 255, 255))
                class_name = class_names.get(class_id, f"class_{class_id}")
                
                # Draw rectangle
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                
                # Draw label background
                label_text = f"{class_name}"
                if font:
                    bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    text_width = len(label_text) * 6
                    text_height = 12
                
                draw.rectangle(
                    [x_min, y_min - text_height - 4, x_min + text_width + 4, y_min],
                    fill=color
                )
                
                # Draw label text
                draw.text(
                    (x_min + 2, y_min - text_height - 2),
                    label_text,
                    fill=(255, 255, 255),
                    font=font
                )
    
    return image


def visualize_samples(
    dataset_dir: str = "yolo_dataset",
    split: str = "train",
    num_samples: int = 10,
    random_seed: int = 42,
    output_dir: str = "visualizations"
):
    """Visualize random samples from the dataset."""
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images" / split
    labels_dir = dataset_path / "labels" / split
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg"))
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    # Filter to only images with annotations (non-empty label files)
    annotated_images = []
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            annotated_images.append((img_path, label_path))
    
    if not annotated_images:
        print(f"No annotated images found in {split} split")
        return
    
    print(f"Found {len(annotated_images)} annotated images in {split} split")
    print(f"Showing {min(num_samples, len(annotated_images))} random samples\n")
    
    # Random sample
    random.seed(random_seed)
    samples = random.sample(annotated_images, min(num_samples, len(annotated_images)))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Visualize each sample
    for idx, (img_path, label_path) in enumerate(samples):
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Draw bounding boxes
        image = draw_bboxes(image, label_path, CLASS_NAMES, CLASS_COLORS)
        
        # Count annotations
        num_annotations = 0
        class_counts = {}
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    num_annotations += 1
                    class_id = int(parts[0])
                    class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Save visualization
        output_file = output_path / f"{split}_{idx+1:02d}_{img_path.stem}.jpg"
        image.save(output_file)
        
        print(f"Sample {idx+1}: {img_path.name}")
        print(f"  Annotations: {num_annotations} ({', '.join([f'{k}:{v}' for k, v in class_counts.items()])})")
        print(f"  Saved to: {output_file}")
        print()
    
    print(f"\nAll visualizations saved to: {output_path}")
    print(f"\nClass color legend:")
    for class_id, class_name in CLASS_NAMES.items():
        color = CLASS_COLORS[class_id]
        print(f"  {class_name}: RGB{color}")


def show_statistics(dataset_dir: str = "yolo_dataset"):
    """Show dataset statistics."""
    dataset_path = Path(dataset_dir)
    
    for split in ["train", "test"]:
        stats_file = dataset_path / f"{split}_stats.json"
        if stats_file.exists():
            import json
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print(f"\n{split.upper()} Split Statistics:")
            print(f"  Total images: {stats['total_images']}")
            print(f"  Images with annotations: {stats['images_with_annotations']} ({stats['images_with_annotations']/stats['total_images']*100:.1f}%)")
            print(f"  Total annotations: {stats['total_annotations']}")
            print(f"  Average annotations per annotated image: {stats['total_annotations']/stats['images_with_annotations']:.2f}")
            print(f"\n  Annotations per class:")
            for class_name, count in stats['annotations_per_class'].items():
                percentage = (count / stats['total_annotations'] * 100) if stats['total_annotations'] > 0 else 0
                print(f"    {class_name}: {count} ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO dataset")
    parser.add_argument("--dataset", type=str, default="yolo_dataset",
                        help="Dataset directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="Dataset split to visualize")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to visualize")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--output", type=str, default="visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only show statistics, don't create visualizations")
    
    args = parser.parse_args()
    
    # Show statistics
    show_statistics(args.dataset)
    
    if not args.stats_only:
        print("\n" + "=" * 60)
        visualize_samples(
            dataset_dir=args.dataset,
            split=args.split,
            num_samples=args.num_samples,
            random_seed=args.seed,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()

