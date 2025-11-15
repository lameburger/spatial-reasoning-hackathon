"""
Train YOLOv8 detector on ConstructionSite dataset.
"""
from ultralytics import YOLO
import argparse
from pathlib import Path


def train_yolo(
    model_size: str = "n",  # n=nano, s=small, m=medium
    data_yaml: str = "yolo_data.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = None,  # None = auto, or "0", "cpu", etc.
    project: str = "runs/detect",
    name: str = "construction_yolo",
):
    """
    Train YOLOv8 model.
    
    Args:
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        data_yaml: Path to data.yaml configuration
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size for training
        device: Device to use (None for auto)
        project: Project directory
        name: Run name
    """
    # Load model
    model_name = f"yolov8{model_size}.pt"
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Verify data.yaml exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}. Run convert_to_yolo.py first.")
    
    print(f"Training configuration:")
    print(f"  Model: {model_name}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Device: {device or 'auto'}")
    print()
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,  # Validate during training
        plots=True,  # Generate training plots
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best model saved to: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Results directory: {results.save_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on ConstructionSite dataset")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                        help="Model size: n=nano, s=small, m=medium, l=large, x=xlarge")
    parser.add_argument("--data", type=str, default="yolo_data.yaml",
                        help="Path to data.yaml configuration")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Image size for training")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., '0', 'cpu', None for auto)")
    parser.add_argument("--name", type=str, default="construction_yolo",
                        help="Run name")
    
    args = parser.parse_args()
    
    train_yolo(
        model_size=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        name=args.name,
    )


if __name__ == "__main__":
    main()

