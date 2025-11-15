"""
Export YOLO model to ONNX, TorchScript, or TFLite for edge inference.
"""
from ultralytics import YOLO
import argparse
from pathlib import Path


def export_yolo(
    model_path: str,
    format: str = "onnx",  # onnx, torchscript, tflite, coreml, etc.
    imgsz: int = 640,
    simplify: bool = True,  # Simplify ONNX model
    optimize: bool = True,  # Optimize for inference
):
    """
    Export YOLO model to different formats.
    
    Args:
        model_path: Path to trained model weights (.pt file)
        format: Export format ('onnx', 'torchscript', 'tflite', 'coreml', etc.)
        imgsz: Image size for export
        simplify: Simplify ONNX model (ONNX only)
        optimize: Optimize for inference
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Exporting to {format.upper()} format...")
    print(f"Image size: {imgsz}")
    print()
    
    # Export
    if format.lower() == "onnx":
        success = model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=simplify,
            optimize=optimize,
        )
    elif format.lower() == "torchscript":
        success = model.export(
            format="torchscript",
            imgsz=imgsz,
            optimize=optimize,
        )
    elif format.lower() == "tflite":
        success = model.export(
            format="tflite",
            imgsz=imgsz,
            int8=True,  # Quantize to int8 for edge devices
        )
    elif format.lower() == "coreml":
        success = model.export(
            format="coreml",
            imgsz=imgsz,
        )
    else:
        # Generic export
        success = model.export(
            format=format,
            imgsz=imgsz,
        )
    
    if success:
        # Find exported file
        model_dir = Path(model_path).parent
        model_name = Path(model_path).stem
        
        # YOLO exports to the same directory as the model
        exported_files = list(model_dir.glob(f"{model_name}.*"))
        exported_files = [f for f in exported_files if f.suffix != ".pt"]
        
        print("\n" + "=" * 60)
        print("Export successful!")
        print("=" * 60)
        print("Exported files:")
        for file in exported_files:
            print(f"  {file}")
        
        return exported_files
    else:
        print("Export failed!")
        return None


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to different formats")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model weights (.pt file)")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "torchscript", "tflite", "coreml", "engine", "openvino"],
                        help="Export format")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Image size for export")
    parser.add_argument("--no-simplify", action="store_true",
                        help="Don't simplify ONNX model (ONNX only)")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Don't optimize for inference")
    
    args = parser.parse_args()
    
    export_yolo(
        model_path=args.model,
        format=args.format,
        imgsz=args.img_size,
        simplify=not args.no_simplify,
        optimize=not args.no_optimize,
    )


if __name__ == "__main__":
    main()

