"""
Evaluate YOLO model and compute mAP, precision, recall metrics.
"""
from ultralytics import YOLO
from pathlib import Path
import json
import argparse


def evaluate_yolo(
    model_path: str,
    data_yaml: str = "yolo_data.yaml",
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    save_hard_negatives: bool = True,
    output_dir: str = "evaluation_results",
):
    """
    Evaluate YOLO model and generate metrics.
    
    Args:
        model_path: Path to trained model weights (.pt file)
        data_yaml: Path to data.yaml configuration
        conf_threshold: Confidence threshold for evaluation
        iou_threshold: IoU threshold for mAP calculation
        save_hard_negatives: Whether to save hard negative examples
        output_dir: Directory to save evaluation results
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating on validation set...")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print()
    
    # Run validation
    metrics = model.val(
        data=data_yaml,
        conf=conf_threshold,
        iou=iou_threshold,
        save_json=True,  # Save results in JSON format
        save_hybrid=True,  # Save hybrid labels for hard negatives
        plots=True,  # Generate confusion matrix and other plots
        project=output_dir,
        name="evaluation",
    )
    
    # Extract metrics
    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_metrics": {}
    }
    
    # Per-class metrics
    if hasattr(metrics.box, "maps"):
        class_names = model.names
        for class_id, map50 in enumerate(metrics.box.maps):
            class_name = class_names.get(class_id, f"class_{class_id}")
            results["per_class_metrics"][class_name] = {
                "mAP50": float(map50)
            }
    
    # Save metrics to JSON
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"mAP@0.5: {results['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP50-95']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print("\nPer-class mAP@0.5:")
    for class_name, class_metrics in results["per_class_metrics"].items():
        print(f"  {class_name}: {class_metrics['mAP50']:.4f}")
    
    print(f"\nResults saved to: {output_path}")
    print(f"Metrics JSON: {metrics_path}")
    
    if save_hard_negatives:
        print("\nHard negatives (misses) can be found in the evaluation directory")
        print("Use these for retraining to improve model performance.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model weights (.pt file)")
    parser.add_argument("--data", type=str, default="yolo_data.yaml",
                        help="Path to data.yaml configuration")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for mAP")
    parser.add_argument("--output", type=str, default="evaluation_results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    evaluate_yolo(
        model_path=args.model,
        data_yaml=args.data,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

