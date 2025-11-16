"""
YOLO-based construction hazard detection.
Detects dangerous scenes and objects in construction site images/videos.
"""
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import json


# Hazard detection rules - detect dangerous scenes
HAZARD_RULES = {
    "suspended_load_hazard": {
        "description": "Suspended load hazard - crane/lifting equipment with potential people underneath",
        "requires": ["crane", "person"],  # Crane + person nearby
        "severity": "high"
    },
    "heavy_equipment_proximity": {
        "description": "Heavy equipment proximity hazard - workers too close to heavy machinery",
        "requires": [["excavator", "dump_truck", "forklift"], "person"],  # Any heavy equipment + person
        "severity": "high"
    },
    "person_on_ladder": {
        "description": "Person on ladder - potential fall hazard",
        "requires": ["ladder", "person"],
        "severity": "medium"
    },
    "person_on_scaffold": {
        "description": "Person on scaffold - height safety concern",
        "requires": ["scaffold", "person"],
        "severity": "medium"
    }
}


def calculate_proximity(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate proximity between two bounding boxes.
    Returns distance between box centers (normalized).
    """
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    
    distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    return distance


def detect_hazards(detections: List[Dict], proximity_threshold: float = 0.3) -> List[Dict]:
    """
    Detect hazards based on object combinations and proximity.
    
    Args:
        detections: List of detection dicts with keys: class_name, bbox, confidence
        proximity_threshold: Normalized distance threshold for proximity hazards
    
    Returns:
        List of detected hazards
    """
    hazards = []
    
    # Group detections by class
    by_class = {}
    for det in detections:
        class_name = det["class_name"]
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append(det)
    
    # Check each hazard rule
    for hazard_name, rule in HAZARD_RULES.items():
        required = rule["requires"]
        
        # Handle nested requirements (e.g., ["excavator", "dump_truck", "forklift"] means any of these)
        if len(required) == 2:
            req1, req2 = required
            
            # If req1 is a list, check if any of those classes are present
            if isinstance(req1, list):
                req1_detections = []
                for class_name in req1:
                    if class_name in by_class:
                        req1_detections.extend(by_class[class_name])
            else:
                req1_detections = by_class.get(req1, [])
            
            req2_detections = by_class.get(req2, []) if isinstance(req2, str) else []
            
            # Check if both requirements are met and objects are close
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
    
    return hazards


def detect_image(
    model_path: str,
    image_path: str,
    conf_threshold: float = 0.25,
    save_output: bool = True,
    output_dir: str = "detection_results"
) -> Dict:
    """
    Detect hazards in a single image.
    
    Args:
        model_path: Path to YOLO model weights (.pt file)
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save annotated output image
        output_dir: Directory to save results
    
    Returns:
        Dictionary with detections and hazards
    """
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path, conf=conf_threshold)
    result = results[0]
    
    # Extract detections
    detections = []
    if result.boxes is not None:
        boxes = result.boxes
        class_names = model.names
        
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = class_names[class_id]
            
            detections.append({
                "class_name": class_name,
                "class_id": class_id,
                "bbox": bbox.tolist(),
                "confidence": conf
            })
    
    # Detect hazards
    hazards = detect_hazards(detections)
    
    # Prepare output
    output = {
        "image_path": str(image_path),
        "detections": detections,
        "hazards": hazards,
        "num_detections": len(detections),
        "num_hazards": len(hazards)
    }
    
    # Save annotated image
    if save_output:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Draw detections and hazards
        annotated_img = result.plot()
        
        # Save image
        image_name = Path(image_path).stem
        output_image_path = output_path / f"{image_name}_detected.jpg"
        cv2.imwrite(str(output_image_path), annotated_img)
        output["output_image"] = str(output_image_path)
        
        # Save JSON results
        json_path = output_path / f"{image_name}_results.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        output["output_json"] = str(json_path)
    
    return output


def detect_video(
    model_path: str,
    video_path: str,
    conf_threshold: float = 0.25,
    save_output: bool = True,
    output_dir: str = "detection_results",
    show_preview: bool = False
) -> Dict:
    """
    Detect hazards in a video file.
    
    Args:
        model_path: Path to YOLO model weights (.pt file)
        video_path: Path to input video
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save annotated output video
        output_dir: Directory to save results
        show_preview: Whether to show live preview window
    
    Returns:
        Dictionary with summary statistics
    """
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video writer
    output_video_path = None
    out = None
    if save_output:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem
        output_video_path = output_path / f"{video_name}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Process video
    frame_count = 0
    total_detections = 0
    total_hazards = 0
    frame_results = []
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold)
        result = results[0]
        
        # Extract detections
        detections = []
        if result.boxes is not None:
            boxes = result.boxes
            class_names = model.names
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = class_names[class_id]
                
                detections.append({
                    "class_name": class_name,
                    "class_id": class_id,
                    "bbox": bbox.tolist(),
                    "confidence": conf
                })
        
        # Detect hazards
        hazards = detect_hazards(detections)
        
        total_detections += len(detections)
        total_hazards += len(hazards)
        
        # Draw annotations
        annotated_frame = result.plot()
        
        # Add hazard text overlay
        if hazards:
            y_offset = 30
            for hazard in hazards[:3]:  # Show up to 3 hazards
                text = f"⚠️ {hazard['hazard_type']} ({hazard['severity']})"
                cv2.putText(annotated_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
        
        # Save frame
        if out is not None:
            out.write(annotated_frame)
        
        # Show preview
        if show_preview:
            cv2.imshow('Hazard Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    if out is not None:
        out.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    # Prepare summary
    summary = {
        "video_path": str(video_path),
        "total_frames": frame_count,
        "total_detections": total_detections,
        "total_hazards": total_hazards,
        "avg_detections_per_frame": total_detections / max(frame_count, 1),
        "avg_hazards_per_frame": total_hazards / max(frame_count, 1),
        "output_video": str(output_video_path) if output_video_path else None
    }
    
    # Save summary
    if save_output:
        json_path = output_path / f"{Path(video_path).stem}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        summary["output_json"] = str(json_path)
    
    return summary


def detect_webcam(
    model_path: str,
    conf_threshold: float = 0.25,
    camera_id: int = 0
):
    """
    Detect hazards in real-time from webcam feed.

    Args:
        model_path: Path to YOLO model weights (.pt file)
        conf_threshold: Confidence threshold for detections
        camera_id: Camera device ID (0 for default webcam)
    """
    # Load model
    model = YOLO(model_path)

    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera {camera_id}")

    print("=" * 60)
    print("WEBCAM HAZARD DETECTION")
    print("=" * 60)
    print(f"Press 'q' to quit")
    print(f"Press 's' to save current frame")
    print("=" * 60)

    frame_count = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam")
                break

            # Run inference
            results = model(frame, conf=conf_threshold, verbose=False)
            result = results[0]

            # Extract detections
            detections = []
            if result.boxes is not None:
                boxes = result.boxes
                class_names = model.names

                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = class_names[class_id]

                    detections.append({
                        "class_name": class_name,
                        "class_id": class_id,
                        "bbox": bbox.tolist(),
                        "confidence": conf
                    })

            # Detect hazards
            hazards = detect_hazards(detections)

            # Draw annotations
            annotated_frame = result.plot()

            # Add hazard text overlay
            if hazards:
                y_offset = 30
                for hazard in hazards[:3]:  # Show up to 3 hazards
                    text = f"WARNING: {hazard['hazard_type']} ({hazard['severity']})"
                    cv2.putText(annotated_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 30

            # Add info overlay
            info_text = f"Detections: {len(detections)} | Hazards: {len(hazards)} | Frame: {frame_count}"
            cv2.putText(annotated_frame, info_text, (10, annotated_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show frame
            cv2.imshow('Webcam Hazard Detection', annotated_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current frame
                output_dir = Path("detection_results/webcam")
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(save_path), annotated_frame)
                print(f"Saved frame to {save_path}")
                saved_count += 1

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        if saved_count > 0:
            print(f"Saved {saved_count} frames to detection_results/webcam/")


def main():
    parser = argparse.ArgumentParser(description="YOLO Construction Hazard Detection")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to YOLO model weights (.pt file)")
    parser.add_argument("--source", type=str, default=None,
                       help="Path to image or video file, or 'webcam' for webcam")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", type=str, default="detection_results",
                       help="Output directory for results")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save output files")
    parser.add_argument("--show", action="store_true",
                       help="Show preview window (for video)")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device ID (default: 0)")
    
    args = parser.parse_args()

    # Check if webcam mode
    if args.source is None or args.source.lower() == 'webcam':
        print("Starting webcam detection...")
        detect_webcam(
            model_path=args.model,
            conf_threshold=args.conf,
            camera_id=args.camera
        )
        return

    # Check if source is image or video
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source file not found: {args.source}")
        return

    # Determine file type
    ext = source_path.suffix.lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    if is_video:
        print("=" * 60)
        print("VIDEO HAZARD DETECTION")
        print("=" * 60)
        results = detect_video(
            model_path=args.model,
            video_path=args.source,
            conf_threshold=args.conf,
            save_output=not args.no_save,
            output_dir=args.output_dir,
            show_preview=args.show
        )
        print("\n" + "=" * 60)
        print("DETECTION SUMMARY")
        print("=" * 60)
        print(f"Total frames: {results['total_frames']}")
        print(f"Total detections: {results['total_detections']}")
        print(f"Total hazards: {results['total_hazards']}")
        print(f"Avg detections/frame: {results['avg_detections_per_frame']:.2f}")
        print(f"Avg hazards/frame: {results['avg_hazards_per_frame']:.2f}")
        if results.get('output_video'):
            print(f"Output video: {results['output_video']}")
    else:
        print("=" * 60)
        print("IMAGE HAZARD DETECTION")
        print("=" * 60)
        results = detect_image(
            model_path=args.model,
            image_path=args.source,
            conf_threshold=args.conf,
            save_output=not args.no_save,
            output_dir=args.output_dir
        )
        print("\n" + "=" * 60)
        print("DETECTION RESULTS")
        print("=" * 60)
        print(f"Detections: {results['num_detections']}")
        print(f"Hazards: {results['num_hazards']}")
        
        if results['detections']:
            print("\nDetected objects:")
            for det in results['detections']:
                print(f"  - {det['class_name']}: {det['confidence']:.2f}")
        
        if results['hazards']:
            print("\n⚠️  HAZARDS DETECTED:")
            for hazard in results['hazards']:
                print(f"  - {hazard['hazard_type']} ({hazard['severity']})")
                print(f"    {hazard['description']}")
        else:
            print("\n✓ No hazards detected")
        
        if results.get('output_image'):
            print(f"\nOutput image: {results['output_image']}")


if __name__ == "__main__":
    main()

