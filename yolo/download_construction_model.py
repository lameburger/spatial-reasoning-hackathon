"""
Download Construction Hazard Detection model from HuggingFace.
Based on: https://github.com/yihong1120/Construction-Hazard-Detection
"""
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
import sys


def download_model(model_size="n"):
    """
    Download YOLO11 construction hazard detection model.
    
    Args:
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
    """
    repo_id = "yihong1120/Construction-Hazard-Detection-YOLO11"
    
    # First, list available files
    try:
        print("Checking available files in repository...")
        files = list_repo_files(repo_id=repo_id, repo_type="model")
        print(f"Found {len(files)} files")
        model_files = [f for f in files if f.endswith('.pt')]
        print(f"Model files: {model_files}")
        print()
    except Exception as e:
        print(f"Could not list files: {e}")
        model_files = []
    
    # Try different possible paths
    possible_paths = [
        f"models/pt/yolo11{model_size}.pt",
        f"yolo11{model_size}.pt",
        f"models/yolo11{model_size}.pt",
        f"best.pt",
    ]
    
    # Also try any .pt file we found
    if model_files:
        possible_paths = model_files[:1] + possible_paths
    
    model_path = None
    for path in possible_paths:
        try:
            print(f"Trying to download: {path}")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                filename=path,
                local_dir="models",
            )
            model_path = Path(downloaded_path)
            if model_path.exists():
                print(f"✅ Model downloaded successfully!")
                print(f"Location: {model_path.absolute()}")
                break
        except Exception as e:
            print(f"  ⚠️ Not found at {path}: {e}")
            continue
    
    if model_path and model_path.exists():
        print(f"\n✅ Success! Model ready at: {model_path}")
        print(f"\nUse it with:")
        print(f"python detect_hazards.py --model {model_path} --video vlm_fine_tuned/site.mp4")
        return str(model_path)
    else:
        print("\n⚠️ Could not download automatically.")
        print("\nManual download instructions:")
        print("1. Visit: https://huggingface.co/yihong1120/Construction-Hazard-Detection-YOLO11")
        print("2. Click 'Files and versions' tab")
        print("3. Find the .pt model file (e.g., yolo11n.pt)")
        print("4. Click download")
        print("5. Place it in: models/yolo11n.pt")
        print("6. Use: python detect_hazards.py --model models/yolo11n.pt")
        return None


if __name__ == "__main__":
    model_size = sys.argv[1] if len(sys.argv) > 1 else "n"
    model_path = download_model(model_size)
    
    if not model_path:
        print("\n" + "=" * 60)
        print("ALTERNATIVE: Use the GitHub repo directly")
        print("=" * 60)
        print("The model classes are:")
        print("  0: Hardhat")
        print("  1: Mask")
        print("  2: NO-Hardhat")
        print("  3: NO-Mask")
        print("  4: NO-Safety Vest")
        print("  5: Person")
        print("  6: Safety Cone")
        print("  7: Safety Vest")
        print("  8: Machinery")
        print("  9: Utility Pole")
        print("  10: Vehicle")
        print()
        print("These map to:")
        print("  - Person → person")
        print("  - Machinery → excavator/crane/forklift")
        print("  - Vehicle → dump_truck")
