import torch
from transformers import Idefics3Processor, Idefics3ForConditionalGeneration
from peft import PeftModel
from PIL import Image
import cv2
import numpy as np
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import os
import json

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
ADAPTER_PATH = "./smolvlm_construction_finetuned"
BATCH_SIZE_CACHE_FILE = "optimal_batch_size.json"

# Optimization settings
USE_TORCH_COMPILE = True
TARGET_VRAM_USAGE = 0.90  # Use 90% of available VRAM
MAX_NEW_TOKENS = 128
USE_CACHE = True

# Global cache for batch size
_OPTIMAL_BATCH_SIZE = None

print("Loading model and processor with maximum VRAM optimization...")
processor = Idefics3Processor.from_pretrained(MODEL_ID)

# Load model with optimizations
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# Apply torch.compile for faster inference
if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
    print("Compiling model with torch.compile() for maximum performance...")
    model = torch.compile(model, mode="max-autotune", fullgraph=False)

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_vram:.2f} GB")
    print(f"CUDA Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

def find_optimal_batch_size(force_recalculate=False):
    """Dynamically find the optimal batch size that maximizes VRAM usage"""
    global _OPTIMAL_BATCH_SIZE
    
    # Return cached value if available
    if _OPTIMAL_BATCH_SIZE is not None and not force_recalculate:
        print(f"Using cached optimal batch size: {_OPTIMAL_BATCH_SIZE}")
        return _OPTIMAL_BATCH_SIZE
    
    # Try to load from file cache
    if os.path.exists(BATCH_SIZE_CACHE_FILE) and not force_recalculate:
        try:
            with open(BATCH_SIZE_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                cached_batch_size = cache_data.get('optimal_batch_size')
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                cached_gpu = cache_data.get('gpu_name')
                
                if cached_batch_size and cached_gpu == gpu_name:
                    print(f"Loaded optimal batch size from cache: {cached_batch_size}")
                    _OPTIMAL_BATCH_SIZE = cached_batch_size
                    return cached_batch_size
                else:
                    print("Cache invalid (different GPU), recalculating...")
        except Exception as e:
            print(f"Could not load cache: {e}")
    
    if not torch.cuda.is_available():
        return 4
    
    print("\nFinding optimal batch size for maximum VRAM utilization...")
    print("This only needs to run once and will be cached.")
    
    # Start with a conservative estimate
    batch_size = 4
    max_batch_size = 32
    
    # Create dummy frames for testing
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    while batch_size < max_batch_size:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Test with current batch size
            images = [Image.fromarray(dummy_frame) for _ in range(batch_size)]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Test"}
                ]
            }]
            
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=images, text=[prompt] * batch_size, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # Just do forward pass to check memory, don't generate
                    _ = model(**inputs)
                    
                    # Do minimal generation test to simulate real usage
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=2,  # Only generate 2 tokens for testing
                        do_sample=False,
                        use_cache=USE_CACHE,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
            
            # Check memory usage
            peak_memory = torch.cuda.max_memory_allocated(0) / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            usage_ratio = peak_memory / total_memory
            
            print(f"Batch size {batch_size}: {peak_memory:.2f}GB / {total_memory:.2f}GB ({usage_ratio*100:.1f}%)")
            
            if usage_ratio > TARGET_VRAM_USAGE:
                # Went over target, use previous batch size
                optimal_batch = max(4, batch_size - 2)
                break
            
            # Try larger batch
            batch_size += 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # OOM, use previous batch size
                optimal_batch = max(4, batch_size - 2)
                break
            else:
                raise e
    else:
        optimal_batch = batch_size
    
    torch.cuda.empty_cache()
    
    # Cache the result
    _OPTIMAL_BATCH_SIZE = optimal_batch
    try:
        cache_data = {
            'optimal_batch_size': optimal_batch,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'vram_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None
        }
        with open(BATCH_SIZE_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Cached optimal batch size to {BATCH_SIZE_CACHE_FILE}")
    except Exception as e:
        print(f"Could not save cache: {e}")
    
    print(f"\nOptimal batch size: {optimal_batch}\n")
    return optimal_batch

def analyze_frames_batch(frames, batch_size):
    """Analyze multiple frames in a batch for maximum GPU utilization"""
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe construction site and identify safety violations."}
        ]
    }]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Process all images together
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=USE_CACHE,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
    
    # Decode all outputs
    responses = []
    for i in range(len(frames)):
        response = processor.decode(outputs[i], skip_special_tokens=True)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        responses.append(response)
    
    return responses

def live_video_analysis_max_vram(source=0, analyze_every_n_frames=30, is_live=False):
    """
    Maximized VRAM video analysis with dynamic batch sizing
    Supports both video files and live webcam input
    
    Args:
        source: Video file path (str) or webcam index (int, default 0)
        analyze_every_n_frames: Process every Nth frame
        is_live: True for webcam, False for video file
    """
    # Find optimal batch size first
    BATCH_SIZE = find_optimal_batch_size()
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    
    # Get video FPS for proper playback speed
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0 or is_live:
        video_fps = 30  # Default for webcam or undetectable FPS
    
    # For live video, minimize delay. For file, match video FPS
    delay_ms = 1 if is_live else int(1000 / video_fps)
    
    frame_count = 0
    current_analyses = ["Starting analysis..."] * BATCH_SIZE
    frame_buffer = []
    fps_history = deque(maxlen=30)
    
    source_type = "LIVE WEBCAM" if is_live else "VIDEO FILE"
    print(f"\n{'='*60}")
    print(f"Source: {source_type}")
    print(f"Video FPS: {video_fps}, Display delay: {delay_ms}ms per frame")
    print(f"Optimal batch size: {BATCH_SIZE}")
    print(f"Analyzing every {analyze_every_n_frames} frames")
    print(f"Target VRAM usage: {TARGET_VRAM_USAGE*100:.0f}%")
    print(f"Press 'q' to quit")
    print(f"{'='*60}\n")
    
    last_time = time.time()
    analysis_times = []
    total_frames_analyzed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Process remaining frames in buffer before exiting
            if len(frame_buffer) > 0:
                print(f"\nProcessing final batch of {len(frame_buffer)} frames...")
                current_analyses = analyze_frames_batch(frame_buffer, len(frame_buffer))
                total_frames_analyzed += len(frame_buffer)
            break
        
        # Collect frames for batch processing
        if frame_count % analyze_every_n_frames == 0:
            frame_buffer.append(frame.copy())
            
            # Show progress while collecting frames
            if len(frame_buffer) < BATCH_SIZE and len(frame_buffer) % 4 == 0:
                print(f"Collecting frames for batch: {len(frame_buffer)}/{BATCH_SIZE}")
            
            # Process batch when buffer is full or after timeout
            if len(frame_buffer) >= BATCH_SIZE:
                start_time = time.time()
                print(f"\n{'='*60}")
                print(f"Analyzing batch of {BATCH_SIZE} frames (starting at frame {frame_count})...")
                
                current_analyses = analyze_frames_batch(frame_buffer, BATCH_SIZE)
                
                analysis_time = time.time() - start_time
                analysis_times.append(analysis_time)
                fps = BATCH_SIZE / analysis_time
                fps_history.append(fps)
                total_frames_analyzed += BATCH_SIZE
                
                # Memory stats
                current_vram = torch.cuda.memory_allocated(0) / 1024**3
                peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                print(f"Batch time: {analysis_time:.2f}s | Throughput: {fps:.2f} FPS")
                print(f"VRAM: {current_vram:.2f}GB / {total_vram:.2f}GB ({current_vram/total_vram*100:.1f}%)")
                print(f"Peak VRAM: {peak_vram:.2f}GB ({peak_vram/total_vram*100:.1f}%)")
                print(f"Average throughput: {np.mean(fps_history):.2f} FPS")
                print(f"Total frames analyzed: {total_frames_analyzed}")
                
                # Only show sample result for first few batches to reduce spam
                if len(analysis_times) <= 3:
                    print(f"\nSample result (frame {frame_count}):")
                    print(f"  {current_analyses[0][:150]}...")
                print(f"{'='*60}")
                
                # Clear buffer
                frame_buffer = []
                
                # Reset peak memory counter for next batch
                torch.cuda.reset_peak_memory_stats()
        
        # Display frame with latest analysis
        display_frame = frame.copy()
        
        # Add text overlay
        y_offset = 30
        max_width = 80
        
        # Display latest analysis
        analysis_text = current_analyses[0] if current_analyses else "Processing..."
        words = analysis_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) < max_width:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)
        
        # Add performance stats
        current_fps = 1.0 / (time.time() - last_time) if time.time() > last_time else 0
        cv2.putText(display_frame, f"Display FPS: {current_fps:.1f}", (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if fps_history:
            cv2.putText(display_frame, f"Analysis: {np.mean(fps_history):.1f} FPS | Batch: {BATCH_SIZE}", 
                       (10, display_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(display_frame, f"Analyzed: {total_frames_analyzed}", 
                   (10, display_frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for i, line in enumerate(lines[:5]):
            cv2.putText(display_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
        
        cv2.imshow('Construction Safety Analysis - MAX VRAM', display_frame)
        
        last_time = time.time()
        frame_count += 1
        
        # Use proper delay for real-time playback
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    if analysis_times:
        print("\n" + "="*60)
        print("FINAL PERFORMANCE STATISTICS")
        print("="*60)
        print(f"Optimal batch size: {BATCH_SIZE}")
        print(f"Total batches processed: {len(analysis_times)}")
        print(f"Total frames analyzed: {total_frames_analyzed}")
        print(f"Average batch time: {np.mean(analysis_times):.2f}s")
        print(f"Average throughput: {np.mean([BATCH_SIZE/t for t in analysis_times]):.2f} FPS")
        print(f"Best batch time: {np.min(analysis_times):.2f}s ({BATCH_SIZE/np.min(analysis_times):.2f} FPS)")
        print(f"Total analysis time: {sum(analysis_times):.2f}s")
        
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"\nVRAM utilization: {peak_vram:.2f}GB / {total_vram:.2f}GB ({peak_vram/total_vram*100:.1f}%)")
        print("="*60)

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized VLM inference with maximum VRAM utilization')
    parser.add_argument('source', nargs='?', default='0', 
                        help='Video file path or webcam index (default: 0 for webcam)')
    parser.add_argument('--live', action='store_true', 
                        help='Enable live webcam mode (reduces display latency)')
    parser.add_argument('--analyze-every', type=int, default=30, 
                        help='Analyze every Nth frame (default: 30)')
    parser.add_argument('--recalculate-batch', action='store_true',
                        help='Force recalculation of optimal batch size')
    
    args = parser.parse_args()
    
    # Recalculate batch size if requested
    if args.recalculate_batch:
        print("Forcing recalculation of optimal batch size...")
        find_optimal_batch_size(force_recalculate=True)
    
    # Parse source: if it's a digit, treat as webcam index, otherwise as file path
    try:
        source = int(args.source)
        is_live = True
        print(f"Using webcam index: {source}")
    except ValueError:
        source = args.source
        is_live = args.live
        print(f"Using video file: {source}")
    
    live_video_analysis_max_vram(
        source=source, 
        analyze_every_n_frames=args.analyze_every,
        is_live=is_live
    )

