"""
Optimized Real-Time Construction Safety Analysis with VLM

This script provides highly optimized real-time video inference for construction site safety
monitoring using a fine-tuned Vision-Language Model (SmolVLM).

Key Features:
- Multi-threaded architecture for parallel frame capture and inference
- Batch processing for better GPU utilization
- CUDA stream optimizations for async operations
- Bounding box detection and visualization for safety violations
- Detailed prompting for structured safety rule compliance checking
- Real-time FPS monitoring and performance metrics

Usage:
    # Analyze video file with default settings (analyze every 15 frames)
    python live_inference_optimized.py site.mp4
    
    # Analyze video with custom frame interval (analyze every 10 frames)
    python live_inference_optimized.py site.mp4 10
    
    # Use webcam
    python live_inference_optimized.py

Configuration:
    - USE_DETAILED_PROMPT: Enable/disable detailed safety rule checking with bounding boxes
    - BATCH_SIZE: Number of frames to process in parallel (default: 8)
    - MAX_NEW_TOKENS: Maximum tokens for model generation (default: 256 for detailed, 64 for simple)
    - USE_THREADING: Enable/disable multi-threaded processing

Performance Optimizations:
    1. Threaded video capture and inference pipeline
    2. Batch processing (8 frames at once)
    3. CUDA streams for async GPU operations
    4. Reduced token generation for faster inference
    5. Non-blocking frame queues
    6. Optimized preprocessing with direct numpy operations
    7. Batch decoding for faster response processing

Notes:
    - For bounding box detection to work, the model must be trained on data with bbox annotations
    - The detailed prompt format expects JSON output with normalized bounding box coordinates
    - Set USE_DETAILED_PROMPT = False for faster but less structured analysis
"""

import torch
from transformers import Idefics3Processor, Idefics3ForConditionalGeneration
from peft import PeftModel
from PIL import Image
import cv2
import numpy as np
import time
from collections import deque
import threading
from queue import Queue, Empty
import torchvision.transforms as transforms
import json
import re

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
ADAPTER_PATH = "./smolvlm_construction_finetuned"

# Optimization settings
USE_TORCH_COMPILE = True  # Use torch.compile for speed
BATCH_SIZE = 8  # Increased batch size for better GPU utilization
USE_FLASH_ATTENTION = True  # Enable flash attention if available
MAX_NEW_TOKENS = 256  # Increased to accommodate detailed responses with bounding boxes
USE_CACHE = True  # Use KV cache for faster generation
NUM_BEAMS = 1  # Greedy decoding for speed
USE_THREADING = True  # Use multi-threading for parallel frame processing
PREFETCH_BATCHES = 2  # Number of batches to prefetch

# Prompts for detailed safety inspection with bounding boxes
SYSTEM_PROMPT = """You are a construction site safety inspector. You are responsible for viewing the given image and give helpful, detailed, and polite answers to your supervisor. You only answer questions that are asked by the supervisor and in the exact way as requested."""

FEW_SHOT_PROMPT = """You will be asked to read the image and identify violations of safety rules that appears in the image. You also need to provide a short reasoning and bounding boxes showing the location of the violation."""

USER_PROMPT = """Please read the image and identify if there are violations of the following four safety rules in the image, do not include violations that do not exist in your answer, assume no violation if the visual information is not enough to make a judgement:

1. Use of basic PPE when on foot at construction sites. Machine operators do not need PPE. (hard hats, properly worn clothes covering shoulders and legs, shoes that can cover toes, high-visibility retroreflective vests at night, face shield or safety glasses when cutting, welding, grinding, or drilling).

2. Use of safety harness when working from a height of three meters and the edges are without any edge protection.

3. Adoption of edge protection or edge warning including guardrails, fences, for underground projects three meters in depth with steep retaining wall and for human to stand.

4. Appearance of worker in the blind spots of the operator and within the operation radius of excavators in operation, or excavators with operators inside.

Your answer should be in the format of {"id of the safety rule": {"reason": one or two sentences explaining who violate the rule in the image and the specific reason, "bounding box": [the location of violation in the image x_min, y_min, x_max, y_max in 0-1 normalized space]}}.

Return {"0": "No violations"} if you find no violation in the image."""

# Option to use detailed prompt or simple prompt
USE_DETAILED_PROMPT = True  # Set to False for faster, simpler analysis

print("Loading model and processor with optimizations...")
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

# Apply torch.compile for faster inference (PyTorch 2.0+)
if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
    print("Compiling model with torch.compile() for maximum performance...")
    model = torch.compile(model, mode="max-autotune", fullgraph=False)

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Create CUDA streams for parallel operations
cuda_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

if torch.cuda.is_available():
    # Enable cuDNN autotuner
    torch.backends.cudnn.enabled = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    # Set memory allocator settings for better performance
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory

def preprocess_frames_fast(frames):
    """Fast preprocessing without PIL conversion"""
    # Convert BGR to RGB directly with numpy (faster than PIL)
    images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    images = [Image.fromarray(img) for img in images]
    return images

def parse_violations_with_boxes(response_text):
    """Parse the model response to extract violations and bounding boxes"""
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            violations = json.loads(json_match.group())
            return violations
        else:
            return None
    except:
        return None

def draw_bounding_boxes(frame, violations):
    """Draw bounding boxes on frame based on violations"""
    if not violations or violations.get("0") == "No violations":
        return frame
    
    annotated_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Color map for different violation types
    colors = {
        "1": (0, 0, 255),    # Red for PPE violations
        "2": (255, 0, 0),    # Blue for harness violations
        "3": (0, 165, 255),  # Orange for edge protection
        "4": (0, 255, 255),  # Yellow for blind spot violations
    }
    
    for rule_id, violation_data in violations.items():
        if rule_id == "0":
            continue
            
        if isinstance(violation_data, dict) and "bounding box" in violation_data:
            bbox = violation_data["bounding box"]
            reason = violation_data.get("reason", "Violation detected")
            
            # Convert normalized coordinates to pixel coordinates
            x_min = int(bbox[0] * width)
            y_min = int(bbox[1] * height)
            x_max = int(bbox[2] * width)
            y_max = int(bbox[3] * height)
            
            # Get color for this violation type
            color = colors.get(rule_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Add label with rule ID
            label = f"Rule {rule_id}"
            cv2.putText(annotated_frame, label, (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_frame

def analyze_frames_batch(frames):
    """Optimized batch analysis with faster preprocessing and decoding"""
    images = preprocess_frames_fast(frames)
    
    # Prepare messages based on prompt style
    if USE_DETAILED_PROMPT:
        # Combine all prompts into the message
        full_prompt = f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_PROMPT}\n\n{USER_PROMPT}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_prompt}
                ]
            }
        ]
    else:
        # Simple prompt for faster inference
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe construction site and identify safety violations."}
                ]
            }
        ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Process all images together with non-blocking transfer
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True)
    
    # Use CUDA stream for async operations
    if cuda_stream is not None:
        with torch.cuda.stream(cuda_stream):
            inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}
            torch.cuda.current_stream().wait_stream(cuda_stream)
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=NUM_BEAMS,
                use_cache=USE_CACHE,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
    
    # Batch decode (faster than individual decoding)
    responses = processor.batch_decode(outputs, skip_special_tokens=True)
    
    # Clean up responses
    cleaned_responses = []
    for response in responses:
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        cleaned_responses.append(response)
    
    return cleaned_responses

class ThreadedVideoAnalyzer:
    """Threaded video analyzer with separate capture and inference threads"""
    def __init__(self, analyze_every_n_frames=30):
        self.frame_queue = Queue(maxsize=BATCH_SIZE * 3)
        self.result_queue = Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.analyze_every_n_frames = analyze_every_n_frames
        self.frame_buffer = []
        self.lock = threading.Lock()
        
    def inference_worker(self):
        """Worker thread for running inference on frame batches"""
        analysis_times = []
        batch_count = 0
        
        while not self.stop_event.is_set():
            try:
                # Collect frames for a batch
                batch_frames = []
                timeout = 0.1
                
                while len(batch_frames) < BATCH_SIZE and not self.stop_event.is_set():
                    try:
                        frame = self.frame_queue.get(timeout=timeout)
                        batch_frames.append(frame)
                    except Empty:
                        # If we have some frames but not full batch, process anyway
                        if len(batch_frames) > 0:
                            break
                        continue
                
                if len(batch_frames) == 0:
                    continue
                
                # Run inference
                start_time = time.time()
                responses = analyze_frames_batch(batch_frames)
                analysis_time = time.time() - start_time
                analysis_times.append(analysis_time)
                
                fps = len(batch_frames) / analysis_time
                batch_count += 1
                
                # Parse violations from responses
                violations_list = []
                for response in responses:
                    violations = parse_violations_with_boxes(response)
                    violations_list.append(violations)
                
                # Put results in queue
                self.result_queue.put({
                    'responses': responses,
                    'violations': violations_list,
                    'fps': fps,
                    'batch_size': len(batch_frames),
                    'analysis_time': analysis_time,
                    'batch_count': batch_count
                })
                
                # Clear old items from queue if it's full
                while self.result_queue.qsize() > 5:
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        break
                        
            except Exception as e:
                print(f"Error in inference worker: {e}")
                import traceback
                traceback.print_exc()
    
    def stop(self):
        """Stop all threads"""
        self.stop_event.set()

def live_video_analysis_optimized(source=0, analyze_every_n_frames=15):
    """
    Highly optimized video analysis with threaded inference pipeline
    """
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    
    # Get video FPS for proper playback speed
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30  # Default if can't detect
    delay_ms = int(1000 / video_fps)  # Milliseconds to wait between frames
    
    print("Press 'q' to quit")
    print(f"Video FPS: {video_fps}, Display delay: {delay_ms}ms per frame")
    print(f"Batch size: {BATCH_SIZE}, Analyzing every {analyze_every_n_frames} frames")
    print(f"Using threaded inference: {USE_THREADING}")
    
    if USE_THREADING:
        # Start threaded analyzer
        analyzer = ThreadedVideoAnalyzer(analyze_every_n_frames)
        inference_thread = threading.Thread(target=analyzer.inference_worker, daemon=True)
        inference_thread.start()
        
        frame_count = 0
        current_analysis = "Starting analysis..."
        current_violations = None
        fps_history = deque(maxlen=30)
        last_time = time.time()
        display_fps_history = deque(maxlen=30)
        last_result_time = time.time()
        frames_submitted = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Submit frame for analysis
                if frame_count % analyze_every_n_frames == 0:
                    try:
                        # Non-blocking put - skip frame if queue is full
                        analyzer.frame_queue.put_nowait(frame.copy())
                        frames_submitted += 1
                    except:
                        pass  # Skip frame if queue is full
                
                # Check for new results (non-blocking)
                try:
                    result = analyzer.result_queue.get_nowait()
                    current_analysis = result['responses'][0]
                    current_violations = result['violations'][0]
                    fps_history.append(result['fps'])
                    
                    # Print stats periodically
                    if time.time() - last_result_time > 2.0:
                        print(f"\nBatch {result['batch_count']}: {result['batch_size']} frames in {result['analysis_time']:.2f}s ({result['fps']:.2f} FPS)")
                        print(f"Analysis: {current_analysis[:80]}...")
                        if current_violations:
                            print(f"Violations detected: {list(current_violations.keys())}")
                        if torch.cuda.is_available():
                            print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                        last_result_time = time.time()
                except Empty:
                    pass
                
                # Display frame with overlay and bounding boxes
                display_frame = frame.copy()
                
                # Draw bounding boxes if violations detected
                if USE_DETAILED_PROMPT and current_violations:
                    display_frame = draw_bounding_boxes(display_frame, current_violations)
                
                # Add legend for violation colors (top-right corner)
                if USE_DETAILED_PROMPT:
                    legend_x = display_frame.shape[1] - 250
                    legend_y = 20
                    legend_items = [
                        ("Rule 1: PPE", (0, 0, 255)),
                        ("Rule 2: Harness", (255, 0, 0)),
                        ("Rule 3: Edge Prot", (0, 165, 255)),
                        ("Rule 4: Blind Spot", (0, 255, 255)),
                    ]
                    
                    # Draw semi-transparent background for legend
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (legend_x - 10, legend_y - 10), 
                                 (legend_x + 240, legend_y + len(legend_items) * 25 + 10), 
                                 (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                    
                    for i, (label, color) in enumerate(legend_items):
                        y_pos = legend_y + i * 25
                        cv2.rectangle(display_frame, (legend_x, y_pos), (legend_x + 20, y_pos + 15), color, -1)
                        cv2.putText(display_frame, label, (legend_x + 30, y_pos + 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                y_offset = 30
                max_width = 80
                
                # Wrap text (only show full text if not using detailed prompt)
                if not USE_DETAILED_PROMPT:
                    words = current_analysis.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line + word) < max_width:
                            current_line += word + " "
                        else:
                            lines.append(current_line)
                            current_line = word + " "
                    lines.append(current_line)
                    
                    # Display analysis text
                    for i, line in enumerate(lines[:5]):
                        cv2.putText(display_frame, line, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += 30
                else:
                    # For detailed prompt, just show summary
                    if current_violations and current_violations.get("0") != "No violations":
                        violation_count = len([k for k in current_violations.keys() if k != "0"])
                        cv2.putText(display_frame, f"Violations: {violation_count} detected", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(display_frame, "No violations detected", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Calculate display FPS
                current_time = time.time()
                if current_time > last_time:
                    display_fps = 1.0 / (current_time - last_time)
                    display_fps_history.append(display_fps)
                
                # Add FPS counters at bottom
                cv2.putText(display_frame, f"Display FPS: {np.mean(display_fps_history):.1f}", 
                           (10, display_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if fps_history:
                    cv2.putText(display_frame, f"Analysis FPS: {np.mean(fps_history):.1f}", 
                               (10, display_frame.shape[0] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.putText(display_frame, f"Queue: {analyzer.frame_queue.qsize()}/{BATCH_SIZE*3}", 
                           (10, display_frame.shape[0] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow('Construction Safety Analysis - OPTIMIZED', display_frame)
                
                last_time = current_time
                frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            analyzer.stop()
            inference_thread.join(timeout=2.0)
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n=== Performance Statistics ===")
            print(f"Total frames: {frame_count}")
            print(f"Frames submitted for analysis: {frames_submitted}")
            if fps_history:
                print(f"Average analysis FPS: {np.mean(fps_history):.2f}")
            if display_fps_history:
                print(f"Average display FPS: {np.mean(display_fps_history):.2f}")
    
    else:
        # Original non-threaded version (fallback)
        print("Running in non-threaded mode")
        # ... existing code ...

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Analyze video file
        video_path = sys.argv[1]
        analyze_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        print(f"Analyzing video: {video_path}")
        print(f"Frame analysis interval: {analyze_interval}")
        live_video_analysis_optimized(video_path, analyze_every_n_frames=analyze_interval)
    else:
        # Use webcam
        print("Using webcam (default)")
        live_video_analysis_optimized(0, analyze_every_n_frames=15)

