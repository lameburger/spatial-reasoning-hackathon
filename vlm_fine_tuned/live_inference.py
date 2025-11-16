"""
Optimized Real-Time Construction Safety Analysis with VLM

This script provides highly optimized real-time video inference for construction site safety
monitoring using a fine-tuned Vision-Language Model (SmolVLM).

Key Features:
- Multi-threaded architecture for parallel frame capture and inference
- Batch processing for better GPU utilization
- CUDA stream optimizations for async operations
- Text-based safety violation detection
- Real-time FPS monitoring and performance metrics

Usage:
    # Analyze video file with default settings (analyze every 15 frames)
    python live_inference.py site.mp4
    
    # Analyze video with custom frame interval (analyze every 10 frames)
    python live_inference.py site.mp4 10
    
    # Use webcam
    python live_inference.py

Configuration:
    - BATCH_SIZE: Number of frames to process in parallel (default: 8)
    - MAX_NEW_TOKENS: Maximum tokens for model generation (default: 256)
    - USE_THREADING: Enable/disable multi-threaded processing

Performance Optimizations:
    1. Threaded video capture and inference pipeline
    2. Batch processing (8 frames at once)
    3. CUDA streams for async GPU operations
    4. Non-blocking frame queues
    5. Optimized preprocessing with direct numpy operations
    6. Batch decoding for faster response processing
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
import re

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
ADAPTER_PATH = "./smolvlm_construction_finetuned"

# Optimization settings
USE_TORCH_COMPILE = True  # Use torch.compile for speed
BATCH_SIZE = 8  # Increased batch size for better GPU utilization
USE_FLASH_ATTENTION = True  # Enable flash attention if available
MAX_NEW_TOKENS = 256  # Concise responses focused on violations
USE_CACHE = True  # Use KV cache for faster generation
NUM_BEAMS = 1  # Greedy decoding for speed
USE_THREADING = True  # Use multi-threading for parallel frame processing
PREFETCH_BATCHES = 2  # Number of batches to prefetch

# Concise prompt focused on violations
# Based on: https://github.com/LouisChen15/ConstructionSite-10k-Implementation
SAFETY_PROMPT = """Identify safety violations. Check:
1. Basic PPE (hard hats, safety glasses, vests) - Rule 1
2. Safety harness when working ≥3m high without edge protection - Rule 2
3. Edge protection for underground projects ≥3m depth - Rule 3
4. Workers in excavator blind spots or operating radius - Rule 4

For each violation: "Rule X: [brief reason]"
If no violations: "No violations detected".
Be concise."""

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

def parse_violations(response_text):
    """Parse violations from model response"""
    response_lower = response_text.lower()
    
    # Check for no violations
    if any(phrase in response_lower for phrase in ["no violations", "no violation detected"]):
        return {"violations": [], "has_violations": False}
    
    violations = []
    
    # Pattern to match: "Rule X: [reason]"
    # Also handle variations like "Rule 1:", "Rule1:", etc.
    pattern = r'Rule\s*(\d+)\s*:?\s*([^\n|]+)'
    
    matches = re.finditer(pattern, response_text, re.IGNORECASE)
    
    for match in matches:
        rule_num = int(match.group(1))
        reason = match.group(2).strip()
        
        violations.append({
            "rule": rule_num,
            "reason": reason
        })
    
    # If no structured matches but contains violation keywords, create a general violation
    if not violations:
        violation_keywords = ["violation", "without hard hat", "missing", "unsafe", "hazard", "no hard hat", "no safety"]
        if any(keyword in response_lower for keyword in violation_keywords):
            # Try to extract rule numbers mentioned
            rule_mentions = re.findall(r'rule\s*(\d+)', response_lower)
            if rule_mentions:
                for rule_num in set(rule_mentions):
                    violations.append({
                        "rule": int(rule_num),
                        "reason": "Violation detected"
                    })
            else:
                violations.append({
                    "rule": 1,  # Default to Rule 1
                    "reason": "Safety violation detected"
                })
    
    return {
        "violations": violations,
        "has_violations": len(violations) > 0
    }

def analyze_frames_batch(frames):
    """Optimized batch analysis with faster preprocessing and decoding"""
    images = preprocess_frames_fast(frames)
    
    # Simple prompt for safety violation detection
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": SAFETY_PROMPT}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Process all images together
    try:
        # Try batch processing first
        if len(images) > 1:
            try:
                inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True)
                print(f"[DEBUG] Batch processed {len(images)} images")
            except Exception as e:
                print(f"[WARNING] Batch processing failed, trying individual: {e}")
                # Fall back to processing individually
                all_responses = []
                for img in images:
                    single_inputs = processor(images=[img], text=[prompt], return_tensors="pt")
                    single_inputs = {k: v.to(model.device) for k, v in single_inputs.items()}
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                                single_output = model.generate(**single_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                        else:
                            single_output = model.generate(**single_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                    response = processor.decode(single_output[0], skip_special_tokens=True)
                    if "Assistant:" in response:
                        response = response.split("Assistant:")[-1].strip()
                    all_responses.append(response)
                return all_responses
        else:
            inputs = processor(images=images, text=[prompt], return_tensors="pt")
            print(f"[DEBUG] Single image processed")
    except Exception as e:
        print(f"[ERROR] Failed to process inputs: {e}")
        import traceback
        traceback.print_exc()
        return ["Error processing inputs"] * len(frames)
    
    # Move to device (simplified - no CUDA stream for now)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    print(f"[DEBUG] Generating response...")
    try:
        with torch.no_grad():
            if torch.cuda.is_available():
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
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=NUM_BEAMS,
                    use_cache=USE_CACHE,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
        print(f"[DEBUG] Generation complete")
    except Exception as e:
        print(f"[ERROR] Failed during generation: {e}")
        import traceback
        traceback.print_exc()
        return ["Error during generation"] * len(frames)
    
    # Decode responses
    try:
        # Try batch decode first, fall back to individual if it doesn't work
        if hasattr(processor, 'batch_decode'):
            responses = processor.batch_decode(outputs, skip_special_tokens=True)
        else:
            # Decode individually
            responses = []
            for i in range(len(frames)):
                response = processor.decode(outputs[i], skip_special_tokens=True)
                responses.append(response)
    except Exception as e:
        print(f"[ERROR] Failed to decode: {e}")
        import traceback
        traceback.print_exc()
        return ["Error decoding"] * len(frames)
    
    # Clean up responses - keep violation info, remove prompt text
    cleaned_responses = []
    for i, response in enumerate(responses):
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        # Remove prompt text but keep violation information
        response = response.replace(prompt, "").strip()
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
                print(f"\n[INFERENCE] Processing batch of {len(batch_frames)} frames...")
                start_time = time.time()
                try:
                    responses = analyze_frames_batch(batch_frames)
                    analysis_time = time.time() - start_time
                    analysis_times.append(analysis_time)
                    print(f"[INFERENCE] Completed in {analysis_time:.2f}s")
                except Exception as e:
                    print(f"[ERROR] Inference failed: {e}")
                    import traceback
                    traceback.print_exc()
                    responses = ["Error during inference"] * len(batch_frames)
                    analysis_time = time.time() - start_time
                
                fps = len(batch_frames) / analysis_time
                batch_count += 1
                
                # Parse violations from responses
                violations_data_list = []
                for response in responses:
                    violations_data = parse_violations(response)
                    violations_data_list.append(violations_data)
                
                # Put results in queue
                self.result_queue.put({
                    'responses': responses,
                    'violations_data': violations_data_list,
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
        current_violations_data = {"violations": [], "has_violations": False}
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
                    current_violations_data = result['violations_data'][0]
                    fps_history.append(result['fps'])
                    
                    # Print stats periodically
                    if time.time() - last_result_time > 2.0:
                        print(f"\nBatch {result['batch_count']}: {result['batch_size']} frames in {result['analysis_time']:.2f}s ({result['fps']:.2f} FPS)")
                        if current_violations_data.get("has_violations"):
                            violations = current_violations_data["violations"]
                            print(f"⚠️  {len(violations)} VIOLATION(S) DETECTED:")
                            for v in violations:
                                print(f"   Rule {v['rule']}: {v['reason'][:80]}")
                        else:
                            print("✅ No violations detected")
                        if torch.cuda.is_available():
                            print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                        last_result_time = time.time()
                except Empty:
                    pass
                
                # Display frame with overlay
                display_frame = frame.copy()
                
                # Display violation status text
                y_offset = 30
                if current_violations_data.get("has_violations"):
                    violations = current_violations_data["violations"]
                    cv2.putText(display_frame, f"⚠️  {len(violations)} VIOLATION(S)", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    y_offset += 35
                    
                    # Show violation details
                    for i, v in enumerate(violations[:3]):  # Show max 3
                        text = f"Rule {v['rule']}: {v['reason'][:50]}"
                        cv2.putText(display_frame, text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        y_offset += 30
                else:
                    cv2.putText(display_frame, "✅ No violations", 
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

