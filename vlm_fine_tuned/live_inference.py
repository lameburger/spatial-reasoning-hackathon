import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from peft import PeftModel
from PIL import Image
import cv2
import numpy as np

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
ADAPTER_PATH = "./smolvlm_construction_finetuned"

print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="mps" if torch.backends.mps.is_available() else "auto",
)

print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

def analyze_frame(frame):
    """Analyze a single video frame"""
    # Convert BGR to RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this construction site image and identify any safety violations."}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=[[image]], text=text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    return response

def live_video_analysis(source=0, analyze_every_n_frames=30):
    """
    Analyze live video feed
    source: 0 for webcam, or path to video file
    analyze_every_n_frames: Process every Nth frame (30 = ~1 per second at 30fps)
    """
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    
    frame_count = 0
    current_analysis = "Starting analysis..."
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze frame periodically
        if frame_count % analyze_every_n_frames == 0:
            print(f"Analyzing frame {frame_count}...")
            current_analysis = analyze_frame(frame)
            print(f"Result: {current_analysis}\n")
        
        # Display frame with analysis text
        display_frame = frame.copy()
        
        # Add text overlay (split into multiple lines if too long)
        y_offset = 30
        max_width = 80
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
        
        for line in lines[:5]:  # Show only first 5 lines
            cv2.putText(display_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
        
        cv2.imshow('Construction Safety Analysis', display_frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Analyze video file
        video_path = sys.argv[1]
        print(f"Analyzing video: {video_path}")
        live_video_analysis(video_path, analyze_every_n_frames=30)
    else:
        # Use webcam
        print("Using webcam (default)")
        live_video_analysis(0, analyze_every_n_frames=30)