import torch
from transformers import Idefics3Processor, Idefics3ForConditionalGeneration
from peft import PeftModel
from PIL import Image
import sys

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
ADAPTER_PATH = "./smolvlm_construction_finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model and processor... (using device: {DEVICE})")
processor = Idefics3Processor.from_pretrained(MODEL_ID)
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

def analyze_image(image_path, prompt="Describe this construction site image and identify any safety violations."):
    """Analyze a construction site image"""
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=[image], text=text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = analyze_image(image_path)
    print("\n" + "="*50)
    print("ANALYSIS RESULT:")
    print("="*50)
    print(result)
    print("="*50 + "\n")

