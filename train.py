import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATASET_ID = "LouisChen15/ConstructionSite"
OUTPUT_DIR = "./smolvlm_construction_finetuned"
MAX_STEPS = 1000
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4

print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load model in fp16 without quantization (RTX 5070 has enough VRAM)
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Prepare model for training with LoRA
model.gradient_checkpointing_enable()

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading dataset...")
dataset = load_dataset(DATASET_ID, split="train")

def format_example(example):
    """Convert dataset example to training format"""
    caption = example["image_caption"]
    
    # Build violation info if exists
    violations = []
    for rule_num in range(1, 5):
        rule_key = f"rule_{rule_num}_violation"
        if example.get(rule_key) and example[rule_key].get("reason"):
            violations.append(f"Rule {rule_num} Violation: {example[rule_key]['reason']}")
    
    violation_text = " ".join(violations) if violations else "No safety violations detected."
    
    # Create conversation format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this construction site image and identify any safety violations."}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"Description: {caption}\n\nSafety Assessment: {violation_text}"}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
    
    return {
        "image": example["image"],
        "text": prompt,
    }

print("Formatting dataset...")
train_dataset = dataset.map(format_example, remove_columns=dataset.column_names)

def collate_fn(examples):
    """Collate function for batching"""
    images = [example["image"].convert("RGB") for example in examples]
    texts = [example["text"] for example in examples]
    
    batch = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    
    return batch

# Training config
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_steps=250,
    save_total_limit=2,
    bf16=False,
    fp16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    remove_unused_columns=False,
    dataset_text_field="text",
    dataloader_pin_memory=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"Training complete! Model saved to {OUTPUT_DIR}")

