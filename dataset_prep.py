from datasets import load_dataset
from huggingface_hub import login, HfFolder
import os
import sys


def authenticate_hf():
    """Authenticate with Hugging Face using token from env or cached credentials."""
    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if token:
        login(token=token)
        return True
    else:
        print("Error: No Hugging Face token found. Set HF_TOKEN or run 'huggingface-cli login'")
        return False


def load_dataset_with_auth(dataset_name: str):
    """Load a Hugging Face dataset with proper error handling for gated repositories."""
    try:
        dataset = load_dataset(dataset_name)
        return dataset
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "gated" in error_msg.lower():
            print("Error: Access denied. Enable 'Access to public gated repositories' in token settings.")
        raise


def explore_dataset(dataset):
    """Display dataset information and a sample entry."""
    print(dataset)
    
    sample = dataset["train"][0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image caption: {sample['image_caption']}")
    print(f"Violations: rule_1={sample['rule_1_violation']}, rule_2={sample['rule_2_violation']}")


def main():
    """Main function to prepare and explore the dataset."""
    if not authenticate_hf():
        sys.exit(1)
    
    dataset = load_dataset_with_auth("LouisChen15/ConstructionSite")
    explore_dataset(dataset)


if __name__ == "__main__":
    main()
