import os
import glob
import json
import logging
from typing import List, Dict, Any
from datasets import Dataset

logger = logging.getLogger(__name__)

def load_medqa_text_data(data_dir: str, condition: str) -> List[str]:
    """
    Loads text data for training based on the condition.
    Returns a list of strings, where each string is the content of one book.
    """
    target_dir = os.path.join(data_dir, condition)
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Data directory for condition '{condition}' not found at {target_dir}")
    
    txt_files = glob.glob(os.path.join(target_dir, "*.txt"))
    if not txt_files:
         raise FileNotFoundError(f"No text files found in {target_dir}")

    logger.info(f"Loading {len(txt_files)} training files from {target_dir}")
    texts = []
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
            
    return texts

def load_heldout_val_data(data_dir: str) -> List[str]:
    """
    Loads the held-out validation texts.
    Returns a list of strings, one per book.
    """
    val_path = os.path.join(data_dir, "val")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Held-out validation directory not found at {val_path}")
    
    txt_files = glob.glob(os.path.join(val_path, "*.txt"))
    if not txt_files:
         raise FileNotFoundError(f"No text files found in {val_path}")
         
    texts = []
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
            
    return texts

def load_mcqa_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads MCQA data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {file_path}")
    return data

def get_fewshot_examples(train_jsonl_path: str, num_shots: int = 4) -> List[Dict[str, Any]]:
    """
    Loads the first `num_shots` examples from the training JSONL file.
    """
    if not os.path.exists(train_jsonl_path):
        raise FileNotFoundError(f"Training JSONL file not found at {train_jsonl_path}")
    
    examples = []
    with open(train_jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_shots:
                break
            if line.strip():
                examples.append(json.loads(line))
    return examples

def format_mcqa_prompt(example: Dict[str, Any], include_answer: bool = False) -> str:
    """
    Formats a single MCQA example into a prompt.
    """
    prompt = f"Question: {example['question']}\n"
    for opt, text in example['options'].items():
        prompt += f"{opt}: {text}\n"
    prompt += "Answer:"
    
    if include_answer:
        prompt += f" {example['answer_idx']}\n\n"
    
    return prompt

def create_mcqa_dataset(data: List[Dict[str, Any]], fewshot_examples: List[Dict[str, Any]] = None) -> Dataset:
    """
    Creates a HuggingFace Dataset for MCQA evaluation.
    """
    formatted_data = []
    
    fewshot_prefix = ""
    if fewshot_examples:
        for ex in fewshot_examples:
            fewshot_prefix += format_mcqa_prompt(ex, include_answer=True)
            
    for item in data:
        prompt = fewshot_prefix + format_mcqa_prompt(item, include_answer=False)
        formatted_data.append({
            "prompt": prompt,
            "label": item['answer_idx'],
            "options": item['options'],
            "raw_example": item
        })
        
    return Dataset.from_list(formatted_data)
