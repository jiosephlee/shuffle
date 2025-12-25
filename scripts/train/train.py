import os
import sys
import argparse
import logging
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

try:
    import liger_kernel
    is_liger_available = True
except ImportError:
    is_liger_available = False

# Ensure scripts path is in sys.path
sys.path.append('../../')
sys.path.append('../')

from utils.data_utils import (
    load_medqa_text_data,
    load_heldout_val_data,
    load_mcqa_data,
    get_fewshot_examples,
    create_mcqa_dataset
)
from scripts.train.callbacks import (
    HeldOutEvalCallback,
    MedQAEvalCallback,
    AttentionEntropyCallback
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Olmo on Shuffle Experiments")
    parser.add_argument("--condition", type=str, required=True, 
                        choices=['ordered', 'shuffled_para', 'shuffled_sent', 'shuffled_word'],
                        help="Shuffling condition for training data")
    # Calculate default data_dir relative to this script
    # script is in scripts/train/train.py -> ../../data/medqa is the target
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    default_data_dir = os.path.join(base_dir, "data", "medqa")

    parser.add_argument("--model_id", type=str, default="allenai/Olmo-3-1025-7B", 
                        help="Model ID to use (default: allenai/Olmo-3-1025-7B)")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Path to data directory")
    # Output dir is fixed to "output"
    parser.add_argument("--test", action="store_true", help="Run in test mode (no wandb, 1 step)")
    parser.add_argument("--full_finetune", action="store_true", default=True, help="Use full finetuning")
    parser.add_argument("--turn_on_attention_tracking", action="store_true", help="Enable attention entropy tracking")
    
    # New Hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--packing", action="store_true", help="Enable packing (and padding_free)")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation (e.g. flash_attention_2)")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing")
    parser.add_argument("--use_liger_kernel", action=argparse.BooleanOptionalAction, default=True, help="Enable Liger Kenel")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup WandB
    if not args.test:
        wandb.init(project="shuffle-effect", name=f"olmo-7b-{args.condition}", config=vars(args))
    else:
        logger.info("Test mode: WandB disabled.")

    # 2. Load Model & Tokenizer
    if args.test:
        logger.info("Test mode detected. Loading Qwen/Qwen2.5-0.5B...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    else:
        logger.info(f"Loading model: {args.model_id}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id, 
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
                trust_remote_code=True,
                attn_implementation=args.attn_implementation
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Data
    logger.info(f"Loading data for condition: {args.condition}")
    
    # Train Data (List of strings, one per book)
    train_texts = load_medqa_text_data(args.data_dir, args.condition)
    
    if args.test:
        train_texts = [t for t in train_texts]
    
    # Convert to Dataset expected by SFTTrainer
    train_dataset = Dataset.from_dict({"text": train_texts})
    
    # Held-out Data (List of strings)
    heldout_texts = load_heldout_val_data(args.data_dir)
    
    # MCQA Data
    test_jsonl = os.path.join(args.data_dir, "test.jsonl")
    train_jsonl = os.path.join(args.data_dir, "train.jsonl")
    
    mcqa_data = load_mcqa_data(test_jsonl)
    fewshot_ex = get_fewshot_examples(train_jsonl, num_shots=4)
    
    
    zero_shot_ds = create_mcqa_dataset(mcqa_data, fewshot_examples=None)
    few_shot_ds = create_mcqa_dataset(mcqa_data, fewshot_examples=fewshot_ex)
    
    # --- SMOKE TEST: MCQA PROMPTS ---
    print("\n" + "="*50)
    print("SMOKE TEST: MCQA ZERO-SHOT PROMPT")
    print("="*50)
    if len(zero_shot_ds) > 0:
        print(f"--- Prompt Start ---\n{zero_shot_ds[0]['prompt']}\n--- Prompt End ---")
        print(f"Target Label: {zero_shot_ds[0]['label']}")
    else:
        print("[WARNING] No Zero-Shot data found!")

    print("\n" + "="*50)
    print("SMOKE TEST: MCQA FEW-SHOT PROMPT")
    print("="*50)
    if len(few_shot_ds) > 0:
        print(f"--- Prompt Start ---\n{few_shot_ds[0]['prompt']}\n--- Prompt End ---")
        print(f"Target Label: {few_shot_ds[0]['label']}")
    else:
        print("[WARNING] No Few-Shot data found!")
    print("="*50 + "\n")
    # -------------------------------
    
    # 4. Setup Callbacks
    callbacks = []
    
    # Heldout Eval
    callbacks.append(HeldOutEvalCallback(heldout_texts, tokenizer, log_freq=10))
    
    # MCQA Eval
    callbacks.append(MedQAEvalCallback(zero_shot_ds, few_shot_ds, tokenizer, log_freq=20))
    
    # Attention Entropy
    if args.turn_on_attention_tracking:
        callbacks.append(AttentionEntropyCallback(tokenizer, log_freq=20))
    
    # 5. Training Args (SFTConfig)
    sft_config = SFTConfig(
        dataset_text_field="text",
        output_dir=os.path.join("output", args.condition),
        
        # Hyperparams
        per_device_train_batch_size=args.per_device_train_batch_size if not args.test else 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps if not args.test else 1,
        num_train_epochs=args.num_train_epochs if not args.test else 0.01,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        
        # Packing
        packing=args.packing,
        packing_strategy="wrapped",
        # padding_free=args.packing, 
        max_length=4096, # Context length
        sequential_sampling = True,
        # Strategies
        gradient_checkpointing=args.gradient_checkpointing,
        use_liger_kernel=args.use_liger_kernel and is_liger_available,
        logging_steps=1,
        save_strategy="no", # Always no as requested
        report_to="wandb" if not args.test else "none",
        
        # Dtype
        fp16=False,
        bf16=True if torch.cuda.is_available() and not args.test else False,
        
        dataloader_num_workers=0
    )
    
    if args.test:
        sft_config.max_steps = 5

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
        processing_class=tokenizer,
        callbacks=callbacks
    )

    # --- SMOKE TEST: INSPECT DATALOADER ---
    print("\n" + "="*50)
    print("SMOKE TEST: INSPECTING DATALOADER (First Batch)")
    print("="*50)
    try:
        dataloader = trainer.get_train_dataloader()
        # Inspect first batch
        first_batch = next(iter(dataloader))
        
        # Check input_ids shape
        input_ids = first_batch['input_ids']
        print(f"Batch Input IDs Shape: {input_ids.shape}")
        
        # Decode texts
        decoded_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        
        for i, text in enumerate(decoded_texts):
            print(f"\n--- Batch Item {i} (Length: {len(input_ids[i])} tokens) ---")
            print(text)
            print("--- End Batch Item ---")
            
    except Exception as e:
        logger.error(f"Failed to inspect dataloader: {e}")
    print("="*50 + "\n")
    # --------------------------------------
    
    # 6. Train
    if args.test:
        logger.info("Test mode: Skipping actual training loop logic (or running dry run if needed).")
        #trainer.train() 
        pass 
    else:
        logger.info("Starting Full Training...")
        trainer.train()
        # No save_model since save_strategy="no", but usually we want final model?
        # User said "set save_strategy always to no", implies maybe no intermediate checkpoints?
        # But we should save the final model probably.
        trainer.save_model()

if __name__ == "__main__":
    main()
