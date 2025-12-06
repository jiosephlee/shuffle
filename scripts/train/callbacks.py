import os
import torch
import logging
import math
import wandb
import numpy as np
from transformers import TrainerCallback, TrainerState, TrainerControl
from torch.utils.data import DataLoader

# Handle imports based on execution path
try:
    from scripts.utils.eval_utils import get_constrained_predictions, calculate_attention_entropy, log_debug_info
    from scripts.utils.data_utils import load_heldout_val_data, create_mcqa_dataset
except ImportError:
    # Try relative import if run from scripts/train
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from scripts.utils.eval_utils import get_constrained_predictions, calculate_attention_entropy, log_debug_info
    from scripts.utils.data_utils import load_heldout_val_data, create_mcqa_dataset

logger = logging.getLogger(__name__)

class HeldOutEvalCallback(TrainerCallback):
    def __init__(self, eval_datasets, tokenizer, log_freq=500):
        """
        eval_datasets: List of strings, where each string is the held-out text of a book.
        """
        self.tokenizer = tokenizer
        self.eval_datasets = eval_datasets 
        self.log_freq = log_freq

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % self.log_freq == 0 and state.global_step > 0:
            logger.info("Running Held-out Evaluation...")
            model.eval()
            
            total_ppl = 0
            n_books = len(self.eval_datasets)
            
            # Context length for sliding window
            context_length = 2048
            stride = context_length

            for i, text in enumerate(self.eval_datasets):
                # Tokenize
                tokens = self.tokenizer(text, return_tensors='pt', truncation=False, padding=False)['input_ids'][0]
                
                nlls = []
                with torch.inference_mode():
                    for j in range(0, tokens.size(-1), stride):
                        begin_loc = max(j + stride - context_length, 0)
                        end_loc = min(j + stride, tokens.size(-1))
                        trg_len = end_loc - j
                        
                        input_ids = tokens[begin_loc:end_loc].to(model.device).unsqueeze(0)
                        target_ids = input_ids.clone()
                        target_ids[:, :-trg_len] = -100
                        
                        if input_ids.size(1) == 0: continue

                        outputs = model(input_ids, labels=target_ids)
                        neg_log_likelihood = outputs.loss * trg_len
                        nlls.append(neg_log_likelihood)
                
                if nlls:
                    ppl = torch.exp(torch.stack(nlls).sum() / end_loc).item()
                else:
                    ppl = 0.0
                    
                total_ppl += ppl
                # logger.info(f"Book {i+1} Perplexity: {ppl:.2f}")

            avg_ppl = total_ppl / n_books if n_books > 0 else 0
            logger.info(f"Step {state.global_step}: Average Held-out Perplexity = {avg_ppl:.4f}")
            
            if wandb.run:
                wandb.log({"eval/heldout_perplexity": avg_ppl}, step=state.global_step)
            
            model.train()

class MedQAEvalCallback(TrainerCallback):
    def __init__(self, zero_shot_ds, few_shot_ds, tokenizer, log_freq=500):
        self.zero_shot_ds = zero_shot_ds
        self.few_shot_ds = few_shot_ds
        self.tokenizer = tokenizer
        self.log_freq = log_freq
        self.logged_prompts = False

    def evaluate_ds(self, ds, model, name, state):
        correct = 0
        total = 0
        
        for i, example in enumerate(ds):
            prompt = example['prompt']
            label_idx = example['label'] # 'A', 'B', 'C', 'D'
            
            inputs = self.tokenizer(prompt, return_tensors='pt').to(model.device)
            
            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :] # Last token logits
            
            # Constrained prediction
            preds = get_constrained_predictions(logits.unsqueeze(0), self.tokenizer, model.device)
            pred_char, prob_dict = preds[0]
            
            is_correct = (pred_char == label_idx)
            if is_correct:
                correct += 1
            total += 1
            
            # Log debug info for the first example
            if i == 0 and not self.logged_prompts:
                log_debug_info(prompt, logits, self.tokenizer, f"{name} (Step {state.global_step})", logger)

        return correct / total if total > 0 else 0

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % self.log_freq == 0 and state.global_step > 0:
            logger.info("Running MedQA Evaluation...")
            model.eval()
            
            zs_acc = self.evaluate_ds(self.zero_shot_ds, model, "Zero-Shot MCQA", state)
            fs_acc = self.evaluate_ds(self.few_shot_ds, model, "Few-Shot MCQA", state)
            
            if not self.logged_prompts:
                self.logged_prompts = True
                
            logger.info(f"Step {state.global_step}: Zero-Shot Acc = {zs_acc:.4f}, Few-Shot Acc = {fs_acc:.4f}")
            
            if wandb.run:
                wandb.log({
                    "eval/medqa_zeroshot_acc": zs_acc,
                    "eval/medqa_fewshot_acc": fs_acc
                }, step=state.global_step)
            
            model.train()

class AttentionEntropyCallback(TrainerCallback):
    def __init__(self, tokenizer, log_freq=100):
        self.tokenizer = tokenizer
        self.log_freq = log_freq
        # Dummy input for checking attention
        self.dummy_input = tokenizer("Medical reasoning requires attention.", return_tensors='pt')

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % self.log_freq == 0 and state.global_step > 0:
            model.eval()
            inputs = self.dummy_input.to(model.device)
            
            with torch.inference_mode():
                # Ensure output_attentions is True
                outputs = model(**inputs, output_attentions=True)
                attentions = outputs.attentions
            
            entropy_stats = calculate_attention_entropy(attentions)
            
            if wandb.run:
                wandb.log(entropy_stats, step=state.global_step)
            
            model.train()
