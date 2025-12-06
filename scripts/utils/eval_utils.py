import torch
import torch.nn.functional as F
import logging
import numpy as np
from typing import List, Dict, Tuple
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def get_constrained_predictions(
    logits: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    device: torch.device
) -> Tuple[str, Dict[str, float]]:
    """
    Given logits for the last token, returns the predicted option (A, B, C, D) 
    and the normalized probabilities for each option.
    """
    # Define target tokens
    # Note: Depending on tokenizer, " A" might be different from "A". 
    # Usually in MCQA prompts "Answer:" ends with space, so we expect " A", " B", etc.
    # We will check both " A" and "A" and take the max, or just " A" if we are consistent.
    # Let's assume standard space-prefixed tokens if applicable.
    
    options = ["A", "B", "C", "D"]
    option_ids = []
    
    # Try with leading space first (common in Llama/Olmo tokenizers after "Answer:")
    # But check if tokenizer adds space automatically or not. 
    # Safest is to encode " A" and take the last token id.
    
    for opt in options:
        # Encoder " A" or "A" depending on your prompt format. 
        # format_mcqa_prompt ends with "Answer:", so next token is likely " A" (space A).
        # We can try to support both simple "A" and " A" by taking the one with higher logit if ambiguous, 
        # but standard is usually just one. Let's try " A".
        ids = tokenizer.encode(" " + opt, add_special_tokens=False)
        if len(ids) == 0:
             # Fallback to no space
             ids = tokenizer.encode(opt, add_special_tokens=False)
        option_ids.append(ids[-1]) # Take the last token id (e.g. for " A")
        
    option_ids_tensor = torch.tensor(option_ids, device=device)
    
    # Extract logits for just these options
    # logits shape: [batch_size, vocab_size] -> we assume batch_size=1 usually for simple loop, or batched.
    # If batched, we need to handle it. Assuming batch_size=1 for now based on typical eval loops or simple gathering.
    # Let's support batched input [batch_size, vocab_size].
    
    target_logits = logits[:, option_ids_tensor] # [batch_size, 4]
    
    # Softmax over these 4 options
    probs = F.softmax(target_logits, dim=-1)
    
    # Get predictions
    pred_indices = torch.argmax(probs, dim=-1) # [batch_size]
    
    results = []
    
    # Convert back to list of (pred_char, prob_dict)
    for i in range(logits.size(0)):
        pred_char = options[pred_indices[i].item()]
        prob_dict = {opt: probs[i, j].item() for j, opt in enumerate(options)}
        results.append((pred_char, prob_dict))
        
    return results

def calculate_attention_entropy(attentions: Tuple[torch.Tensor]) -> Dict[str, float]:
    """
    Calculates entropy of the attention distribution for the last token 
    at the 1st, middle, and last layers.
    Args:
        attentions: Tuple of tensors (one per layer), each of shape (batch_size, num_heads, seq_len, seq_len)
    """
    num_layers = len(attentions)
    indices = [0, num_layers // 2, num_layers - 1]
    layer_names = ["first", "middle", "last"]
    
    entropy_stats = {}
    
    for idx, name in zip(indices, layer_names):
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        attn_map = attentions[idx]
        
        # We care about the attention of the *last token* (query) attending to all previous tokens (keys)
        # Shape: [batch_size, num_heads, 1, seq_len]
        # Or just take the last row: attn_map[..., -1, :]
        last_token_attn = attn_map[..., -1, :] # [batch_size, num_heads, seq_len]
        
        # Avoid log(0)
        epsilon = 1e-9
        
        # Entropy = - sum(p * log(p))
        # Sum over seq_len dimension
        entropy = -torch.sum(last_token_attn * torch.log(last_token_attn + epsilon), dim=-1) # [batch_size, num_heads]
        
        # Average over heads and batch
        avg_entropy = entropy.mean().item()
        
        entropy_stats[f"attn_entropy_{name}_layer"] = avg_entropy
        
    return entropy_stats

def log_debug_info(
    prompt: str,
    logits: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    step_name: str,
    logger: logging.Logger,
    top_k: int = 10
):
    """
    Logs the prompt and top-k predicted tokens/probs for debugging.
    """
    logger.info(f"\n--- DEBUG INFO: {step_name} ---")
    logger.info(f"PROMPT (truncated): {prompt[:500]}..." if len(prompt) > 500 else f"PROMPT: {prompt}")
    
    # Logits shape: [vocab_size] (assuming single example)
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)
    
    log_msg = "Top predictions:\n"
    for i in range(top_k):
        token_id = top_indices[i].item()
        token_prob = top_probs[i].item()
        token_str = tokenizer.decode([token_id])
        log_msg += f"  {i+1}. '{token_str}' (ID: {token_id}): {token_prob:.4f}\n"
        
    logger.info(log_msg)
    logger.info("--------------------------------")
