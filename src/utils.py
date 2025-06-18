import math
import re
import torch as th
import torch.nn.functional as F
import os
from pathlib import Path

from torch import nn
from transformers import CLIPProcessor, CLIPModel


def get_gpu():
    if th.cuda.is_available():
        return th.device("cuda")
    elif th.backends.mps.is_available():
        return th.device("mps")
    else:
        return th.device("cpu")
    

def get_clip_model(device=None, cache_dir="clip_model_cache"):
    """Get or initialize CLIP model and processor with disk caching"""
    if device is None:
        device = get_gpu()
    
    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True, parents=True)
    
    model_path = cache_path / "clip_model"
    processor_path = cache_path / "clip_processor"
    
    # Check if model is already cached on disk
    if model_path.exists() and processor_path.exists():
        try:
            # Load from disk cache
            clip_model = CLIPModel.from_pretrained(str(model_path))
            clip_processor = CLIPProcessor.from_pretrained(str(processor_path))
            
            # Move model to correct device
            clip_model = clip_model.to(device)
            clip_model.eval()
            
            print(f"CLIP model loaded from cache: {cache_dir}")
            return clip_model, clip_processor
        except Exception as e:
            print(f"Error loading cached CLIP model: {e}. Downloading fresh model...")
    
    # Download and cache if not available
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    
    # Save to disk for future use
    try:
        clip_model.save_pretrained(str(model_path))
        clip_processor.save_pretrained(str(processor_path))
        print(f"CLIP model cached to: {cache_dir}")
    except Exception as e:
        print(f"Failed to cache CLIP model: {e}")
    
    return clip_model, clip_processor


def calculate_clip_weights(prompts, device=None, default_weight=7.5, cache_dir="clip_model_cache"):
    """
    Calculate prompt weights using CLIP embeddings to determine semantic importance.
    
    Args:
        prompts: List of prompt strings
        device: Computation device
        default_weight: Base weight value
        cache_dir: Directory to cache CLIP models
    
    Returns:
        List of calculated weights
    """
    if not prompts:
        return []
        
    if device is None:
        device = get_gpu()
        
    # Get CLIP model and processor with disk caching
    clip_model, clip_processor = get_clip_model(device, cache_dir=cache_dir)
    
    with th.no_grad():
        # Encode all prompts with CLIP
        inputs = clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_features = clip_model.get_text_features(**inputs)
        
        # Normalize embeddings
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Calculate pairwise similarity matrix
        similarity_matrix = th.matmul(text_features, text_features.transpose(0, 1))
        
        # Calculate distinctiveness score for each prompt (lower similarity = more distinct)
        # We exclude self-similarity (diagonal elements)
        n = len(prompts)
        distinctiveness = []
        
        for i in range(n):
            # Get all similarities except self-similarity
            similarities = [similarity_matrix[i, j].item() for j in range(n) if i != j]
            
            # If there are other prompts to compare with
            if similarities:
                # Lower average similarity means more distinctive prompt
                avg_similarity = sum(similarities) / len(similarities)
                # Convert to distinctiveness (inverse of similarity)
                distinct_score = 1.0 - avg_similarity
            else:
                distinct_score = 1.0  # Single prompt case
                
            distinctiveness.append(distinct_score)
        
        # Calculate attention importance from distinctiveness
        total_distinctiveness = sum(distinctiveness)
        importance_weights = []
        
        # Calculate weights based on distinctiveness and normalize to default weight
        if total_distinctiveness > 0:
            base_scale = default_weight * 0.8  # Allow range above and below default
            
            for score in distinctiveness:
                # Scale importance to reasonable range around default_weight
                normalized_score = score / (total_distinctiveness / len(distinctiveness))
                weight = base_scale + (base_scale * 0.5 * (normalized_score - 1))
                importance_weights.append(max(1.0, min(15.0, weight)))  # Clamp to reasonable range
        else:
            importance_weights = [default_weight] * len(prompts)
            
        return importance_weights


def prepare_prompt(prompt_string, default_weight=7.5, use_clip=True, device=None, cache_dir="clip_model_cache"):
    """
    Enhanced prompt preparation with automatic CLIP-based weight calculation.
    
    Formats:
    - Simple: "A cat and a dog"
    - Weighted: "A cat:9.5 and a dog:5.2"
    - Negated: "not a cat and a dog"
    
    Args:
        prompt_string: The input prompt string
        default_weight: Default weight if not using CLIP or for explicitly weighted prompts
        use_clip: Whether to use CLIP for automatic weight calculation
        device: Computation device
        
    Returns:
        - List of processed prompts
        - List of corresponding weights
    """
    # Convert to lowercase for consistency
    prompt_string = prompt_string.lower()
    
    # Replace conjunctions with separators
    prompt_string = re.sub(r'\s+and\s+', '|', prompt_string)
    
    # Split by separator
    prompts = [x.strip() for x in prompt_string.split('|')]
    
    # Extract weights explicitly specified in prompts and handle negation
    weight_pattern = re.compile(r'(.*?)(?::(\d+(?:\.\d+)?))?$')
    
    processed_prompts = []
    explicit_weights = []
    has_explicit_weights = False
    negation_flags = []
    
    # First pass: process explicit weights and negations
    for prompt in prompts:
        # Check if this is a negated prompt
        is_negated = prompt.startswith("not ")
        negation_flags.append(is_negated)
        
        # Extract any explicitly specified weight
        match = weight_pattern.match(prompt)
        if match:
            clean_prompt, explicit_weight = match.groups()
            clean_prompt = clean_prompt.strip()
            
            # If negated, remove "not " prefix
            if is_negated:
                clean_prompt = clean_prompt.replace("not ", "", 1).strip()
            
            processed_prompts.append(clean_prompt)
            
            # Handle explicit weight if provided
            if explicit_weight:
                has_explicit_weights = True
                weight = float(explicit_weight)
                if is_negated:
                    weight = -abs(weight)
                explicit_weights.append(weight)
            else:
                # Placeholder for now
                explicit_weights.append(None)
        else:
            # No explicit weight
            clean_prompt = prompt
            if is_negated:
                clean_prompt = clean_prompt.replace("not ", "", 1).strip()
            
            processed_prompts.append(clean_prompt)
            explicit_weights.append(None)
    
    # Second pass: calculate weights
    final_weights = []
    
    # If using CLIP and no explicit weights are provided
    if use_clip and not all(explicit_weights):
        try:
            # Calculate weights using CLIP
            clip_weights = calculate_clip_weights(processed_prompts, device, default_weight, cache_dir=cache_dir)
            
            # Apply negation and merge with any explicit weights
            for i, (explicit_w, clip_w, is_neg) in enumerate(zip(explicit_weights, clip_weights, negation_flags)):
                if explicit_w is not None:
                    # Explicit weight takes precedence
                    final_weights.append(explicit_w)
                else:
                    # Use CLIP-calculated weight
                    w = clip_w
                    if is_neg:
                        w = -abs(w)
                    final_weights.append(w)
        except Exception as e:
            print(f"CLIP weight calculation failed: {e}. Falling back to default weights.")
            # Fall back to default weights
            for i, (explicit_w, is_neg) in enumerate(zip(explicit_weights, negation_flags)):
                if explicit_w is not None:
                    final_weights.append(explicit_w)
                else:
                    w = default_weight
                    if is_neg:
                        w = -abs(w)
                    final_weights.append(w)
    else:
        # Not using CLIP or all weights explicitly provided
        for i, (explicit_w, is_neg) in enumerate(zip(explicit_weights, negation_flags)):
            if explicit_w is not None:
                final_weights.append(explicit_w)
            else:
                w = default_weight
                if is_neg:
                    w = -abs(w)
                final_weights.append(w)
    
    assert len(processed_prompts) == len(final_weights)
    return processed_prompts, final_weights
