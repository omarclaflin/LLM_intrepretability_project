"""
Neuroscience analysis on SAE features with Neuron Input RSA approach.

This script performs RSA analysis on individual SAE features by examining their
top 50 input neuron weights. For each SAE feature, we:
1. Identify the top 50 neurons from the previous layer that most strongly influence it
2. Extract activations from only those 50 neurons (collapsed across token window)
3. Run RSA to test representational clustering within vs across categories
4. Rank SAE features by their RSA discriminative power

This contrasts with approaches that select features first, then run RSA on the selected set.
"""

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import random
from typing import List, Tuple, Dict, Any, Optional
import time
import datetime
import logging
import os
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle numpy scalars - check type module rather than specific types
    if isinstance(obj, np.number):
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        elif np.issubdtype(type(obj), np.complexfloating):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        else:
            return obj.item()  # Fallback for other numeric types

    # Handle numpy bool
    if isinstance(obj, np.bool_):
        return bool(obj)

    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(i) for i in obj]

    # Return unchanged for other types
    return obj

class SparseAutoencoder(torch.nn.Module):
    """Simple Sparse Autoencoder module."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Encoder maps input activation vector to sparse feature vector
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU() # ReLU enforces non-negativity for sparsity
        )
        # Decoder maps sparse feature vector back to original activation space
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        # x: input activation vector (batch_size, seq_len, input_dim)
        features = self.encoder(x) # features: (batch_size, seq_len, hidden_dim)
        # features are already non-negative due to ReLU

        # Reconstruction maps features back to the input activation space
        reconstruction = self.decoder(features) # reconstruction: (batch_size, seq_len, input_dim)

        return features, reconstruction # Return both the features and the reconstruction

# Modified function to accept category keys
def load_statements_from_json(json_file, category_keys):
    """
    Load statements from the provided JSON file using specified keys.

    Args:
        json_file: Path to JSON file containing statements
        category_keys: A list or tuple of exactly two keys to use for the two categories

    Returns:
        A dictionary mapping category names (keys) to lists of statements (values)
        and a list of the category names in the order they were loaded.
    """
    if len(category_keys) != 2:
        raise ValueError("Exactly two category keys must be provided.")

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    statements = {}
    for key in category_keys:
        if key in data and isinstance(data[key], list):
            statements[key] = data[key]
        else:
            raise KeyError(f"Key '{key}' not found or is not a list in the JSON file.")

    return statements, list(category_keys) # Return statements dict and ordered list of names


def compute_middle_position(text, tokenizer):
    """
    Compute the middle token position for a given text.

    Args:
        text: Input text
        tokenizer: Tokenizer object

    Returns:
        Middle token position and total sequence length
    """
    # Tokenize without adding special tokens initially to get raw token count
    tokens = tokenizer.encode(text, add_special_tokens=False)
    seq_len = len(tokens)
    middle_pos = seq_len // 2

    return middle_pos, seq_len

def extract_hidden_states_from_middle(model, tokenizer, texts, token_distance, target_layer=16):
    """
    Extract hidden states for windows around the middle of each text.

    Args:
        model: Base language model
        tokenizer: Tokenizer
        texts: List of texts
        token_distance: Number of tokens to include before and after the middle
        target_layer: Which model layer to extract hidden states from

    Returns:
        Array of shape (n_texts, 2*token_distance+1, input_dim)
        and list of actual middle positions
    """
    device = next(model.parameters()).device
    window_size = 2 * token_distance + 1
    all_hidden_states = []
    middle_positions = []
    valid_indices = []

    # Pre-compute middle positions and check validity
    for i, text in enumerate(tqdm(texts, desc="Pre-computing middle positions")):
        # Tokenize with special tokens here for model input later
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        seq_len = torch.sum(inputs["attention_mask"][0]).item()

        # Compute middle position based on tokens *without* special tokens
        raw_tokens = tokenizer.encode(text, add_special_tokens=False)
        raw_seq_len = len(raw_tokens)
        raw_middle_pos = raw_seq_len // 2

        # We need to find the corresponding position in the sequence *with* special tokens
        # This is tricky and depends on tokenizer behavior. A simple approach is to
        # find the index of the token that was at the raw_middle_pos in the full sequence.
        # This might not be perfect but is a reasonable approximation for minimal changes.
        # If the raw sequence is empty, skip.
        if raw_seq_len > 0:
             # Let's simplify and just assume the middle token position in the raw sequence
             # roughly corresponds to that position plus the BOS token index in the full sequence.
             # This is imprecise but keeps changes minimal.
             approx_middle_pos_in_full = raw_middle_pos + (1 if tokenizer.bos_token else 0)

             # Check if we have enough tokens around this approximate middle position in the *full* sequence
             if approx_middle_pos_in_full - token_distance >= 0 and approx_middle_pos_in_full + token_distance < seq_len:
                 middle_positions.append(approx_middle_pos_in_full) # Store the position in the full sequence
                 valid_indices.append(i)
             else:
                 # If not enough tokens, log and skip
                 logging.debug(f"Skipping text '{text[:50]}...' (ID: {i}) due to insufficient tokens around middle "
                               f"(seq_len: {seq_len}, raw_seq_len: {raw_seq_len}, raw_middle: {raw_middle_pos}, "
                               f"approx_full_middle: {approx_middle_pos_in_full}, window_size: {window_size})")
        else:
             logging.debug(f"Skipping empty raw text for ID {i}")

    logging.info(f"Found {len(valid_indices)} texts with sufficient tokens around approximate middle")

    # Now extract hidden states for the valid indices using the pre-computed middle positions
    for i, idx in enumerate(tqdm(valid_indices, desc="Extracting hidden states")):
        text = texts[idx]
        middle_pos = middle_positions[i] # Use the pre-computed position in the full sequence

        inputs = tokenizer(text, return_tensors="pt").to(device)
        seq_len = torch.sum(inputs["attention_mask"][0]).item() # Recalculate seq_len for safety

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[target_layer]

            # Ensure hidden_states and inputs match sequence length
            if hidden_states.size(1) != seq_len:
                 logging.warning(f"Hidden state length ({hidden_states.size(1)}) does not match input sequence length ({seq_len}) for text ID {idx}. Skipping.")
                 continue

            # Extract window around middle position using the pre-computed middle_pos
            start_pos = middle_pos - token_distance
            end_pos = middle_pos + token_distance + 1

            # Double-check bounds before slicing
            if start_pos >= 0 and end_pos <= hidden_states.size(1):
                window = hidden_states[0, start_pos:end_pos]
                all_hidden_states.append(window.cpu().numpy())
            else:
                 logging.warning(f"Window bounds out of range for text ID {idx} (start: {start_pos}, end: {end_pos}, seq_len: {hidden_states.size(1)}). Skipping.")


    # Stack all valid hidden states
    if all_hidden_states:
        all_hidden_states = np.stack(all_hidden_states)
        logging.info(f"Extracted hidden states shape: {all_hidden_states.shape}")
    else:
        logging.error("No valid hidden states extracted!")
        all_hidden_states = np.array([])

    # Return hidden states and the middle positions *for the texts that were actually processed*
    return all_hidden_states, [middle_positions[i] for i in range(len(valid_indices))]


def compute_within_correlations(activations):
    """
    Compute within-set representational similarity.

    Args:
        activations: Array of shape (n_texts, n_features)

    Returns:
        window_correlations: Correlations for the window
    """
    n_texts = activations.shape[0]

    # Check if there are enough samples and features to compute correlations
    if n_texts < 2 or activations.shape[1] == 0:
        logging.warning("Not enough samples or features for within-set correlation.")
        return np.array([])

    # Check if activations have enough variance to compute similarity
    if np.std(activations) > 1e-9:
        window_corr_matrix = cosine_similarity(activations)
        # Extract upper triangle (excluding diagonal)
        window_correlations = window_corr_matrix[np.triu_indices(n_texts, k=1)]
    else:
        logging.warning("Zero variance in activations for within-set correlation.")
        window_correlations = np.array([])

    return window_correlations

def compute_across_correlations(set1_activations, set2_activations):
    """
    Compute cross-correlations between two sets.

    Args:
        set1_activations: Array for set 1 of shape (n_texts_set1, n_features)
        set2_activations: Array for set 2 of shape (n_texts_set2, n_features)

    Returns:
        window_cross_correlations: Cross-correlations
    """
    # Check if there are enough samples and features
    if set1_activations.shape[0] == 0 or set2_activations.shape[0] == 0 or set1_activations.shape[1] == 0:
        logging.warning("Not enough samples or features for across-set correlation.")
        return np.array([])

    # Check if activations have enough variance
    if np.std(set1_activations) > 1e-9 and np.std(set2_activations) > 1e-9:
        window_cross_corr = cosine_similarity(set1_activations, set2_activations)
        window_cross_correlations = window_cross_corr.flatten()
    else:
        logging.warning("Zero variance in activations for across-set correlation.")
        window_cross_correlations = np.array([])

    return window_cross_correlations

def fisher_transform(r):
    """
    Apply Fisher's r-to-z transformation to correlation values.
    Handles empty arrays.
    """
    if r.size == 0:
        return np.array([])
    # Clip correlation values to valid range for arctanh
    return np.arctanh(np.clip(r, -0.999, 0.999))

def bootstrap_significance_test(within_corrs, across_corrs, n_bootstraps=1000):
    """
    Run bootstrapped significance test between two sets of correlations.

    Args:
        within_corrs: Correlation values within set
        across_corrs: Correlation values across sets
        n_bootstraps: Number of bootstrap samples

    Returns:
        p_value: Bootstrap p-value
        observed_diff: Observed difference in means
    """
    # Apply Fisher transformation
    within_z = fisher_transform(1 - within_corrs) # Use dissimilarity (1-r)
    across_z = fisher_transform(1 - across_corrs) # Use dissimilarity (1-r)

    # Check if there's enough data after transformation
    if within_z.size == 0 or across_z.size == 0:
        logging.warning("Not enough valid correlations for bootstrap test.")
        return 1.0, np.nan # Return non-significant p-value and NaN effect size

    # Observed difference in means (across dissimilarity - within dissimilarity)
    # A positive difference indicates the feature differentiates the sets.
    observed_diff = np.mean(across_z) - np.mean(within_z)

    # Bootstrap
    bootstrap_diffs = []
    combined = np.concatenate([within_z, across_z])
    n_within = len(within_z)
    n_across = len(across_z)

    # Ensure combined has at least 2 elements for shuffling
    if len(combined) < 2:
         logging.warning("Combined correlations array too small for bootstrapping.")
         return 1.0, observed_diff

    for _ in range(n_bootstraps):
        # Permute the combined data
        permuted_combined = np.random.permutation(combined)
        # Split back into two groups of original sizes
        bootstrap_within = permuted_combined[:n_within]
        bootstrap_across = permuted_combined[n_within:n_within + n_across] # Ensure correct slicing

        # Compute difference in means for bootstrap sample
        # Handle cases where bootstrap samples might have zero variance (unlikely with enough data)
        if len(bootstrap_within) > 0 and len(bootstrap_across) > 0:
             bootstrap_diff = np.mean(bootstrap_across) - np.mean(bootstrap_within)
             bootstrap_diffs.append(bootstrap_diff)
        else:
             # If a sample is empty (shouldn't happen with correct slicing), skip or assign NaN
             bootstrap_diffs.append(np.nan) # Or just continue

    # Filter out any NaN bootstrap differences if they occurred
    bootstrap_diffs = np.array(bootstrap_diffs)
    bootstrap_diffs = bootstrap_diffs[~np.isnan(bootstrap_diffs)]

    if len(bootstrap_diffs) == 0:
        logging.warning("No valid bootstrap differences computed.")
        return 1.0, observed_diff

    # Compute p-value (proportion of bootstrap differences >= observed)
    # We are testing if observed_diff is significantly greater than 0 (across > within)
    p_value = np.mean(bootstrap_diffs >= observed_diff) # One-tailed test for positive effect

    return p_value, observed_diff

def run_individual_feature_rsa(set1_hidden_states, set2_hidden_states, sae_model, n_top_neurons=50, n_bootstraps=1000):
    """
    Run RSA analysis on each individual SAE feature using its top input neurons.

    Args:
        set1_hidden_states: Hidden states for set 1, shape (n_texts_set1, window_size, input_dim)
        set2_hidden_states: Hidden states for set 2, shape (n_texts_set2, window_size, input_dim)
        sae_model: Trained SAE model
        n_top_neurons: Number of top input neurons to use per SAE feature
        n_bootstraps: Number of bootstrap samples for significance testing

    Returns:
        results: List of dicts with RSA results for each SAE feature
    """
    # Get decoder weights (shape: input_dim, hidden_dim)
    decoder_weights = sae_model.decoder.weight.data.cpu().numpy()  # Shape: (input_dim, hidden_dim)
    n_sae_features = decoder_weights.shape[1]

    results = []

    # Collapse across token window for both sets
    set1_collapsed = np.mean(set1_hidden_states, axis=1)  # Shape: (n_texts_set1, input_dim)
    set2_collapsed = np.mean(set2_hidden_states, axis=1)  # Shape: (n_texts_set2, input_dim)

    logging.info(f"Running RSA analysis on {n_sae_features} SAE features...")

    for feature_idx in tqdm(range(n_sae_features), desc="Processing SAE features"):
        # Get decoder weights for this feature (which input neurons it reconstructs)
        feature_weights = decoder_weights[:, feature_idx]  # Shape: (input_dim,)

        # Find top n_top_neurons by absolute weight value
        top_neuron_indices = np.argsort(np.abs(feature_weights))[-n_top_neurons:]

        # Extract activations for only these top neurons
        set1_feature_activations = set1_collapsed[:, top_neuron_indices]  # Shape: (n_texts_set1, n_top_neurons)
        set2_feature_activations = set2_collapsed[:, top_neuron_indices]  # Shape: (n_texts_set2, n_top_neurons)

        # Compute within-set correlations for set1
        within_correlations = compute_within_correlations(set1_feature_activations)

        # Compute across-set correlations
        across_correlations = compute_across_correlations(set1_feature_activations, set2_feature_activations)

        # Run bootstrap significance test
        p_value, effect_size = bootstrap_significance_test(within_correlations, across_correlations, n_bootstraps)

        # Store results
        result = {
            'feature_idx': feature_idx,
            'top_neuron_indices': top_neuron_indices.tolist(),
            'top_neuron_weights': feature_weights[top_neuron_indices].tolist(),
            'p_value': p_value,
            'effect_size': effect_size,
            'n_within_correlations': len(within_correlations),
            'n_across_correlations': len(across_correlations)
        }
        results.append(result)

        # Log progress for very significant features
        if p_value < 0.01:
            logging.debug(f"Feature {feature_idx}: p={p_value:.4f}, effect={effect_size:.4f}")

    return results

def plot_rsa_results_summary(rsa_results, output_path, top_k=50):
    """
    Plot summary of RSA results across all SAE features.
    """
    # Extract p-values and effect sizes
    p_values = [r['p_value'] for r in rsa_results]
    effect_sizes = [r['effect_size'] for r in rsa_results]
    feature_indices = [r['feature_idx'] for r in rsa_results]

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: P-value distribution
    axes[0, 0].hist(p_values, bins=50, alpha=0.7, color='blue')
    axes[0, 0].axvline(0.05, color='red', linestyle='--', label='p=0.05')
    axes[0, 0].set_xlabel('P-value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of RSA P-values')
    axes[0, 0].legend()

    # Plot 2: Effect size distribution
    axes[0, 1].hist(effect_sizes, bins=50, alpha=0.7, color='green')
    axes[0, 1].axvline(0, color='red', linestyle='--', label='No effect')
    axes[0, 1].set_xlabel('Effect Size')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of RSA Effect Sizes')
    axes[0, 1].legend()

    # Plot 3: P-value vs Effect size scatter
    axes[1, 0].scatter(effect_sizes, [-np.log10(max(p, 1e-10)) for p in p_values], alpha=0.6)
    axes[1, 0].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Effect Size')
    axes[1, 0].set_ylabel('-log10(P-value)')
    axes[1, 0].set_title('Volcano Plot: Effect Size vs Significance')
    axes[1, 0].legend()

    # Plot 4: Top features by p-value
    # Sort by p-value and show top k
    sorted_results = sorted(rsa_results, key=lambda x: x['p_value'])
    top_features = sorted_results[:top_k]
    top_p_values = [r['p_value'] for r in top_features]
    top_feature_indices = [r['feature_idx'] for r in top_features]

    y_pos = np.arange(len(top_features))
    axes[1, 1].barh(y_pos, [-np.log10(max(p, 1e-10)) for p in top_p_values])
    axes[1, 1].axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    axes[1, 1].set_yticks(y_pos[::5])  # Show every 5th label to avoid crowding
    axes[1, 1].set_yticklabels([f'F{idx}' for idx in top_feature_indices[::5]])
    axes[1, 1].set_xlabel('-log10(P-value)')
    axes[1, 1].set_ylabel('SAE Feature')
    axes[1, 1].set_title(f'Top {top_k} Features by Significance')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Neuroscience analysis on SAE features with Neuron Input RSA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to trained SAE model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--input_json", type=str, required=True, help="JSON file with statements for two categories")
    parser.add_argument("--token_distance", type=int, default=5, help="Number of tokens to analyze before and after middle")
    parser.add_argument("--n_top_neurons", type=int, default=50, help="Number of top input neurons to use per SAE feature")
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstrap samples for significance testing")
    parser.add_argument("--target_layer", type=int, default=16, help="Model layer to extract hidden states from")

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"NeuronInputRSA_neuroscience_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "analysis.log"),
            logging.StreamHandler()
        ]
    )

    # Log parameters
    logging.info(f"Analysis started with parameters:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    # Save parameters for reproducibility
    with open(output_dir / "parameters.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load models
    logging.info("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Set pad_token to eos_token if it's None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token for tokenizer")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")

    # Load SAE model
    logging.info("Loading SAE model...")
    state_dict = torch.load(args.sae_path, map_location="cuda")

    # Determine dimensions from state dict
    if 'decoder.weight' in state_dict:
        decoder_weight = state_dict['decoder.weight']
        input_dim = decoder_weight.shape[0]
        hidden_dim = decoder_weight.shape[1]
    elif 'encoder.0.weight' in state_dict:
        encoder_weight = state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
    else:
        raise ValueError("Could not determine SAE dimensions from state dict keys")

    logging.info(f"Creating SAE with input_dim={input_dim}, hidden_dim={hidden_dim}")
    sae_model = SparseAutoencoder(input_dim, hidden_dim)
    sae_model.load_state_dict(state_dict)
    sae_model.to(model.device)
    sae_model.eval()

    # Load statements from JSON and determine category names dynamically
    logging.info(f"Loading statements from {args.input_json}...")
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        category_keys = list(data.keys())
        if len(category_keys) != 2:
            raise ValueError(f"Input JSON must contain exactly two top-level keys, but found {len(category_keys)}: {category_keys}")

        statements_dict, category_names = load_statements_from_json(args.input_json, category_keys)

        # Assign statements to set1 and set2 based on the order of keys found
        set1_name = category_names[0]
        set2_name = category_names[1]
        set1_statements = statements_dict[set1_name]
        set2_statements = statements_dict[set2_name]

        logging.info(f"Loaded {len(set1_statements)} statements for category '{set1_name}' and {len(set2_statements)} statements for category '{set2_name}'")

    except Exception as e:
        logging.error(f"Error loading or parsing input JSON: {e}")
        return # Exit if JSON loading/parsing fails


    # Extract hidden states centered on the middle of each statement
    logging.info(f"Extracting hidden states for '{set1_name}' statements...")
    set1_hidden_states, set1_middles = extract_hidden_states_from_middle(
        model, tokenizer, set1_statements,
        args.token_distance, args.target_layer
    )

    logging.info(f"Extracting hidden states for '{set2_name}' statements...")
    set2_hidden_states, set2_middles = extract_hidden_states_from_middle(
        model, tokenizer, set2_statements,
        args.token_distance, args.target_layer
    )

    if len(set1_hidden_states) == 0 or len(set2_hidden_states) == 0:
        logging.error("No valid hidden states extracted from one or both sets. Cannot proceed with analysis.")
        return

    logging.info(f"Extracted hidden states with shapes: {set1_hidden_states.shape}, {set2_hidden_states.shape}")

    # Run individual feature RSA analysis
    logging.info("Running individual SAE feature RSA analysis...")
    rsa_results = run_individual_feature_rsa(
        set1_hidden_states, set2_hidden_states, sae_model,
        n_top_neurons=args.n_top_neurons, n_bootstraps=args.n_bootstraps
    )

    # Sort results by p-value (most significant first)
    rsa_results_sorted = sorted(rsa_results, key=lambda x: x['p_value'])

    logging.info(f"RSA analysis complete. Found {len(rsa_results)} features.")
    logging.info(f"Most significant feature: Feature {rsa_results_sorted[0]['feature_idx']} (p={rsa_results_sorted[0]['p_value']:.6f})")

    # Save detailed results
    with open(output_dir / "rsa_results_detailed.json", "w") as f:
        json.dump(convert_numpy_types(rsa_results_sorted), f, indent=2)
    logging.info("Detailed RSA results saved to rsa_results_detailed.json")

    # Save summary of top features
    top_features_summary = []
    for i, result in enumerate(rsa_results_sorted[:100]):  # Top 100 features
        summary = {
            'rank': i + 1,
            'feature_idx': result['feature_idx'],
            'p_value': result['p_value'],
            'effect_size': result['effect_size'],
            'n_within_correlations': result['n_within_correlations'],
            'n_across_correlations': result['n_across_correlations']
        }
        top_features_summary.append(summary)

    with open(output_dir / "top_features_summary.json", "w") as f:
        json.dump(convert_numpy_types(top_features_summary), f, indent=2)
    logging.info("Top features summary saved to top_features_summary.json")

    # Save feature indices in part4-compatible format
    top_feature_indices = [result['feature_idx'] for result in rsa_results_sorted[:100]]
    with open(output_dir / "top_features.json", "w") as f:
        json.dump(convert_numpy_types({
            "top_feature_indices": top_feature_indices,
            "selection_method": "neuron_input_rsa_significance"
        }), f, indent=2)
    logging.info("Part4-compatible feature indices saved to top_features.json")

    # Plot summary results
    logging.info("Creating summary visualization...")
    plot_rsa_results_summary(rsa_results, output_dir / "rsa_summary_plots.png", top_k=50)

    # Save analysis summary
    n_significant = sum(1 for r in rsa_results if r['p_value'] < 0.05)
    n_very_significant = sum(1 for r in rsa_results if r['p_value'] < 0.01)

    summary_stats = {
        'total_features_analyzed': len(rsa_results),
        'significant_features_p005': n_significant,
        'very_significant_features_p001': n_very_significant,
        'top_feature_idx': rsa_results_sorted[0]['feature_idx'],
        'top_feature_p_value': rsa_results_sorted[0]['p_value'],
        'top_feature_effect_size': rsa_results_sorted[0]['effect_size'],
        'analysis_parameters': {
            'n_top_neurons': args.n_top_neurons,
            'n_bootstraps': args.n_bootstraps,
            'token_distance': args.token_distance,
            'target_layer': args.target_layer
        }
    }

    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(convert_numpy_types(summary_stats), f, indent=2)

    logging.info(f"Analysis complete! Results saved to {output_dir}")
    logging.info(f"Summary: {n_significant}/{len(rsa_results)} features significant at p<0.05")
    logging.info(f"Top feature: {rsa_results_sorted[0]['feature_idx']} (p={rsa_results_sorted[0]['p_value']:.6f}, effect={rsa_results_sorted[0]['effect_size']:.4f})")

if __name__ == "__main__":
    main()