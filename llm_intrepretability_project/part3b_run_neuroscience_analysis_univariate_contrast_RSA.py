"""
Neuroscience analysis on SAE features with univariate-contrast RSA approach.

This script performs the same analysis as part3 but with a key difference:
1. Runs t-tests on ALL SAE features (not just top-k by activation)
2. Selects top 50 features by statistical significance for RSA analysis
3. Tests whether the most discriminative features show representational clustering

This contrasts with the original approach that selected features by activation magnitude.
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

def extract_sae_activations_from_middle(model, sae_model, tokenizer, texts, token_distance, target_layer=16):
    """
    Extract SAE activations for windows around the middle of each text.

    Args:
        model: Base language model
        sae_model: Sparse autoencoder model
        tokenizer: Tokenizer
        texts: List of texts
        token_distance: Number of tokens to include before and after the middle
        target_layer: Which model layer to extract hidden states from

    Returns:
        Array of shape (n_texts, 2*token_distance+1, n_features)
        and list of actual middle positions
    """
    device = next(model.parameters()).device
    window_size = 2 * token_distance + 1
    all_activations = []
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

    # Now extract activations for the valid indices using the pre-computed middle positions
    for i, idx in enumerate(tqdm(valid_indices, desc="Extracting activations")):
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

            features, _ = sae_model(hidden_states)

            # Extract window around middle position using the pre-computed middle_pos
            start_pos = middle_pos - token_distance
            end_pos = middle_pos + token_distance + 1

            # Double-check bounds before slicing
            if start_pos >= 0 and end_pos <= features.size(1):
                window = features[0, start_pos:end_pos]
                all_activations.append(window.cpu().numpy())
            else:
                 logging.warning(f"Window bounds out of range for text ID {idx} (start: {start_pos}, end: {end_pos}, seq_len: {features.size(1)}). Skipping.")


    # Stack all valid activations
    if all_activations:
        all_activations = np.stack(all_activations)
        logging.info(f"Extracted activations shape: {all_activations.shape}")
    else:
        logging.error("No valid activations extracted!")
        all_activations = np.array([])

    # Return activations and the middle positions *for the texts that were actually processed*
    return all_activations, [middle_positions[i] for i in range(len(valid_indices))]


def run_t_tests(set1_activations, set2_activations, feature_indices=None):
    """
    Run t-tests comparing two sets of activations.

    Args:
        set1_activations: Array of shape (n_texts_set1, 2*token_distance+1, n_features)
        set2_activations: Array of shape (n_texts_set2, 2*token_distance+1, n_features)
        feature_indices: Indices of features to analyze (None for all)

    Returns:
        p_values: Array of shape (2*token_distance+1, len(feature_indices))
        t_values: Array of shape (2*token_distance+1, len(feature_indices))
    """
    n_positions = set1_activations.shape[1]

    if feature_indices is None:
        feature_indices = np.arange(set1_activations.shape[2])

    n_features = len(feature_indices)
    p_values = np.zeros((n_positions, n_features))
    t_values = np.zeros((n_positions, n_features))

    for pos in tqdm(range(n_positions), desc="Running t-tests"):
        for i, feat in enumerate(feature_indices):
            # Extract values for this position and feature
            vals1 = set1_activations[:, pos, feat]
            vals2 = set2_activations[:, pos, feat]

            # Run t-test
            # Check for sufficient data points and non-zero variance
            if len(vals1) > 1 and len(vals2) > 1 and (np.std(vals1) > 1e-9 or np.std(vals2) > 1e-9):
                 t_stat, p_value = stats.ttest_ind(vals1, vals2, equal_var=False)
                 p_values[pos, i] = p_value
                 t_values[pos, i] = t_stat
            else:
                 # Assign NaN or 1.0 if t-test cannot be performed
                 p_values[pos, i] = 1.0 # Not significant if cannot test
                 t_values[pos, i] = np.nan # Not applicable t-stat


    return p_values, t_values

def zscore_p_values(p_values):
    """
    Z-score p-values using -log10 transformation.

    Args:
        p_values: Array of p-values

    Returns:
        z_scored: Z-scored p-values
    """
    # Convert p-values to z-scores (negative log transform)
    # Handle p-values that are 1.0 (from skipped t-tests) or very close to 0
    log_p = -np.log10(np.clip(p_values, 1e-10, 1.0 - 1e-10)) # Clip to avoid log(0) or log(1) issues

    # Compute mean and std for z-scoring, ignoring NaNs if any (from t-stats, though not used here)
    # For z-scoring log_p, we should probably ignore infinities resulting from p=0
    finite_log_p = log_p[np.isfinite(log_p)]

    if len(finite_log_p) > 1:
        mean = np.mean(finite_log_p)
        std = np.std(finite_log_p)

        # Z-score the values, handling potential division by zero if std is tiny
        if std > 1e-9:
            z_scored = (log_p - mean) / std
        else:
            # If std is zero (all finite log_p values are the same), set z-score to 0
            z_scored = np.zeros_like(log_p)
            z_scored[~np.isfinite(log_p)] = np.nan # Keep infinities/NaNs as they were

    else:
        # If not enough finite values to compute mean/std, return NaNs or zeros
        z_scored = np.zeros_like(log_p) # Or np.full_like(log_p, np.nan)
        z_scored[~np.isfinite(log_p)] = np.nan # Keep infinities/NaNs as they were


    return z_scored

def compute_within_correlations(activations, feature_indices=None):
    """
    Compute within-set representational similarity.

    Args:
        activations: Array of shape (n_texts, 2*token_distance+1, n_features)
        feature_indices: Indices of features to use (None for all)

    Returns:
        window_correlations: Correlations for the entire window
        position_correlations: List of correlations for each token position
    """
    n_texts = activations.shape[0]
    n_positions = activations.shape[1]

    # Extract only the selected features if specified
    if feature_indices is not None:
        selected_activations = activations[:, :, feature_indices]
    else:
        selected_activations = activations

    # Check if there are enough samples and features to compute correlations
    if n_texts < 2 or selected_activations.shape[2] == 0:
        logging.warning("Not enough samples or features for within-set correlation.")
        return np.array([]), [np.array([]) for _ in range(n_positions)]


    # 1. Compute correlations for the entire window (mean across token positions)
    # Handle case where selected_activations might be empty after feature selection
    if selected_activations.shape[2] > 0:
        window_means = np.mean(selected_activations, axis=1)  # Shape: (n_texts, n_selected_features)
        # Check if window_means has enough variance to compute similarity
        if np.std(window_means) > 1e-9:
            window_corr_matrix = cosine_similarity(window_means)
            # Extract upper triangle (excluding diagonal)
            window_correlations = window_corr_matrix[np.triu_indices(n_texts, k=1)]
        else:
            logging.warning("Zero variance in window means for within-set correlation.")
            window_correlations = np.array([])
    else:
         logging.warning("No selected features for within-set window correlation.")
         window_correlations = np.array([])


    # 2. Compute correlations for each token position separately
    position_correlations = []
    for pos in range(n_positions):
        pos_activations = selected_activations[:, pos, :]
        # Check if pos_activations has enough variance to compute similarity
        if pos_activations.shape[1] > 0 and np.std(pos_activations) > 1e-9:
            pos_corr_matrix = cosine_similarity(pos_activations)
            pos_correlations_flat = pos_corr_matrix[np.triu_indices(n_texts, k=1)]
            position_correlations.append(pos_correlations_flat)
        else:
            logging.warning(f"Zero variance or no features at position {pos} for within-set correlation.")
            position_correlations.append(np.array([]))

    return window_correlations, position_correlations

def compute_across_correlations(set1_activations, set2_activations, feature_indices=None):
    """
    Compute cross-correlations between two sets.

    Args:
        set1_activations: Array for set 1
        set2_activations: Array for set 2
        feature_indices: Indices of features to use (None for all)

    Returns:
        window_cross_correlations: Cross-correlations for the entire window
        position_cross_correlations: List of cross-correlations for each token position
    """
    n_positions = set1_activations.shape[1] # Assumes same number of positions for both sets

    # Extract only the selected features if specified
    if feature_indices is not None:
        set1_selected = set1_activations[:, :, feature_indices]
        set2_selected = set2_activations[:, :, feature_indices]
    else:
        set1_selected = set1_activations
        set2_selected = set2_activations

    # Check if there are enough samples and features
    if set1_selected.shape[0] == 0 or set2_selected.shape[0] == 0 or set1_selected.shape[2] == 0:
        logging.warning("Not enough samples or features for across-set correlation.")
        return np.array([]), [np.array([]) for _ in range(n_positions)]

    # 1. Compute window means
    set1_window_means = np.mean(set1_selected, axis=1)  # Shape: (n_texts_set1, n_selected_features)
    set2_window_means = np.mean(set2_selected, axis=1)  # Shape: (n_texts_set2, n_selected_features)

    # Compute cross-correlations for window means
    # Check if window_means have enough variance
    if np.std(set1_window_means) > 1e-9 and np.std(set2_window_means) > 1e-9:
        window_cross_corr = cosine_similarity(set1_window_means, set2_window_means)
        window_cross_correlations = window_cross_corr.flatten()
    else:
        logging.warning("Zero variance in window means for across-set window correlation.")
        window_cross_correlations = np.array([])


    # 2. Compute cross-correlations for each position
    position_cross_correlations = []
    for pos in range(n_positions):
        set1_pos = set1_selected[:, pos, :]
        set2_pos = set2_selected[:, pos, :]
        # Check if pos_activations have enough variance
        if set1_pos.shape[1] > 0 and np.std(set1_pos) > 1e-9 and np.std(set2_pos) > 1e-9:
            pos_cross_corr = cosine_similarity(set1_pos, set2_pos)
            position_cross_correlations.append(pos_cross_corr.flatten())
        else:
            logging.warning(f"Zero variance or no features at position {pos} for across-set correlation.")
            position_cross_correlations.append(np.array([]))

    return window_cross_correlations, position_cross_correlations

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

# Modified function to accept category names for plotting
def plot_p_value_heatmap(p_values, t_values, feature_indices, output_path, token_distance=5,
                         statements_dict=None, category_names=None, tokenizer=None):
    """
    Plot heatmap of p-values from t-tests with example statements vertically aligned.
    Uses category names for labels.
    """
    # Z-score the p-values for visualization
    z_scored = zscore_p_values(p_values)

    # Create figure - with more space for examples at the bottom
    plt.figure(figsize=(30, 24))  # Increased even more for larger text

    # Create position labels
    positions = list(range(-token_distance, token_distance + 1))
    position_labels = [f"{pos}" for pos in positions]

    # Main heatmap (top 60% of figure)
    main_ax = plt.subplot2grid((5, 1), (0, 0), rowspan=3)

    # Create heatmap
    # Handle case where z_scored might be empty
    if z_scored.size > 0:
        sns.heatmap(
            z_scored,
            cmap="viridis",
            cbar_kws={"label": "Z-scored -log10(p-value)"},
            yticklabels=position_labels,
            ax=main_ax,
            # Add vmin/vmax for consistent color scale if needed, e.g., vmin=-3, vmax=3
        )

        # Add bold line at middle position
        middle_pos = token_distance
        main_ax.axhline(y=middle_pos + 0.5, color='r', linewidth=2)

        # Add title and labels
        main_ax.set_xlabel("Feature Index", fontsize=18)
        main_ax.set_ylabel("Token Position (relative to middle)", fontsize=18)
        main_ax.set_title("Statistical Significance of SAE Features Across Token Positions", fontsize=24)
    else:
        main_ax.text(0.5, 0.5, "No valid p-values to plot heatmap",
                     horizontalalignment='center', verticalalignment='center', fontsize=16)
        main_ax.set_title("Statistical Significance Heatmap", fontsize=24)
        main_ax.axis('off')


    # Example area (bottom 40% of figure)
    if statements_dict and category_names and tokenizer and len(category_names) == 2:
        set1_statements = statements_dict.get(category_names[0], [])
        set2_statements = statements_dict.get(category_names[1], [])

        if len(set1_statements) >= 2 and len(set2_statements) >= 2:
            example_ax = plt.subplot2grid((5, 1), (3, 0), rowspan=2)
            example_ax.axis('off')

            # Sample 2 examples from each set
            example_statements = random.sample(set1_statements, 2) + random.sample(set2_statements, 2)

            # Calculate mean z-scores across features for color intensity
            # Handle case where z_scored might be empty
            if z_scored.size > 0:
                mean_z_scores = np.mean(z_scored, axis=1)
                norm = plt.Normalize(vmin=np.min(mean_z_scores), vmax=np.max(mean_z_scores))
                cmap = plt.cm.viridis
            else:
                # Use a neutral color if no z-scores
                mean_z_scores = np.zeros(2 * token_distance + 1)
                norm = plt.Normalize(vmin=0, vmax=1)
                cmap = plt.cm.gray # Or any neutral colormap


            # Title for examples - SUPER LARGE TEXT
            example_ax.text(0.5, 0.95, "Example Statements", ha='center', fontsize=32, fontweight='bold')

            # Create a grid layout for examples side by side
            for i, statement in enumerate(example_statements):
                # Horizontal position for each example (4 columns)
                h_pos = 0.125 + i * 0.25  # divide into 4 equal sections (0.125, 0.375, 0.625, 0.875)

                # Tokenize the statement to find the middle position for the example display
                # Use add_special_tokens=False for this calculation to align with how the middle
                # was conceptually defined relative to the original text content.
                tokens = tokenizer.encode(statement, add_special_tokens=False)
                middle_pos_example = len(tokens) // 2

                # Get window tokens relative to this middle position
                # We need to be careful here. The heatmap positions are relative to the middle
                # of the *processed* sequence which might include special tokens.
                # For the example display, we'll show the window around the middle of the *raw* tokens.
                # The color should correspond to the heatmap's position index.
                # This alignment is complex. Let's display the window around the raw middle,
                # but use colors based on the heatmap's position index. This is a visual approximation.

                raw_tokens_list = tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=False)

                start_idx_raw = max(0, middle_pos_example - token_distance)
                end_idx_raw = min(len(raw_tokens_list), middle_pos_example + token_distance + 1)
                window_tokens_raw = raw_tokens_list[start_idx_raw:end_idx_raw]

                # Pad or truncate the displayed tokens to match window size for vertical alignment
                padded_tokens = [''] * (2 * token_distance + 1)
                # Calculate the offset needed to center the raw middle token (index middle_pos_example)
                # within the padded window of size (2*token_distance + 1)
                # The middle index of the padded window is token_distance.
                # The raw middle token is at index `middle_pos_example` in the raw tokens list.
                # The token from `raw_tokens_list` at index `start_idx_raw + j` should go into
                # the padded window at index `j + offset`.
                # So, `middle_pos_example - start_idx_raw + offset = token_distance`
                # `offset = token_distance - (middle_pos_example - start_idx_raw)`
                offset = token_distance - (middle_pos_example - start_idx_raw)


                for j, token in enumerate(window_tokens_raw):
                    if 0 <= j + offset < len(padded_tokens):
                        # Clean common prefix for visualization
                        padded_tokens[j + offset] = token.replace('Ġ', ' ').replace(' ', ' ') # Clean common prefixes

                # Label if from set 1 or set 2
                # Assuming the first two examples are from set 1, and the next two from set 2
                category_label = category_names[0] if i < 2 else category_names[1]
                example_ax.text(h_pos, 0.9, f"Example {i+1} ({category_label})",
                               ha='center', fontsize=28, fontweight='bold')

                # Display tokens vertically, aligned at same position - SUPER LARGE TEXT
                for j, token in enumerate(padded_tokens):
                    # Position from -token_distance to +token_distance
                    pos = j - token_distance
                    # Get the corresponding index in the mean_z_scores array (0 to 2*token_distance)
                    color_index = j # This assumes the padded window index corresponds to the heatmap position index
                    if 0 <= color_index < len(mean_z_scores):
                         color = cmap(norm(mean_z_scores[color_index]))
                    else:
                         color = 'black' # Default color if index is out of bounds (shouldn't happen with correct padding/indexing)


                    # Mark middle token (relative to the raw middle)
                    # The token at index `middle_pos_example` in the raw list is at index `token_distance`
                    # in the padded list if offset calculation is correct.
                    # So, check if the current padded index `j` corresponds to the raw middle token.
                    # This is complex. Let's simplify: mark the token that *would* be at the middle
                    # position (index `token_distance`) in the padded window.
                    if j == token_distance:
                        marker = '* '
                        fontweight = 'bold'
                    else:
                        marker = ''
                        fontweight = 'normal'

                    # Token position label (relative to the center of the window) - LARGER
                    example_ax.text(h_pos - 0.06, 0.8 - j * 0.07, f"{pos:+d}",
                                   fontsize=20, ha='right', va='center')

                    # Display token - SUPER LARGE TEXT
                    example_ax.text(h_pos, 0.8 - j * 0.07, f"{marker}{token}",
                                   fontsize=28, color=color, ha='left', va='center',
                                   fontweight=fontweight,
                                   bbox=dict(facecolor='white', alpha=0.7, pad=4))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create a detailed view of the middle position
    plt.figure(figsize=(20, 6))

    # Extract middle position data
    middle_pos_index = token_distance # Index in the 0 to 2*token_distance range
    # Check if middle_pos_index is valid
    if 0 <= middle_pos_index < p_values.shape[0]:
        middle_p_values = p_values[middle_pos_index, :]
        middle_t_values = t_values[middle_pos_index, :]
        middle_z_scores = z_scored[middle_pos_index, :]

        # Plot t-values at middle position
        plt.subplot(2, 1, 1)
        plt.bar(range(len(middle_t_values)), middle_t_values, color='blue', alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.title("T-values at Middle Position", fontsize=18)
        plt.xlabel("Feature Index", fontsize=16)
        plt.ylabel("T-value", fontsize=16)

        # Plot z-scored p-values at middle position
        plt.subplot(2, 1, 2)
        plt.bar(range(len(middle_z_scores)), middle_z_scores, color='green', alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Add significance threshold line (p=0.05 after Bonferroni correction)
        # Calculate number of tests carefully: number of positions * number of features analyzed
        num_features_analyzed = p_values.shape[1]
        num_positions_analyzed = p_values.shape[0]
        n_tests = num_features_analyzed * num_positions_analyzed
        if n_tests > 0:
             sig_threshold = -np.log10(0.05 / n_tests)
             plt.axhline(y=sig_threshold, color='r', linestyle='--', label=f'p=0.05 (corrected)')
             plt.legend(fontsize=14)
        else:
             logging.warning("Number of tests is zero, cannot plot significance threshold.")


        plt.title("Z-scored -log10(p-values) at Middle Position", fontsize=18)
        plt.xlabel("Feature Index", fontsize=16)
        plt.ylabel("Z-score", fontsize=16)

    else:
        # If middle position index is invalid, plot empty subplots or a message
        plt.subplot(2, 1, 1)
        plt.text(0.5, 0.5, "Middle position data not available",
                 horizontalalignment='center', verticalalignment='center', fontsize=16)
        plt.title("T-values at Middle Position", fontsize=18)
        plt.axis('off')

        plt.subplot(2, 1, 2)
        plt.text(0.5, 0.5, "Middle position data not available",
                 horizontalalignment='center', verticalalignment='center', fontsize=16)
        plt.title("Z-scored -log10(p-values) at Middle Position", fontsize=18)
        plt.axis('off')


    plt.tight_layout()
    plt.savefig(str(output_path).replace(".png", "_middle_position.png"), dpi=300)
    plt.close()

# Modified function to accept category names for plotting
def plot_rsa_results(window_p_value, window_effect, position_p_values, position_effects,
                    output_path, token_distance=5, statements_dict=None, category_names=None,
                    tokenizer=None):
    """
    Plot RSA results with example statements at the bottom.
    Uses category names for labels.
    """
    # Main figure with plots
    plt.figure(figsize=(30, 16))

    # Create position labels
    positions = list(range(-token_distance, token_distance + 1))

    # Main plots (top portion)
    main_ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
    main_ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1)

    # Plot p-values across token positions
    if position_p_values: # Check if list is not empty
        main_ax1.plot(positions, position_p_values, 'o-', linewidth=3, markersize=10, label='Position p-values')
        main_ax1.axhline(0.05, color='r', linestyle='--', linewidth=3, label='p=0.05')

        # Highlight middle position
        main_ax1.axvline(0, color='g', linestyle=':', linewidth=3, label='Middle Position')

        # Add window p-value as text
        main_ax1.text(0.7, 0.8, f"Window p-value: {window_p_value:.4f}",
                 transform=main_ax1.transAxes, fontsize=18,
                 bbox=dict(facecolor='white', alpha=0.8))

        main_ax1.set_xlabel("Token Position (relative to middle)", fontsize=18)
        main_ax1.set_ylabel("p-value", fontsize=18)
        main_ax1.set_title("RSA Significance Across Token Positions", fontsize=24)
        main_ax1.tick_params(axis='both', which='major', labelsize=16)
        main_ax1.legend(fontsize=16)
    else:
        main_ax1.text(0.5, 0.5, "No position p-values to plot",
                      horizontalalignment='center', verticalalignment='center', fontsize=16)
        main_ax1.set_title("RSA Significance Across Token Positions", fontsize=24)
        main_ax1.axis('off')


    # Plot effect sizes across token positions
    if position_effects: # Check if list is not empty
        main_ax2.plot(positions, position_effects, 'o-', color='orange', linewidth=3, markersize=10, label='Position effect sizes')

        # Highlight middle position
        main_ax2.axvline(0, color='g', linestyle=':', linewidth=3, label='Middle Position')

        # Add window effect size as text
        main_ax2.text(0.7, 0.8, f"Window effect size: {window_effect:.4f}",
                 transform=main_ax2.transAxes, fontsize=18,
                 bbox=dict(facecolor='white', alpha=0.8))

        main_ax2.set_xlabel("Token Position (relative to middle)", fontsize=18)
        main_ax2.set_ylabel("Effect Size", fontsize=18)
        main_ax2.set_title("RSA Effect Sizes Across Token Positions", fontsize=24)
        main_ax2.tick_params(axis='both', which='major', labelsize=16)
        main_ax2.legend(fontsize=16)
    else:
         main_ax2.text(0.5, 0.5, "No position effect sizes to plot",
                       horizontalalignment='center', verticalalignment='center', fontsize=16)
         main_ax2.set_title("RSA Effect Sizes Across Token Positions", fontsize=24)
         main_ax2.axis('off')


    # Save main figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create separate figure for examples at the bottom
    if statements_dict and category_names and tokenizer and len(category_names) == 2:
        set1_statements = statements_dict.get(category_names[0], [])
        set2_statements = statements_dict.get(category_names[1], [])

        if len(set1_statements) >= 2 and len(set2_statements) >= 2:
            # Create a new figure for examples
            example_fig = plt.figure(figsize=(30, 16))
            example_ax = example_fig.add_subplot(111)
            example_ax.axis('off')

            # Sample 2 examples from each set
            example_statements = random.sample(set1_statements, 2) + random.sample(set2_statements, 2)

            # Normalize effect sizes for color
            # Handle case where position_effects might be empty
            if position_effects:
                norm = plt.Normalize(vmin=min(position_effects), vmax=max(position_effects))
                cmap = plt.cm.plasma  # Different colormap from heatmap
            else:
                 # Use a neutral color if no effect sizes
                 position_effects = [0] * (2 * token_distance + 1) # Create dummy list for indexing
                 norm = plt.Normalize(vmin=0, vmax=1)
                 cmap = plt.cm.gray # Or any neutral colormap


            # Title for examples - SUPER LARGE TEXT
            example_ax.text(0.5, 0.95, "Example Statements", ha='center', fontsize=32, fontweight='bold')

            # Display examples side by side
            for i, statement in enumerate(example_statements):
                # Horizontal position for each example (4 columns)
                h_pos = 0.125 + i * 0.25  # divide into 4 equal sections

                # Tokenize the statement to find the middle position for the example display
                # Use add_special_tokens=False for this calculation
                tokens = tokenizer.encode(statement, add_special_tokens=False)
                middle_pos_example = len(tokens) // 2

                # Get window tokens relative to this middle position
                raw_tokens_list = tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=False)

                start_idx_raw = max(0, middle_pos_example - token_distance)
                end_idx_raw = min(len(raw_tokens_list), middle_pos_example + token_distance + 1)
                window_tokens_raw = raw_tokens_list[start_idx_raw:end_idx_raw]

                # Pad or truncate the displayed tokens to match window size for vertical alignment
                padded_tokens = [''] * (2 * token_distance + 1)
                offset = token_distance - (middle_pos_example - start_idx_raw)

                for j, token in enumerate(window_tokens_raw):
                    if 0 <= j + offset < len(padded_tokens):
                        padded_tokens[j + offset] = token.replace('Ġ', ' ').replace(' ', ' ') # Clean common prefixes

                # Label if from set 1 or set 2
                # Assuming the first two examples are from set 1, and the next two from set 2
                category_label = category_names[0] if i < 2 else category_names[1]
                example_ax.text(h_pos, 0.9, f"Example {i+1} ({category_label})",
                               ha='center', fontsize=28, fontweight='bold')

                # Display tokens vertically, aligned at same position - SUPER LARGE TEXT
                for j, token in enumerate(padded_tokens):
                    # Position from -token_distance to +token_distance
                    pos = j - token_distance
                    # Color based on effect size at this position
                    color_index = j # This assumes the padded window index corresponds to the position effect index
                    if 0 <= color_index < len(position_effects):
                         color = cmap(norm(position_effects[color_index]))
                    else:
                         color = 'black' # Default color if index is out of bounds


                    # Mark middle token (relative to the raw middle) - see heatmap function for logic
                    if j == token_distance:
                        marker = '* '
                        fontweight = 'bold'
                    else:
                        marker = ''
                        fontweight = 'normal'

                    # Token position label - LARGER
                    example_ax.text(h_pos - 0.06, 0.8 - j * 0.07, f"{pos:+d}",
                                   fontsize=20, ha='right', va='center')

                    # Display token - SUPER LARGE TEXT
                    example_ax.text(h_pos, 0.8 - j * 0.07, f"{marker}{token}",
                                   fontsize=28, color=color, ha='left', va='center',
                                   fontweight=fontweight,
                                   bbox=dict(facecolor='white', alpha=0.7, pad=4))

        # Save example figure
        example_path = str(output_path).replace(".png", "_examples.png")
        example_fig.tight_layout()
        example_fig.savefig(example_path, dpi=300, bbox_inches='tight')
        plt.close(example_fig)


def plot_p_value_distribution(p_values, output_path, token_distance=5):
    """
    Plot distribution of p-values.

    Args:
        p_values: Array of p-values
        output_path: Path to save the plot
        token_distance: Number of tokens before/after middle
    """
    plt.figure(figsize=(15, 10))

    # Plot overall p-value distribution
    plt.subplot(2, 2, 1)
    if p_values.size > 0:
        plt.hist(p_values.flatten(), bins=50, alpha=0.7)
        plt.xlabel("p-value", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Overall Distribution of Raw p-values", fontsize=16)
    else:
        plt.text(0.5, 0.5, "No p-values available", horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title("Overall Distribution of Raw p-values", fontsize=16)
        plt.axis('off')


    # Plot p-value distribution for middle position
    middle_pos = token_distance
    plt.subplot(2, 2, 2)
    if p_values.shape[0] > middle_pos and p_values[middle_pos, :].size > 0:
        plt.hist(p_values[middle_pos, :], bins=50, alpha=0.7, color='green')
        plt.xlabel("p-value", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("P-value Distribution at Middle Position", fontsize=16)
    else:
        plt.text(0.5, 0.5, "Middle position p-values not available", horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title("P-value Distribution at Middle Position", fontsize=16)
        plt.axis('off')

    # Plot z-scored distribution
    plt.subplot(2, 2, 3)
    z_scored = zscore_p_values(p_values)
    if z_scored.size > 0:
        plt.hist(z_scored.flatten(), bins=50, alpha=0.7)
        plt.xlabel("Z-scored -log10(p-value)", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Distribution of Z-scored p-values", fontsize=16)
    else:
        plt.text(0.5, 0.5, "No z-scored p-values available", horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title("Distribution of Z-scored p-values", fontsize=16)
        plt.axis('off')

    # QQ-plot to check for uniformity of p-values
    plt.subplot(2, 2, 4)
    if p_values.size > 1: # Need at least 2 values for QQ plot
        sorted_p = np.sort(p_values.flatten())
        # Remove NaNs if any before plotting
        sorted_p = sorted_p[~np.isnan(sorted_p)]
        if len(sorted_p) > 1:
            expected = np.linspace(0, 1, len(sorted_p) + 2)[1:-1]
            plt.plot(expected, sorted_p, 'o', alpha=0.3)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel("Expected (Uniform Distribution)", fontsize=14)
            plt.ylabel("Observed p-values", fontsize=14)
            plt.title("QQ-Plot of p-values", fontsize=16)
        else:
             plt.text(0.5, 0.5, "Not enough p-values for QQ-Plot", horizontalalignment='center', verticalalignment='center', fontsize=14)
             plt.title("QQ-Plot of p-values", fontsize=16)
             plt.axis('off')
    else:
        plt.text(0.5, 0.5, "Not enough p-values for QQ-Plot", horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title("QQ-Plot of p-values", fontsize=16)
        plt.axis('off')


    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Neuroscience analysis on SAE features with univariate-contrast RSA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to trained SAE model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--input_json", type=str, required=True, help="JSON file with statements for two categories")
    parser.add_argument("--token_distance", type=int, default=5, help="Number of tokens to analyze before and after middle")
    parser.add_argument("--top_k", type=int, default=500, help="Number of top SAE features to analyze for t-tests")
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstrap samples for significance testing")
    parser.add_argument("--target_layer", type=int, default=16, help="Model layer to extract hidden states from")

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"univariate_RSA_neuroscience_analysis_{timestamp}"
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


    # Extract SAE activations centered on the middle of each statement
    logging.info(f"Extracting SAE activations for '{set1_name}' statements...")
    set1_activations, set1_middles = extract_sae_activations_from_middle(
        model, sae_model, tokenizer, set1_statements,
        args.token_distance, args.target_layer
    )

    logging.info(f"Extracting SAE activations for '{set2_name}' statements...")
    set2_activations, set2_middles = extract_sae_activations_from_middle(
        model, sae_model, tokenizer, set2_statements,
        args.token_distance, args.target_layer
    )

    if len(set1_activations) == 0 or len(set2_activations) == 0:
        logging.error("No valid activations extracted from one or both sets. Cannot proceed with analysis.")
        return

    logging.info(f"Extracted activations with shapes: {set1_activations.shape}, {set2_activations.shape}")

    # Run t-tests on ALL features (modified approach)
    logging.info("Running t-tests to compare activations between sets on ALL features...")
    p_values, t_values = run_t_tests(set1_activations, set2_activations, feature_indices=None)

    # Select top features based on significance at middle position
    middle_pos = args.token_distance
    if middle_pos < p_values.shape[0]:
        middle_p_values = p_values[middle_pos, :]
        # Get indices of features sorted by p-value (most significant first)
        significance_ranking = np.argsort(middle_p_values)
        top_k_features = significance_ranking[:args.top_k]
        logging.info(f"Selected top {len(top_k_features)} features by statistical significance at middle position")
    else:
        # Fallback if middle position is invalid
        logging.warning(f"Middle position {middle_pos} out of bounds, using overall ranking")
        overall_p_values = np.mean(p_values, axis=0)
        significance_ranking = np.argsort(overall_p_values)
        top_k_features = significance_ranking[:args.top_k]
        logging.info(f"Selected top {len(top_k_features)} features by overall statistical significance")

    # Plot p-value heatmap (using the selected features for display)
    logging.info("Creating p-value heatmap...")
    # Extract p-values and t-values for the selected features for visualization
    selected_p_values = p_values[:, top_k_features]
    selected_t_values = t_values[:, top_k_features]
    
    plot_p_value_heatmap(
        selected_p_values, selected_t_values, top_k_features,
        output_dir / "p_value_heatmap.png",
        token_distance=args.token_distance,
        statements_dict=statements_dict,
        category_names=category_names,
        tokenizer=tokenizer
    )

    # Plot p-value distribution (using all features)
    logging.info("Creating p-value distribution plot...")
    plot_p_value_distribution(
        p_values, output_dir / "p_value_distribution.png",
        token_distance=args.token_distance
    )

    # Save top features and their statistics
    with open(output_dir / "top_features.json", "w") as f:
        json.dump(convert_numpy_types({
            "top_feature_indices": top_k_features.tolist(),
            "t_values": selected_t_values.tolist(),
            "p_values": selected_p_values.tolist()
        }), f, indent=2)
    logging.info("Top features, t-values, and p-values saved to top_features.json")

    # RSA Analysis (using top 50 features by significance)
    logging.info("Running RSA analysis...")

    # Use top 50 features by significance for RSA
    rsa_k = min(50, len(top_k_features))
    rsa_features = top_k_features[:rsa_k]
    logging.info(f"Using top {len(rsa_features)} features by significance for RSA analysis")

    # Compute within-set correlations for set1 statements
    logging.info(f"Computing within-set correlations for '{set1_name}' statements...")
    set1_window_corrs, set1_position_corrs = compute_within_correlations(
        set1_activations, rsa_features
    )

    # Compute across-set correlations (set1 vs set2)
    logging.info(f"Computing across-set correlations ('{set1_name}' vs '{set2_name}')...")
    across_window_corrs, across_position_corrs = compute_across_correlations(
        set1_activations, set2_activations, rsa_features
    )

    # Run bootstrap significance tests
    logging.info(f"Running bootstrap significance tests with {args.n_bootstraps} samples...")

    # Window-level test
    window_p_value, window_effect = bootstrap_significance_test(
        set1_window_corrs, across_window_corrs, args.n_bootstraps
    )
    logging.info(f"Window RSA: p-value={window_p_value:.4f}, effect={window_effect:.4f}")

    # Position-level tests
    position_p_values = []
    position_effects = []
    # Ensure both set1_position_corrs and across_position_corrs have the same length
    if len(set1_position_corrs) == len(across_position_corrs):
        for pos in tqdm(range(len(set1_position_corrs)), desc="Bootstrap tests by position"):
            pos_p_value, pos_effect = bootstrap_significance_test(
                set1_position_corrs[pos], across_position_corrs[pos], args.n_bootstraps
            )
            position_p_values.append(pos_p_value)
            position_effects.append(pos_effect)
            logging.debug(f"Position {pos - args.token_distance}: p={pos_p_value:.4f}, effect={pos_effect:.4f}")
    else:
         logging.error("Mismatch in number of positions for within and across correlations. Skipping position RSA tests.")
         position_p_values = [1.0] * (2 * args.token_distance + 1) # Fill with non-significant results
         position_effects = [np.nan] * (2 * args.token_distance + 1) # Fill with NaN effects

    # Plot RSA results
    logging.info("Creating RSA results plot...")
    plot_rsa_results(
        window_p_value, window_effect,
        position_p_values, position_effects,
        output_dir / "rsa_results.png",
        token_distance=args.token_distance,
        statements_dict=statements_dict,
        category_names=category_names,
        tokenizer=tokenizer
    )

    # Save RSA results
    with open(output_dir / "rsa_results.json", "w") as f:
        json.dump(convert_numpy_types({
            "window_p_value": window_p_value,
            "window_effect_size": window_effect,
            "position_p_values": position_p_values,
            "position_effect_sizes": position_effects,
            "rsa_feature_indices": rsa_features.tolist()
        }), f, indent=2)
    logging.info("RSA results saved to rsa_results.json")

    # Save tokens for example display reference (small subset)
    logging.info("Saving token information for examples...")
    token_info = {
        f"{set1_name}_examples": [],
        f"{set2_name}_examples": []
    }

    # Take a small sample of original statements to show tokenization
    sample_set1 = random.sample(set1_statements, min(10, len(set1_statements)))
    sample_set2 = random.sample(set2_statements, min(10, len(set2_statements)))

    for statement in sample_set1:
        tokens = tokenizer.encode(statement, add_special_tokens=False)
        middle_pos = len(tokens) // 2
        window_start = max(0, middle_pos - args.token_distance)
        window_end = min(len(tokens), middle_pos + args.token_distance + 1)
        window_tokens_ids = tokens[window_start:window_end]
        window_tokens_str = tokenizer.convert_ids_to_tokens(window_tokens_ids, skip_special_tokens=False) # Show raw tokens

        token_info[f"{set1_name}_examples"].append({
            "statement": statement,
            "raw_tokens_ids": tokens,
            "raw_middle_position": middle_pos,
            "raw_window_tokens": window_tokens_str
        })

    for statement in sample_set2:
        tokens = tokenizer.encode(statement, add_special_tokens=False)
        middle_pos = len(tokens) // 2
        window_start = max(0, middle_pos - args.token_distance)
        window_end = min(len(tokens), middle_pos + args.token_distance + 1)
        window_tokens_ids = tokens[window_start:window_end]
        window_tokens_str = tokenizer.convert_ids_to_tokens(window_tokens_ids, skip_special_tokens=False) # Show raw tokens

        token_info[f"{set2_name}_examples"].append({
            "statement": statement,
            "raw_tokens_ids": tokens,
            "raw_middle_position": middle_pos,
            "raw_window_tokens": window_tokens_str
        })

    with open(output_dir / "token_info.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(token_info), f, indent=2)
    logging.info("Example token information saved to token_info.json")

    logging.info(f"Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()