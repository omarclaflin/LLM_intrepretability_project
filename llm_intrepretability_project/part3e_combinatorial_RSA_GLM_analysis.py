"""
Combinatorial RSA Feature Selection Analysis

Runs 6 different analysis combinations:
1. RSA values + SAE + Discriminative RSA
2. RSA values + SAE + Categorical RSA  
3. RSA values + Raw middle layer + Discriminative RSA
4. RSA values + Raw middle layer + Categorical RSA
5. Raw activation values + SAE (no RSA)
6. Raw activation values + Raw middle layer (no RSA)

For each combination:
- Extract activations (SAE or raw middle layer)
- Optionally compute RSA discriminability scores
- Select top 5 features (by RSA discriminability or max activation)
- Train L1-regularized classifier on 5 selected features
- Find top 25 class predictions for interpretation
- Run weighted clamping interventions using classifier weights
"""

import argparse
import asyncio
import torch
import numpy as np
from transformers import LlamaTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import time
import datetime
import logging
import os
import requests
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import ttest_ind, pearsonr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.number):
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        else:
            return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(i) for i in obj]
    return obj

class SparseAutoencoder(torch.nn.Module):
    """Simple Sparse Autoencoder module."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return features, reconstruction

def load_statements_from_json(json_file, category_keys):
    """Load statements from JSON file using specified keys."""
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

    return statements, list(category_keys)

def extract_activations_from_start(model, sae_model, tokenizer, texts, token_distance, target_layer=16, use_sae=True):
    """
    Extract activations from start of each text.
    
    Args:
        use_sae: If True, extract SAE features. If False, extract raw hidden states.
        
    Returns:
        activations: (n_texts, n_features) - averaged across token positions
        valid_indices: Which texts were successfully processed
    """
    device = next(model.parameters()).device
    all_activations = []
    valid_indices = []

    # Check text validity
    for i, text in enumerate(tqdm(texts, desc="Checking text lengths")):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        seq_len = torch.sum(inputs["attention_mask"][0]).item()
        
        start_pos = 1 if tokenizer.bos_token else 0
        if start_pos < seq_len:
            valid_indices.append(i)
        else:
            logging.debug(f"Skipping text '{text[:50]}...' due to insufficient content tokens")

    logging.info(f"Found {len(valid_indices)} texts with sufficient content tokens")

    # Extract activations
    for i, idx in enumerate(tqdm(valid_indices, desc="Extracting activations")):
        text = texts[idx]
        inputs = tokenizer(text, return_tensors="pt").to(device)
        seq_len = torch.sum(inputs["attention_mask"][0]).item()

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[target_layer]

            # Extract window from start
            start_pos = 1 if tokenizer.bos_token else 0
            end_pos = min(start_pos + token_distance, seq_len)
            
            window = hidden_states[0, start_pos:end_pos]
            
            # Pad with zeros if needed
            if window.shape[0] < token_distance:
                padding = torch.zeros((token_distance - window.shape[0], window.shape[1]), device=window.device)
                window = torch.cat([window, padding], dim=0)
            
            if use_sae:
                # Process through SAE
                features, _ = sae_model(window.to(sae_model.encoder[0].weight.dtype))
                # Average across token positions
                avg_features = torch.mean(features, dim=0).cpu().numpy()
            else:
                # Use raw hidden states, average across token positions
                avg_features = torch.mean(window, dim=0).cpu().numpy()
            
            all_activations.append(avg_features)

    if all_activations:
        all_activations = np.stack(all_activations)
        logging.info(f"Extracted activations shape: {all_activations.shape}")
    else:
        logging.error("No valid activations extracted!")
        all_activations = np.array([])

    return all_activations, valid_indices

def compute_discriminative_rsa(set1_activations, set2_activations, token_distance, target_layer, model, sae_model, tokenizer, texts_set1, texts_set2, valid_indices_1, valid_indices_2, use_sae=True):
    """
    Compute discriminative RSA: within category 1 vs across category.
    
    Returns RSA discriminability scores per feature.
    """
    logging.info("Computing discriminative RSA scores...")
    
    # First extract full token-level activations (not averaged)
    all_token_activations_1 = []
    all_token_activations_2 = []
    
    device = next(model.parameters()).device
    
    # Extract token-level activations for set1
    for idx in tqdm(valid_indices_1, desc="Extracting token activations set1"):
        text = texts_set1[idx]
        inputs = tokenizer(text, return_tensors="pt").to(device)
        seq_len = torch.sum(inputs["attention_mask"][0]).item()

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[target_layer]

            start_pos = 1 if tokenizer.bos_token else 0
            end_pos = min(start_pos + token_distance, seq_len)
            window = hidden_states[0, start_pos:end_pos]
            
            if window.shape[0] < token_distance:
                padding = torch.zeros((token_distance - window.shape[0], window.shape[1]), device=window.device)
                window = torch.cat([window, padding], dim=0)
            
            if use_sae:
                features, _ = sae_model(window.to(sae_model.encoder[0].weight.dtype))
                all_token_activations_1.append(features.cpu().numpy())
            else:
                all_token_activations_1.append(window.cpu().numpy())
    
    # Extract token-level activations for set2
    for idx in tqdm(valid_indices_2, desc="Extracting token activations set2"):
        text = texts_set2[idx]
        inputs = tokenizer(text, return_tensors="pt").to(device)
        seq_len = torch.sum(inputs["attention_mask"][0]).item()

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[target_layer]

            start_pos = 1 if tokenizer.bos_token else 0
            end_pos = min(start_pos + token_distance, seq_len)
            window = hidden_states[0, start_pos:end_pos]
            
            if window.shape[0] < token_distance:
                padding = torch.zeros((token_distance - window.shape[0], window.shape[1]), device=window.device)
                window = torch.cat([window, padding], dim=0)
            
            if use_sae:
                features, _ = sae_model(window.to(sae_model.encoder[0].weight.dtype))
                all_token_activations_2.append(features.cpu().numpy())
            else:
                all_token_activations_2.append(window.cpu().numpy())
    
    token_activations_1 = np.stack(all_token_activations_1)  # (n_samples_1, token_distance, n_features)
    token_activations_2 = np.stack(all_token_activations_2)  # (n_samples_2, token_distance, n_features)
    
    n_samples_1, _, n_features = token_activations_1.shape
    n_samples_2 = token_activations_2.shape[0]
    
    logging.info(f"RSA computation setup: {n_samples_1} samples set1, {n_samples_2} samples set2, {n_features} features")
    logging.info(f"Token activations ranges - Set1: [{np.min(token_activations_1):.3f}, {np.max(token_activations_1):.3f}]")
    logging.info(f"Token activations ranges - Set2: [{np.min(token_activations_2):.3f}, {np.max(token_activations_2):.3f}]")
    
    rsa_scores = np.zeros(n_features)
    
    # Vectorized RSA computation per feature
    for feature_idx in tqdm(range(n_features), desc="Computing discriminative RSA"):
        # Extract all token patterns for this feature
        patterns_1 = token_activations_1[:, :, feature_idx]  # (n_samples_1, token_distance)
        patterns_2 = token_activations_2[:, :, feature_idx]  # (n_samples_2, token_distance)
        
        # Combine all patterns
        all_patterns = np.vstack([patterns_1, patterns_2])  # (n_samples_1 + n_samples_2, token_distance)
        
        try:
            # Vectorized correlation matrix
            # Check for sufficient variance first
            if np.std(all_patterns) < 1e-9:
                logging.debug(f"Zero variance in patterns for feature {feature_idx}")
                rsa_scores[feature_idx] = 0.0
                continue
                
            corr_matrix = np.corrcoef(all_patterns)  # (total_samples, total_samples)
            
            # Check for NaN in correlation matrix
            if np.isnan(corr_matrix).any():
                logging.debug(f"NaN correlations for feature {feature_idx}")
                rsa_scores[feature_idx] = 0.0
                continue
            
            # Extract correlation sets
            within_cat1_mask = np.triu(np.ones((n_samples_1, n_samples_1)), k=1).astype(bool)
            within_cat1 = corr_matrix[:n_samples_1, :n_samples_1][within_cat1_mask]
            
            across = corr_matrix[:n_samples_1, n_samples_1:].flatten()
            
            # Check for sufficient valid correlations
            if len(within_cat1) == 0 or len(across) == 0:
                rsa_scores[feature_idx] = 0.0
                continue
            
            # Apply Fisher z-transform with better bounds checking
            within_cat1_clipped = np.clip(within_cat1, -0.999, 0.999)
            across_clipped = np.clip(across, -0.999, 0.999)
            
            within_cat1_z = np.arctanh(1 - within_cat1_clipped)
            across_z = np.arctanh(1 - across_clipped)
            
            # Remove any NaN or infinite values
            within_cat1_z = within_cat1_z[np.isfinite(within_cat1_z)]
            across_z = across_z[np.isfinite(across_z)]
            
            # T-test: within category 1 vs across categories
            if len(within_cat1_z) > 1 and len(across_z) > 1:
                t_stat, _ = ttest_ind(within_cat1_z, across_z)
                if np.isfinite(t_stat):
                    rsa_scores[feature_idx] = t_stat
                else:
                    rsa_scores[feature_idx] = 0.0
            else:
                rsa_scores[feature_idx] = 0.0
                
        except Exception as e:
            logging.debug(f"Error computing RSA for feature {feature_idx}: {e}")
            rsa_scores[feature_idx] = 0.0
    
    # Z-score the t-values (this is the final stored value)
    finite_scores = rsa_scores[np.isfinite(rsa_scores)]
    if len(finite_scores) > 1:
        mean_t = np.mean(finite_scores)
        std_t = np.std(finite_scores)
        if std_t > 1e-9:
            rsa_scores = (rsa_scores - mean_t) / std_t
    
    # Report statistics
    n_finite = np.sum(np.isfinite(rsa_scores))
    n_nonzero = np.sum(rsa_scores != 0.0)
    logging.info(f"RSA statistics: {n_finite}/{n_features} finite, {n_nonzero}/{n_features} non-zero")
    logging.info(f"Discriminative RSA computed. Range: [{np.min(rsa_scores):.3f}, {np.max(rsa_scores):.3f}]")
    
    return rsa_scores

def compute_categorical_rsa(set1_activations, set2_activations, token_distance, target_layer, model, sae_model, tokenizer, texts_set1, texts_set2, valid_indices_1, valid_indices_2, use_sae=True):
    """
    Compute categorical RSA: within category1 vs within category2.
    FIXED VERSION - NO NESTED SAMPLE LOOPS
    """
    logging.info("Computing categorical RSA scores...")
    
    # Extract token-level activations (same as discriminative)
    all_token_activations_1 = []
    all_token_activations_2 = []
    
    device = next(model.parameters()).device
    
    # Extract token-level activations for set1
    for idx in tqdm(valid_indices_1, desc="Extracting token activations set1"):
        text = texts_set1[idx]
        inputs = tokenizer(text, return_tensors="pt").to(device)
        seq_len = torch.sum(inputs["attention_mask"][0]).item()

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[target_layer]

            start_pos = 1 if tokenizer.bos_token else 0
            end_pos = min(start_pos + token_distance, seq_len)
            window = hidden_states[0, start_pos:end_pos]
            
            if window.shape[0] < token_distance:
                padding = torch.zeros((token_distance - window.shape[0], window.shape[1]), device=window.device)
                window = torch.cat([window, padding], dim=0)
            
            if use_sae:
                features, _ = sae_model(window.to(sae_model.encoder[0].weight.dtype))
                all_token_activations_1.append(features.cpu().numpy())
            else:
                all_token_activations_1.append(window.cpu().numpy())
    
    # Extract token-level activations for set2
    for idx in tqdm(valid_indices_2, desc="Extracting token activations set2"):
        text = texts_set2[idx]
        inputs = tokenizer(text, return_tensors="pt").to(device)
        seq_len = torch.sum(inputs["attention_mask"][0]).item()

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[target_layer]

            start_pos = 1 if tokenizer.bos_token else 0
            end_pos = min(start_pos + token_distance, seq_len)
            window = hidden_states[0, start_pos:end_pos]
            
            if window.shape[0] < token_distance:
                padding = torch.zeros((token_distance - window.shape[0], window.shape[1]), device=window.device)
                window = torch.cat([window, padding], dim=0)
            
            if use_sae:
                features, _ = sae_model(window.to(sae_model.encoder[0].weight.dtype))
                all_token_activations_2.append(features.cpu().numpy())
            else:
                all_token_activations_2.append(window.cpu().numpy())
    
    token_activations_1 = np.stack(all_token_activations_1)
    token_activations_2 = np.stack(all_token_activations_2)
    
    n_samples_1, _, n_features = token_activations_1.shape
    n_samples_2 = token_activations_2.shape[0]
    
    logging.info(f"Categorical RSA setup: {n_samples_1} samples set1, {n_samples_2} samples set2, {n_features} features")
    logging.info(f"Token activations ranges - Set1: [{np.min(token_activations_1):.3f}, {np.max(token_activations_1):.3f}]")
    logging.info(f"Token activations ranges - Set2: [{np.min(token_activations_2):.3f}, {np.max(token_activations_2):.3f}]")
    
    rsa_scores = np.zeros(n_features)
    
    # VECTORIZED: Single loop through features (50,000 iterations)
    for feature_idx in tqdm(range(n_features), desc="Computing categorical RSA"):
        # Extract all token patterns for this feature
        patterns_1 = token_activations_1[:, :, feature_idx]  # (n_samples_1, token_distance)
        patterns_2 = token_activations_2[:, :, feature_idx]  # (n_samples_2, token_distance)
        
        # Combine all patterns (SAME AS DISCRIMINATIVE)
        all_patterns = np.vstack([patterns_1, patterns_2])  # (total_samples, token_distance)
        
        try:
            # Check for sufficient variance first
            if np.std(all_patterns) < 1e-9:
                logging.debug(f"Zero variance in patterns for feature {feature_idx}")
                rsa_scores[feature_idx] = 0.0
                continue
                
            # Single correlation matrix (SAME AS DISCRIMINATIVE)
            corr_matrix = np.corrcoef(all_patterns)  # (total_samples, total_samples)
            
            # Check for NaN in correlation matrix
            if np.isnan(corr_matrix).any():
                logging.debug(f"NaN correlations for feature {feature_idx}")
                rsa_scores[feature_idx] = 0.0
                continue
            
            # Extract within-category correlations from SAME matrix
            within_cat1_mask = np.triu(np.ones((n_samples_1, n_samples_1)), k=1).astype(bool)
            within_cat1 = corr_matrix[:n_samples_1, :n_samples_1][within_cat1_mask]
            
            within_cat2_mask = np.triu(np.ones((n_samples_2, n_samples_2)), k=1).astype(bool)
            within_cat2 = corr_matrix[n_samples_1:, n_samples_1:][within_cat2_mask]
            
            # Check for sufficient valid correlations
            if len(within_cat1) == 0 or len(within_cat2) == 0:
                rsa_scores[feature_idx] = 0.0
                continue
            
            # Apply Fisher z-transform
            within_cat1_clipped = np.clip(within_cat1, -0.999, 0.999)
            within_cat2_clipped = np.clip(within_cat2, -0.999, 0.999)
            
            within_cat1_z = np.arctanh(1 - within_cat1_clipped)
            within_cat2_z = np.arctanh(1 - within_cat2_clipped)
            
            # Remove any NaN or infinite values
            within_cat1_z = within_cat1_z[np.isfinite(within_cat1_z)]
            within_cat2_z = within_cat2_z[np.isfinite(within_cat2_z)]
            
            # T-test: within category1 vs within category2
            if len(within_cat1_z) > 1 and len(within_cat2_z) > 1:
                t_stat, _ = ttest_ind(within_cat1_z, within_cat2_z)
                if np.isfinite(t_stat):
                    rsa_scores[feature_idx] = t_stat
                else:
                    rsa_scores[feature_idx] = 0.0
            else:
                rsa_scores[feature_idx] = 0.0
                
        except Exception as e:
            logging.debug(f"Error computing categorical RSA for feature {feature_idx}: {e}")
            rsa_scores[feature_idx] = 0.0
    
    # Z-score the t-values
    finite_scores = rsa_scores[np.isfinite(rsa_scores)]
    if len(finite_scores) > 1:
        mean_t = np.mean(finite_scores)
        std_t = np.std(finite_scores)
        if std_t > 1e-9:
            rsa_scores = (rsa_scores - mean_t) / std_t
    
    # Report statistics
    n_finite = np.sum(np.isfinite(rsa_scores))
    n_nonzero = np.sum(rsa_scores != 0.0)
    logging.info(f"RSA statistics: {n_finite}/{n_features} finite, {n_nonzero}/{n_features} non-zero")
    logging.info(f"Categorical RSA computed. Range: [{np.min(rsa_scores):.3f}, {np.max(rsa_scores):.3f}]")
    
    return rsa_scores

def plot_rsa_distribution(rsa_scores, output_path, title_suffix=""):
    """Plot distribution of RSA z-scores."""
    plt.figure(figsize=(12, 8))
    
    # RSA scores are now (n_features,) instead of (n_samples, n_features)
    finite_scores = rsa_scores[np.isfinite(rsa_scores)]
    
    plt.subplot(2, 2, 1)
    plt.hist(finite_scores, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Z-scored T-values')
    plt.ylabel('Frequency')
    plt.title(f'RSA Z-score Distribution {title_suffix}')
    
    plt.subplot(2, 2, 2)
    # QQ plot
    sorted_scores = np.sort(finite_scores)
    n = len(sorted_scores)
    if n > 1:
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
        plt.scatter(theoretical_quantiles, sorted_scores, alpha=0.6)
        plt.plot(theoretical_quantiles, theoretical_quantiles, 'r--')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title(f'Q-Q Plot {title_suffix}')
    
    plt.subplot(2, 2, 3)
    # Cumulative distribution
    plt.hist(finite_scores, bins=50, cumulative=True, density=True, alpha=0.7, color='green')
    plt.xlabel('Z-scored T-values')
    plt.ylabel('Cumulative Density')
    plt.title(f'Cumulative Distribution {title_suffix}')
    
    plt.subplot(2, 2, 4)
    # Top and bottom features
    top_indices = np.argsort(rsa_scores)[-20:]
    bottom_indices = np.argsort(rsa_scores)[:20]
    
    plt.barh(range(20), rsa_scores[top_indices], alpha=0.7, color='red', label='Top 20')
    plt.barh(range(20, 40), rsa_scores[bottom_indices], alpha=0.7, color='blue', label='Bottom 20')
    plt.xlabel('Z-scored T-values')
    plt.ylabel('Feature Rank')
    plt.title(f'Top/Bottom Features {title_suffix}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def select_top_features(activations_1, activations_2, rsa_scores=None, num_features=5):
    """
    Select top features either by RSA discriminability or maximum activation.
    
    Args:
        activations_1, activations_2: Activation matrices (n_samples, n_features)
        rsa_scores: Optional RSA scores for selection (n_features,)
        num_features: Number of features to select
        
    Returns:
        selected_indices: Indices of selected features
        selection_method: Description of selection method used
    """
    if rsa_scores is not None:
        # Use RSA discriminability for feature selection
        # RSA scores are now (n_features,) shape
        selected_indices = np.argsort(rsa_scores)[-num_features:]
        selection_method = "RSA_discriminability"
        
        logging.info(f"Selected features by RSA: {selected_indices}")
        logging.info(f"RSA scores for selected features: {rsa_scores[selected_indices]}")
        
    else:
        # Use maximum activation for feature selection
        max_act_1 = np.max(activations_1, axis=0)
        max_act_2 = np.max(activations_2, axis=0)
        combined_max = np.maximum(max_act_1, max_act_2)
        
        selected_indices = np.argsort(combined_max)[-num_features:]
        selection_method = "max_activation"
        
        logging.info(f"Selected features by max activation: {selected_indices}")
        logging.info(f"Max activations for selected features: {combined_max[selected_indices]}")
    
    return selected_indices, selection_method

def train_linear_classifier(features, labels, test_size=0.2, random_state=42):
    """Train L1-regularized logistic regression classifier."""
    logging.info("Training L1-regularized linear classifier...")
    logging.info(f"Input features shape: {features.shape}")
    logging.info(f"Labels shape: {labels.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier with L1 regularization
    classifier = LogisticRegression(penalty='l1', solver='liblinear', random_state=random_state, max_iter=1000)
    classifier.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = classifier.predict(X_train_scaled)
    test_pred = classifier.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    logging.info(f"Classifier performance - Train: {train_acc:.3f}, Test: {test_acc:.3f}")
    logging.info(f"Classifier weights: {classifier.coef_[0]}")
    
    return {
        'classifier': classifier,
        'scaler': scaler,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'classification_report': classification_report(y_test, test_pred, output_dict=True),
        'weights': classifier.coef_[0]  # Shape: (n_selected_features,)
    }

def find_top_classifier_predictions(model, sae_model, tokenizer, selected_indices, classifier_results, use_sae, target_layer, token_distance, batch_size=2048, num_examples=25):
    """
    Find top examples from Wikipedia training chunks with highest classifier prediction probability.
    Uses same processing and same data chunks as SAE training script.
    """
    logging.info("Finding top classifier predictions from Wikipedia training chunks...")
    
    # Same target chunks that SAE was trained on
    target_chunks = [
        (14000, 15000), (16000, 17000), (66000, 67000), (111000, 112000), (147000, 148000),
        (165000, 166000), (182000, 183000), (187000, 188000), (251000, 252000), (290000, 291000),
        (295000, 296000), (300000, 301000), (313000, 314000), (343000, 344000), (366000, 367000),
        (367000, 368000), (380000, 381000), (400000, 401000), (407000, 408000), (420000, 421000),
        (440000, 441000), (443000, 444000), (479000, 480000), (480000, 481000), (514000, 515000),
        (523000, 524000), (552000, 553000), (579000, 580000), (583000, 584000), (616000, 617000),
        (659000, 660000), (663000, 664000), (690000, 691000), (810000, 811000), (824000, 825000),
        (876000, 877000), (881000, 882000), (908000, 909000), (969000, 970000), (970000, 971000),
        (984000, 985000), (990000, 991000), (995000, 996000), (997000, 998000), (1000000, 1001000),
        (1024000, 1025000), (1099000, 1100000), (1127000, 1128000), (1163000, 1164000), (1182000, 1183000),
        (1209000, 1210000), (1253000, 1254000), (1266000, 1267000), (1270000, 1271000), (1276000, 1277000),
        (1290000, 1291000), (1307000, 1308000), (1326000, 1327000), (1345000, 1346000), (1359000, 1360000),
        (1364000, 1365000), (1367000, 1368000), (1385000, 1386000), (1391000, 1392000), (1468000, 1469000),
        (1508000, 1509000), (1523000, 1524000), (1539000, 1540000), (1574000, 1575000), (1583000, 1584000),
        (1590000, 1591000), (1593000, 1594000), (1599000, 1600000), (1627000, 1628000), (1679000, 1680000),
        (1690000, 1691000), (1691000, 1692000), (1782000, 1783000), (1788000, 1789000)
    ]
    
    # Build set of all target indices
    target_indices = set()
    for start, end in target_chunks:
        target_indices.update(range(start, end))
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    device = next(model.parameters()).device
    
    # Get classifier components
    scaler = classifier_results['scaler']
    classifier = classifier_results['classifier']
    
    # Collect all examples with their prediction probabilities
    all_examples = []
    batch_texts = []
    processed_count = 0
    
    for idx, sample in enumerate(tqdm(dataset, desc="Processing Wikipedia chunks")):
        # Skip if not in target chunks (same as SAE script)
        if idx not in target_indices:
            continue
            
        text = sample["text"]
        if not text.strip():  # Skip empty texts
            continue
            
        batch_texts.append(text)
        
        # Process in batches like SAE script
        if len(batch_texts) >= batch_size:
            try:
                batch_predictions = process_batch_for_classification(
                    model, sae_model, tokenizer, batch_texts, selected_indices, 
                    scaler, classifier, use_sae, target_layer, token_distance)
                
                # Add to results
                for text, prob in zip(batch_texts, batch_predictions):
                    if prob is not None:  # Valid prediction
                        all_examples.append({
                            'text': text,
                            'prediction_prob': float(prob),
                            'example_index': processed_count
                        })
                        processed_count += 1
                
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
            
            batch_texts = []
    
    # Process remaining batch
    if batch_texts:
        try:
            batch_predictions = process_batch_for_classification(
                model, sae_model, tokenizer, batch_texts, selected_indices,
                scaler, classifier, use_sae, target_layer, token_distance)
            
            for text, prob in zip(batch_texts, batch_predictions):
                if prob is not None:
                    all_examples.append({
                        'text': text,
                        'prediction_prob': float(prob),
                        'example_index': processed_count
                    })
                    processed_count += 1
        except Exception as e:
            logging.error(f"Error processing final batch: {e}")
    
    logging.info(f"Processed {len(all_examples)} Wikipedia examples from target chunks")
    
    if not all_examples:
        logging.error("No valid examples processed from Wikipedia chunks")
        return []
    
    # Sort by prediction probability and return top N
    all_examples.sort(key=lambda x: x['prediction_prob'], reverse=True)
    
    logging.info(f"Top prediction probabilities: {[ex['prediction_prob'] for ex in all_examples[:5]]}")
    
    return all_examples[:num_examples]

def process_batch_for_classification(model, sae_model, tokenizer, batch_texts, selected_indices, scaler, classifier, use_sae, target_layer, token_distance):
    """Process a batch of texts and return classifier prediction probabilities."""
    device = next(model.parameters()).device
    batch_predictions = []
    
    for text in batch_texts:
        try:
            # Tokenize - same as SAE script
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            seq_len = torch.sum(inputs["attention_mask"][0]).item()
            
            # Check if sufficient tokens
            start_pos = 1 if tokenizer.bos_token else 0
            if start_pos >= seq_len:
                batch_predictions.append(None)
                continue
            
            # Get activations - same as SAE script
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[target_layer]
                
                # Extract window from start
                end_pos = min(start_pos + token_distance, seq_len)
                window = hidden_states[0, start_pos:end_pos]
                
                # Pad if needed
                if window.shape[0] < token_distance:
                    padding = torch.zeros((token_distance - window.shape[0], window.shape[1]), device=window.device)
                    window = torch.cat([window, padding], dim=0)
                
                if use_sae:
                    # Process through SAE
                    features, _ = sae_model(window.to(sae_model.encoder[0].weight.dtype))
                    # Average across tokens
                    avg_features = torch.mean(features, dim=0).cpu().numpy()
                else:
                    # Use raw hidden states
                    avg_features = torch.mean(window, dim=0).cpu().numpy()
                
                # Extract selected features
                selected_features = avg_features[selected_indices].reshape(1, -1)  # (1, num_features)
                
                # Scale and predict
                selected_features_scaled = scaler.transform(selected_features)
                prediction_probs = classifier.predict_proba(selected_features_scaled)
                
                # Get probability for positive class (class 1)
                class1_prob = prediction_probs[0, 1]
                batch_predictions.append(class1_prob)
                
        except Exception as e:
            logging.debug(f"Error processing single text: {e}")
            batch_predictions.append(None)
    
    return batch_predictions

def load_identification_prompt():
    """Load the identification prompt template from existing prompts folder."""
    prompt_path = Path("feature_identification_prompt.txt")
    if not prompt_path.exists():
        prompt_path = Path("prompts/feature_identification_prompt.txt")

    if not prompt_path.exists():
        logging.warning(f"Prompt template not found at {prompt_path}. Using default.")
        return """I'll analyze examples where a specific weighted combination of neural network features shows high prediction confidence.
Please help identify the pattern this combination might be detecting.

{examples}

{separator}

Based on these examples, what's a concise description of the semantic pattern this weighted feature combination learned?
Format your response as:
Pattern: [concise description]
EXPLANATION: [brief explanation]"""

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def load_scoring_prompt():
    """Load the scoring prompt template."""
    prompt_path = Path("scoring_prompt.txt")
    if not prompt_path.exists():
        prompt_path = Path("prompts/scoring_prompt.txt")

    if not prompt_path.exists():
        logging.warning(f"Scoring prompt template not found at {prompt_path}. Using default.")
        return """You will rate how well a feature pattern matches text.
You must respond with ONLY a single digit (0, 1, 2, or 3) and nothing else.

RATING SCALE:
0 – The feature is completely irrelevant throughout the context.
1 – The feature is related to the context, but not near the highlighted text or only vaguely related.
2 – The feature is only loosely related to the highlighted text or related to the context near the highlighted text.
3 – The feature cleanly identifies the activating text.

Feature Pattern: "{pattern}"
Text: {text}
Rating: ?"""

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

async def identify_weighted_pattern(examples, claude_api_key, identification_prompt, output_dir, combination_name, num_examples=25):
    """Use Claude API to identify pattern in top prediction examples."""
    if not claude_api_key:
        return "Pattern identification skipped (no API key)"
    
    # Format examples
    formatted_examples = []
    for i, example in enumerate(examples[:num_examples]):  # Use num_examples instead of hardcoded 10
        text_content = example['text'].strip()
        prob = example['prediction_prob']
        
        if text_content:
            formatted_examples.append(f"Example {i+1} (Prob: {prob:.3f}): {text_content}")
    
    if not formatted_examples:
        return "No valid examples for pattern identification"
    
    separator = '-' * 40
    
    try:
        claude_prompt = identification_prompt.format(
            examples='\n'.join(formatted_examples),
            separator=separator
        )
    except KeyError as e:
        return f"Error: Prompt template missing placeholder {e}"
    
    # Save prompt
    combo_dir = output_dir / combination_name
    combo_dir.mkdir(exist_ok=True)
    with open(combo_dir / "claude_pattern_prompt.txt", "w", encoding="utf-8") as f:
        f.write(claude_prompt)
    
    # Call Claude API
    api_url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": claude_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": claude_prompt}]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        
        response_data = response.json()
        claude_response = response_data["content"][0]["text"]
        
        # Save response
        with open(combo_dir / "claude_pattern_response.txt", "w", encoding="utf-8") as f:
            f.write(claude_response)
        
        # Parse pattern
        pattern_parts = claude_response.split("EXPLANATION:", 1)
        concise_pattern = pattern_parts[0].strip() if pattern_parts else "Unknown pattern"
        
        with open(combo_dir / "pattern.txt", "w", encoding="utf-8") as f:
            f.write(claude_response)
        
        return concise_pattern
        
    except Exception as e:
        logging.error(f"Error calling Claude API: {e}")
        return f"Error identifying pattern: {str(e)}"

async def score_pattern_match(pattern, examples, claude_api_key, scoring_prompt, output_dir, combination_name, num_examples=25):
    """Use Claude API to score how well the pattern matches each example."""
    if not claude_api_key:
        return []
    
    scores = []
    combo_dir = output_dir / combination_name
    combo_dir.mkdir(exist_ok=True)
    
    # Score each example
    for i, example in enumerate(examples[:num_examples]):
        text_content = example['text'].strip()
        
        if not text_content:
            scores.append(None)
            continue
        
        try:
            scoring_prompt_formatted = scoring_prompt.format(
                pattern=pattern,
                text=text_content
            )
        except KeyError as e:
            logging.error(f"Error formatting scoring prompt: {e}")
            scores.append(None)
            continue
        
        # Call Claude API
        api_url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": claude_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": scoring_prompt_formatted}]
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            
            response_data = response.json()
            claude_response = response_data["content"][0]["text"].strip()
            
            # Extract numerical score
            try:
                score = int(claude_response)
                if score in [0, 1, 2, 3]:
                    scores.append(score)
                else:
                    logging.warning(f"Invalid score {score} for example {i}")
                    scores.append(None)
            except ValueError:
                logging.warning(f"Could not parse score '{claude_response}' for example {i}")
                scores.append(None)
                
        except Exception as e:
            logging.error(f"Error calling Claude API for scoring example {i}: {e}")
            scores.append(None)
    
    # Calculate score distribution
    valid_scores = [s for s in scores if s is not None]
    score_distribution = {}
    if valid_scores:
        for score_val in [0, 1, 2, 3]:
            count = valid_scores.count(score_val)
            score_distribution[score_val] = {
                'count': count,
                'percentage': (count / len(valid_scores)) * 100
            }
        score_distribution['mean'] = sum(valid_scores) / len(valid_scores)
        score_distribution['total_scored'] = len(valid_scores)
    
    # Save scoring results with distribution
    scoring_results = []
    for i, (example, score) in enumerate(zip(examples[:num_examples], scores)):
        scoring_results.append({
            'example_index': i,
            'text': example['text'],
            'prediction_prob': example['prediction_prob'],
            'pattern_match_score': score
        })
    
    full_scoring_results = {
        'individual_scores': scoring_results,
        'score_distribution': score_distribution
    }
    
    with open(combo_dir / "pattern_scoring_results.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(full_scoring_results), f, indent=2)
    
    return scores

def run_weighted_clamping(activations_1, activations_2, texts_1, texts_2, valid_indices_1, valid_indices_2,
                         selected_indices, classifier_results, model, sae_model, tokenizer, use_sae, 
                         target_layer, token_distance, output_dir, combination_name):
    """
    Run weighted clamping interventions using classifier weights.
    
    Uses the same approach as old scripts but applies weights proportionally.
    """
    logging.info("Running weighted clamping interventions...")
    
    # Get classifier weights for the selected features
    weights = classifier_results['weights']  # Shape: (num_features,)
    
    # Define clamping multipliers (same as old scripts)
    clamp_multipliers = [0.0, 2.0, 5.0]
    
    # Use a simple test prompt
    test_prompt = "Human: What's your favorite animal?\n\nAssistant:"
    
    # Get base output (unclamped)
    logging.info("Generating base output (unclamped)...")
    try:
        base_output = generate_with_weighted_clamp(
            test_prompt, None, model, sae_model, tokenizer, use_sae, 
            target_layer, token_distance, selected_indices, weights)
        
        clamp_results = {"base": base_output}
    except Exception as e:
        logging.error(f"Error generating base output: {e}")
        clamp_results = {"base": f"Error: {str(e)}"}
    
    # Generate outputs with weighted clamping
    for multiplier in clamp_multipliers:
        clamp_key = f"clamp_{multiplier:.1f}x_weights"
        logging.info(f"Generating output with {multiplier}x weighted clamping...")
        
        try:
            # Clamp values = weight * multiplier for each selected feature
            clamp_values = weights * multiplier
            output = generate_with_weighted_clamp(
                test_prompt, clamp_values, model, sae_model, tokenizer, use_sae,
                target_layer, token_distance, selected_indices, weights)
            
            clamp_results[clamp_key] = output
        except Exception as e:
            logging.error(f"Error generating clamped output ({multiplier}x): {e}")
            clamp_results[clamp_key] = f"Error: {str(e)}"
    
    # Save clamping results
    combo_dir = output_dir / combination_name
    with open(combo_dir / "clamping_results.json", "w") as f:
        json.dump(convert_numpy_types(clamp_results), f, indent=2)
    
    return clamp_results

def generate_with_weighted_clamp(text, clamp_values, model, sae_model, tokenizer, use_sae, 
                                target_layer, token_distance, selected_indices, weights):
    """
    Generate text with weighted feature clamping.
    
    Args:
        clamp_values: Values to clamp selected features to, or None for no clamping
        selected_indices: Which features (in full space) to clamp
        weights: Classifier weights (for reference)
    """
    device = next(model.parameters()).device
    
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=100).to(device)
    
    # Set up hook for weighted clamping
    hook = None
    
    if clamp_values is not None:
        def weighted_clamp_hook(module, input_tensor, output):
            if use_sae:
                # Hook on SAE - modify features
                features, reconstruction = output
                
                # Clamp selected features to specified values
                for i, feature_idx in enumerate(selected_indices):
                    if feature_idx < features.shape[-1]:
                        features[:, :, feature_idx] = clamp_values[i]
                
                return features, reconstruction
            else:
                # Hook on raw hidden states - modify selected neurons
                hidden_states = output
                
                # Clamp selected neurons to specified values  
                for i, neuron_idx in enumerate(selected_indices):
                    if neuron_idx < hidden_states.shape[-1]:
                        hidden_states[:, :, neuron_idx] = clamp_values[i]
                
                return hidden_states
        
        # Register hook
        if use_sae:
            try:
                hook = sae_model.register_forward_hook(weighted_clamp_hook)
                logging.debug("Registered weighted clamp hook on SAE")
            except Exception as e:
                logging.error(f"Failed to register SAE hook: {e}")
        else:
            # For raw hidden states, need to hook the specific layer
            try:
                target_module = model.model.layers[target_layer]
                hook = target_module.register_forward_hook(weighted_clamp_hook)
                logging.debug(f"Registered weighted clamp hook on layer {target_layer}")
            except Exception as e:
                logging.error(f"Failed to register layer hook: {e}")
    
    # Generate text
    result = ""
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
            )
        
        # Decode new tokens only
        generated_sequence = outputs.sequences[0]
        input_len = inputs['input_ids'].shape[1]
        
        if generated_sequence.shape[0] > input_len:
            result = tokenizer.decode(generated_sequence[input_len:], skip_special_tokens=True)
        else:
            result = ""
            
    except Exception as e:
        logging.error(f"Error during generation: {e}")
        result = f"Error during generation: {str(e)}"
    finally:
        # Remove hook
        if hook is not None:
            hook.remove()
            logging.debug("Removed weighted clamp hook")
    
    return result

async def run_single_combination(combination_name, use_rsa, rsa_type, use_sae, 
                                model, sae_model, tokenizer, texts_1, texts_2, 
                                valid_indices_1, valid_indices_2, args, output_dir, 
                                claude_api_key, identification_prompt, scoring_prompt):
    """Run analysis for a single combination of parameters."""
    logging.info(f"Running combination: {combination_name}")
    
    combo_dir = output_dir / combination_name
    combo_dir.mkdir(exist_ok=True)
    
    # Extract activations
    activations_1, _ = extract_activations_from_start(
        model, sae_model, tokenizer, texts_1, args.token_distance, args.target_layer, use_sae)
    activations_2, _ = extract_activations_from_start(
        model, sae_model, tokenizer, texts_2, args.token_distance, args.target_layer, use_sae)
    
    if len(activations_1) == 0 or len(activations_2) == 0:
        logging.error(f"Failed to extract activations for {combination_name}")
        return None
    
    logging.info(f"Extracted activations shapes: {activations_1.shape}, {activations_2.shape}")
    
    # Compute RSA if needed
    rsa_scores = None
    if use_rsa:
        if rsa_type == "discriminative":
            rsa_scores = compute_discriminative_rsa(
                activations_1, activations_2, args.token_distance, args.target_layer,
                model, sae_model, tokenizer, texts_1, texts_2, valid_indices_1, valid_indices_2, use_sae)
        else:  # categorical
            rsa_scores = compute_categorical_rsa(
                activations_1, activations_2, args.token_distance, args.target_layer,
                model, sae_model, tokenizer, texts_1, texts_2, valid_indices_1, valid_indices_2, use_sae)
        
        # Plot RSA distribution
        plot_rsa_distribution(rsa_scores, 
                            combo_dir / "rsa_distribution.png", f"({rsa_type})")
        
        # Save RSA scores
        with open(combo_dir / "rsa_scores.json", "w") as f:
            json.dump(convert_numpy_types({
                'rsa_scores': rsa_scores,
                'rsa_type': rsa_type
            }), f, indent=2)
    
    # Select top features
    selected_indices, selection_method = select_top_features(
        activations_1, activations_2, rsa_scores, num_features=args.num_features)
    
    # Extract selected features for classifier training
    selected_activations_1 = activations_1[:, selected_indices]  # (n_samples_1, num_features)
    selected_activations_2 = activations_2[:, selected_indices]  # (n_samples_2, num_features)
    
    # Prepare data for classifier
    all_features = np.vstack([selected_activations_1, selected_activations_2])  # (total_samples, num_features)
    all_labels = np.concatenate([np.zeros(len(selected_activations_1)), 
                                np.ones(len(selected_activations_2))])  # (total_samples,)
    
    # Train classifier
    classifier_results = train_linear_classifier(all_features, all_labels)
    
    # Export classifier
    classifier_export = {
        'classifier_state': {
            'coef_': classifier_results['classifier'].coef_.tolist(),
            'intercept_': classifier_results['classifier'].intercept_.tolist(),
            'classes_': classifier_results['classifier'].classes_.tolist()
        },
        'scaler_state': {
            'mean_': classifier_results['scaler'].mean_.tolist(),
            'scale_': classifier_results['scaler'].scale_.tolist()
        },
        'selected_indices': selected_indices.tolist(),
        'selection_method': selection_method
    }
    
    with open(combo_dir / "classifier_export.json", "w") as f:
        json.dump(convert_numpy_types(classifier_export), f, indent=2)
    
    # Save selection and classifier results (exclude non-serializable objects)
    selection_results = {
        'selected_indices': selected_indices.tolist(),
        'selection_method': selection_method,
        'classifier_performance': {
            'train_accuracy': classifier_results['train_accuracy'],
            'test_accuracy': classifier_results['test_accuracy'],
            'weights': classifier_results['weights'].tolist(),
            'classification_report': classifier_results['classification_report']
        }
    }
    
    with open(combo_dir / "feature_selection_and_classifier.json", "w") as f:
        json.dump(convert_numpy_types(selection_results), f, indent=2)
    
    # Find top Wikipedia examples with highest classifier prediction probability
    top_examples = find_top_classifier_predictions(
        model, sae_model, tokenizer, selected_indices, classifier_results, 
        use_sae, args.target_layer, args.token_distance, batch_size=args.batch_size, num_examples=args.num_examples)
    
    # Identify pattern with Claude
    pattern = await identify_weighted_pattern(
        top_examples, claude_api_key, identification_prompt, output_dir, combination_name, num_examples=args.num_examples)
    
    # Score pattern matches for all examples
    pattern_scores = await score_pattern_match(
        pattern, top_examples, claude_api_key, scoring_prompt, output_dir, combination_name, num_examples=args.num_examples)
    
    # Add scores to examples
    for i, score in enumerate(pattern_scores):
        if i < len(top_examples):
            top_examples[i]['pattern_match_score'] = score
    
    # Run weighted clamping interventions
    clamp_results = run_weighted_clamping(
        activations_1, activations_2, texts_1, texts_2, valid_indices_1, valid_indices_2,
        selected_indices, classifier_results, model, sae_model, tokenizer, use_sae,
        args.target_layer, args.token_distance, output_dir, combination_name)
    
    # Save final results
    results = {
        'combination_name': combination_name,
        'use_rsa': use_rsa,
        'rsa_type': rsa_type if use_rsa else None,
        'use_sae': use_sae,
        'selected_indices': selected_indices.tolist(),
        'selection_method': selection_method,
        'pattern': pattern,
        'top_examples': top_examples,  # Save all examples, not just first 10
        'pattern_match_scores': pattern_scores,
        'classifier_performance': {
            'train_accuracy': classifier_results['train_accuracy'],
            'test_accuracy': classifier_results['test_accuracy'],
            'weights': classifier_results['weights'].tolist()
        },
        'clamping_results': clamp_results
    }
    
    with open(combo_dir / "combination_results.json", "w") as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    logging.info(f"Completed combination: {combination_name} - Pattern: {pattern}")
    
    return results

async def main():
    parser = argparse.ArgumentParser(description="Combinatorial RSA feature selection analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to trained SAE model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--input_json", type=str, required=True, help="JSON file with statements")
    parser.add_argument("--token_distance", type=int, default=50, help="Number of tokens from start")
    parser.add_argument("--target_layer", type=int, default=16, help="Model layer to extract from")
    parser.add_argument("--config_dir", type=str, default="../config", help="Directory with API key config")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for Wikipedia processing")
    parser.add_argument("--num_examples", type=int, default=25, help="Number of top examples to analyze")
    parser.add_argument("--num_features", type=int, default=5, help="Number of top features to select")
    parser.add_argument("--start_from", type=str, default=None, help="Start from specific combination (skip completed ones)")
    parser.add_argument("--only_combination", type=str, default=None, help="Run only this specific combination")

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"combinatorial_analysis_{timestamp}"
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

    logging.info("Starting combinatorial RSA feature selection analysis...")

    # Load Claude API key
    claude_api_key = None
    if args.config_dir:
        config_path = Path(args.config_dir) / "api_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    claude_api_key = config.get("claude_api_key")
            except Exception as e:
                logging.warning(f"Failed to load Claude API key: {e}")

    # Load models
    logging.info("Loading models...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")

    # Load SAE
    state_dict = torch.load(args.sae_path, map_location="cuda")
    if 'decoder.weight' in state_dict:
        input_dim, hidden_dim = state_dict['decoder.weight'].shape
    else:
        encoder_weight = state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape

    sae_model = SparseAutoencoder(input_dim, hidden_dim)
    sae_model.load_state_dict(state_dict)
    sae_model.to(model.device)
    sae_model.eval()

    # Load statements
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    category_keys = list(data.keys())
    statements_dict, category_names = load_statements_from_json(args.input_json, category_keys)

    texts_1 = statements_dict[category_names[0]]
    texts_2 = statements_dict[category_names[1]]

    # Get valid indices (same for all combinations)
    _, valid_indices_1 = extract_activations_from_start(
        model, sae_model, tokenizer, texts_1, args.token_distance, args.target_layer, True)
    _, valid_indices_2 = extract_activations_from_start(
        model, sae_model, tokenizer, texts_2, args.token_distance, args.target_layer, True)

    # Load prompts
    identification_prompt = load_identification_prompt()
    scoring_prompt = load_scoring_prompt()

    # Define all combinations
    combinations = [
        ("RSA_SAE_discriminative", True, "discriminative", True),
        ("RSA_SAE_categorical", True, "categorical", True),
        ("RSA_raw_discriminative", True, "discriminative", False),
        ("RSA_raw_categorical", True, "categorical", False),
        ("raw_SAE", False, None, True),
        ("raw_raw", False, None, False)
    ]

    # Filter combinations based on start_from or only_combination
    if args.only_combination:
        combinations = [c for c in combinations if c[0] == args.only_combination]
        if not combinations:
            logging.error(f"Combination '{args.only_combination}' not found")
            return
        logging.info(f"Running only combination: {args.only_combination}")
    elif args.start_from:
        start_idx = None
        for i, (name, _, _, _) in enumerate(combinations):
            if name == args.start_from:
                start_idx = i
                break
        if start_idx is None:
            logging.error(f"Start combination '{args.start_from}' not found")
            return
        combinations = combinations[start_idx:]
        logging.info(f"Starting from combination: {args.start_from} (skipping {start_idx} combinations)")

    # Run all combinations
    all_results = []
    for combination_name, use_rsa, rsa_type, use_sae in combinations:
        try:
            # Check if combination already exists (for resume functionality)
            combo_dir = output_dir / combination_name
            if combo_dir.exists() and (combo_dir / "combination_results.json").exists() and not args.only_combination:
                logging.info(f"Combination {combination_name} already exists, skipping...")
                # Load existing results
                with open(combo_dir / "combination_results.json", "r") as f:
                    existing_result = json.load(f)
                all_results.append(existing_result)
                continue
                
            result = await run_single_combination(
                combination_name, use_rsa, rsa_type, use_sae,
                model, sae_model, tokenizer, texts_1, texts_2,
                valid_indices_1, valid_indices_2, args, output_dir,
                claude_api_key, identification_prompt, scoring_prompt)
            
            if result:
                all_results.append(result)
        except Exception as e:
            logging.error(f"Error in combination {combination_name}: {e}")

    # Save overall summary
    with open(output_dir / "overall_summary.json", "w") as f:
        json.dump(convert_numpy_types(all_results), f, indent=2)

    # Create summary report
    with open(output_dir / "analysis_summary.md", "w", encoding="utf-8") as f:
        f.write("# Combinatorial RSA Feature Selection Analysis Summary\n\n")
        f.write(f"**Categories**: {category_names[0]} vs {category_names[1]}\n")
        f.write(f"**Token Distance**: {args.token_distance}\n")
        f.write(f"**Samples**: {len(valid_indices_1)} + {len(valid_indices_2)}\n\n")
        
        for result in all_results:
            name = result['combination_name']
            pattern = result['pattern']
            train_acc = result['classifier_performance']['train_accuracy']
            test_acc = result['classifier_performance']['test_accuracy']
            selection_method = result['selection_method']
            selected_indices = result['selected_indices']
            
            f.write(f"## {name}\n")
            f.write(f"**Pattern**: {pattern}\n")
            f.write(f"**Feature Selection**: {selection_method}\n")
            f.write(f"**Selected Features**: {selected_indices}\n")
            f.write(f"**Classifier Performance**: Train {train_acc:.3f}, Test {test_acc:.3f}\n")
            
            # Show pattern match scores if available
            if 'pattern_match_scores' in result and result['pattern_match_scores']:
                scores = [s for s in result['pattern_match_scores'] if s is not None]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    f.write(f"**Pattern Match Score**: {avg_score:.2f} (avg of {len(scores)} examples)\n")
            
            # Show sample clamping results
            clamp_results = result.get('clamping_results', {})
            if 'base' in clamp_results:
                f.write(f"**Clamping Sample**:\n")
                f.write(f"- Base: {clamp_results['base'][:100]}...\n")
                if 'clamp_0.0x_weights' in clamp_results:
                    f.write(f"- Zero weights: {clamp_results['clamp_0.0x_weights'][:100]}...\n")
            
            f.write("\n")

    logging.info(f"Combinatorial analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())