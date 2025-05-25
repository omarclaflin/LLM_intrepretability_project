import argparse
import json
import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import your visualization tools
from feature_visualization import FeatureVisualizer

# Parse arguments
parser = argparse.ArgumentParser(description="Visualize extracted features")
parser.add_argument("--results_dir", type=str, required=True, help="Path to extraction results")
parser.add_argument("--category", type=str, required=True, help="Category to visualize (category1, category2, combined)")
parser.add_argument("--feature_rank", type=int, default=1, help="Rank of feature to visualize (1 = top feature)")
parser.add_argument("--visualization_type", type=str, default="heatmap", 
                   choices=["heatmap", "distribution", "specificity"], 
                   help="Type of visualization to generate")
parser.add_argument("--example_set", type=str, default=None, 
                   help="Example set to visualize (e.g., category1_positive)")
args = parser.parse_args()

# Load analysis results
with open(os.path.join(args.results_dir, "feature_analysis.json"), "r") as f:
    analysis = json.load(f)

# Get feature info
if args.category == "category1":
    feature_info = analysis["features"]["category1"]
elif args.category == "category2":
    feature_info = analysis["features"]["category2"]
elif args.category == "combined":
    feature_info = analysis["features"]["combined_analyses"]["positive1_in_combined"]
else:
    raise ValueError(f"Unknown category: {args.category}")

# Get feature index (adjust for 0-indexing)
feature_idx = feature_info["feature_indices"][args.feature_rank - 1]
print(f"Visualizing feature {feature_idx} (rank {args.feature_rank}) for {args.category}")

# Load example activations
if args.example_set is None:
    # Default to the positive examples for the category
    if args.category == "category1":
        args.example_set = "category1_positive"
    elif args.category == "category2":
        args.example_set = "category2_positive"
    else:
        args.example_set = "combined_positive1_positive2"

activation_path = os.path.join(args.results_dir, "activations", f"{args.example_set}_activations.pt")
activation_data = torch.load(activation_path)

# Set up visualizer
visualizer = FeatureVisualizer(None, None)  # We're not using the model here, just the visualizations

# Create visualization output directory
vis_dir = os.path.join(args.results_dir, "visualizations")
os.makedirs(vis_dir, exist_ok=True)

# Generate visualization
output_path = os.path.join(vis_dir, f"{args.category}_feature{feature_idx}_{args.visualization_type}.png")

if args.visualization_type == "heatmap":
    # Create heatmap for the feature across examples
    plt.figure(figsize=(15, 10))
    feature_acts = activation_data["sae_activations"][:, feature_idx].cpu().numpy()
    plt.imshow(feature_acts.reshape(-1, 1), aspect='auto', cmap='YlOrRd')
    plt.colorbar(label="Feature Activation")
    plt.title(f"Feature {feature_idx} Activations Across {args.example_set} Examples")
    plt.xlabel("Feature")
    plt.ylabel("Example Index")
    plt.savefig(output_path)
    plt.close()
    
elif args.visualization_type == "distribution":
    # Plot activation distribution
    feature_acts = activation_data["sae_activations"][:, feature_idx].cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(feature_acts.flatten(), bins=50, alpha=0.7)
    plt.title(f"Distribution of Feature {feature_idx} Activations for {args.example_set}")
    plt.xlabel("Activation Value")
    plt.ylabel("Count")
    plt.savefig(output_path)
    plt.close()
    
elif args.visualization_type == "specificity":
    # For this, we'd need to implement the specificity analysis
    print("Specificity visualization not implemented in this simplified script")

print(f"Visualization saved to {output_path}")