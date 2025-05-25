"""
Compare Feature Selection Methods Results

Analyzes target pattern scoring results across all 6 combinations,
creates overlapping histograms, and ranks methods by performance.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict

def load_combination_results(analysis_dir, combination_name):
    """Load target pattern scoring results for a combination."""
    combo_dir = Path(analysis_dir) / combination_name
    
    # Try to load target pattern scoring results
    target_file = combo_dir / f"target_pattern_scoring_{combination_name}.json"
    if target_file.exists():
        with open(target_file, "r") as f:
            return json.load(f)
    else:
        print(f"Warning: No target pattern scoring results found for {combination_name}")
        return None

def extract_scores(results):
    """Extract valid scores from results."""
    if not results:
        return []
    
    individual_scores = results.get('individual_scores', [])
    scores = []
    for item in individual_scores:
        score = item.get('target_pattern_score')
        if score is not None and isinstance(score, (int, float)):
            scores.append(score)
    
    return scores

def create_comparison_plots(all_results, output_dir):
    """Create comparison plots between methods."""
    
    # Prepare data for plotting
    method_scores = {}
    method_stats = {}
    
    for combo_name, results in all_results.items():
        scores = extract_scores(results)
        if scores:
            method_scores[combo_name] = scores
            method_stats[combo_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores),
                'high_quality': len([s for s in scores if s >= 2]),
                'perfect': len([s for s in scores if s == 3])
            }
    
    if not method_scores:
        print("No valid scores found for any method")
        return
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Overlapping histograms (percentage-based)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    alphas = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    
    for i, (method, scores) in enumerate(method_scores.items()):
        # Calculate percentage distribution
        score_counts = {j: scores.count(j) for j in range(4)}
        total = len(scores)
        percentages = [score_counts[j] / total * 100 for j in range(4)]
        
        # Create x positions with slight offset for each method
        x_positions = [j + (i - 2.5) * 0.12 for j in range(4)]  # Center around each score
        
        axes[0, 0].bar(x_positions, percentages, width=0.1, alpha=alphas[i], 
                      color=colors[i], label=f'{method} (n={total})')
    
    axes[0, 0].set_xlabel('Achievement Pattern Score')
    axes[0, 0].set_ylabel('Percentage of Examples')
    axes[0, 0].set_title('Score Distribution by Method (% of Examples)')
    axes[0, 0].set_xticks([0, 1, 2, 3])
    axes[0, 0].set_xticklabels(['0 (None)', '1 (Vague)', '2 (Loose)', '3 (Clear)'])
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Mean scores with error bars
    methods = list(method_stats.keys())
    means = [method_stats[m]['mean'] for m in methods]
    stds = [method_stats[m]['std'] for m in methods]
    
    bars = axes[0, 1].bar(range(len(methods)), means, yerr=stds, capsize=5, 
                         color=colors[:len(methods)], alpha=0.7)
    axes[0, 1].set_xlabel('Feature Selection Method')
    axes[0, 1].set_ylabel('Mean Achievement Score')
    axes[0, 1].set_title('Average Performance by Method')
    axes[0, 1].set_xticks(range(len(methods)))
    axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{mean:.2f}', ha='center', va='bottom')
    
    # Plot 3: High-quality matches (score >= 2)
    high_quality_pct = [(method_stats[m]['high_quality'] / method_stats[m]['count']) * 100 
                       for m in methods]
    
    bars = axes[1, 0].bar(range(len(methods)), high_quality_pct, 
                         color=colors[:len(methods)], alpha=0.7)
    axes[1, 0].set_xlabel('Feature Selection Method')
    axes[1, 0].set_ylabel('High-Quality Matches (%)')
    axes[1, 0].set_title('Percentage of High-Quality Matches (Score ≥ 2)')
    axes[1, 0].set_xticks(range(len(methods)))
    axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, high_quality_pct):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{pct:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Summary statistics table
    axes[1, 1].axis('off')
    
    # Create ranking table
    ranking_data = []
    for method in methods:
        stats = method_stats[method]
        ranking_data.append({
            'Method': method,
            'Mean Score': f"{stats['mean']:.2f}",
            'Std Dev': f"{stats['std']:.2f}",
            'Sample Size': stats['count'],
            'High Quality (≥2)': f"{stats['high_quality']}/{stats['count']} ({stats['high_quality']/stats['count']*100:.1f}%)",
            'Perfect (=3)': f"{stats['perfect']}/{stats['count']} ({stats['perfect']/stats['count']*100:.1f}%)"
        })
    
    # Sort by mean score (descending)
    ranking_data.sort(key=lambda x: float(x['Mean Score']), reverse=True)
    
    # Create table text
    table_text = "RANKING BY AVERAGE ACHIEVEMENT SCORE\n\n"
    for i, row in enumerate(ranking_data):
        table_text += f"{i+1}. {row['Method']}\n"
        table_text += f"   Mean: {row['Mean Score']} (±{row['Std Dev']})\n"
        table_text += f"   High Quality: {row['High Quality (≥2)']}\n"
        table_text += f"   Perfect: {row['Perfect (=3)']}\n"
        table_text += f"   Sample Size: {row['Sample Size']}\n\n"
    
    axes[1, 1].text(0.05, 0.95, table_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return ranking_data

def create_detailed_comparison_table(all_results, output_dir):
    """Create detailed CSV comparison table."""
    
    detailed_data = []
    
    for combo_name, results in all_results.items():
        if not results:
            continue
            
        individual_scores = results.get('individual_scores', [])
        
        for item in individual_scores:
            detailed_data.append({
                'method': combo_name,
                'example_index': item.get('example_index', ''),
                'text_snippet': item.get('text', '')[:100] + "..." if item.get('text') else '',
                'prediction_prob': item.get('prediction_prob'),
                'discovered_pattern_score': item.get('discovered_pattern_score'),
                'target_pattern_score': item.get('target_pattern_score'),
                'target_pattern': item.get('target_pattern', 'achievements')
            })
    
    # Save to CSV
    if detailed_data:
        df = pd.DataFrame(detailed_data)
        df.to_csv(output_dir / "detailed_method_comparison.csv", index=False)
        print(f"Detailed comparison saved to: {output_dir / 'detailed_method_comparison.csv'}")

def main():
    parser = argparse.ArgumentParser(description="Compare feature selection method results")
    parser.add_argument("--analysis_dir", type=str, required=True, 
                       help="Path to combinatorial analysis directory")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Output directory (defaults to analysis_dir)")
    
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.exists():
        print(f"Error: Analysis directory {analysis_dir} does not exist")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir
    output_dir.mkdir(exist_ok=True)
    
    # Define all combinations
    combinations = [
        "RSA_SAE_discriminative",
        "RSA_SAE_categorical", 
        "RSA_raw_discriminative",
        "RSA_raw_categorical",
        "raw_SAE",
        "raw_raw"
    ]
    
    print("Loading results from all combinations...")
    
    # Load results from all combinations
    all_results = {}
    for combo in combinations:
        print(f"Loading {combo}...")
        results = load_combination_results(analysis_dir, combo)
        if results:
            all_results[combo] = results
            scores = extract_scores(results)
            print(f"  Found {len(scores)} valid scores")
        else:
            print(f"  No results found")
    
    if not all_results:
        print("No results found for any combination")
        return
    
    print(f"\nSuccessfully loaded {len(all_results)} combinations")
    
    # Create comparison plots
    print("Creating comparison plots...")
    ranking_data = create_comparison_plots(all_results, output_dir)
    
    # Create detailed comparison table
    print("Creating detailed comparison table...")
    create_detailed_comparison_table(all_results, output_dir)
    
    # Save ranking summary
    ranking_summary = {
        'analysis_timestamp': analysis_dir.name,
        'target_pattern': 'achievements',
        'total_methods_compared': len(all_results),
        'ranking': ranking_data
    }
    
    with open(output_dir / "method_ranking_summary.json", "w") as f:
        json.dump(ranking_summary, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Main plot: method_comparison_analysis.png")
    print(f"Detailed data: detailed_method_comparison.csv")
    print(f"Ranking: method_ranking_summary.json")
    
    # Print quick summary
    print(f"\nQUICK RANKING SUMMARY:")
    for i, method_data in enumerate(ranking_data):
        print(f"{i+1}. {method_data['Method']}: {method_data['Mean Score']} avg score")

if __name__ == "__main__":
    main()