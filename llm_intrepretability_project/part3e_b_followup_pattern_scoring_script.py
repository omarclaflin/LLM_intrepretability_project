"""
Target Pattern Scoring Script

Scores top Wikipedia examples against a predefined target pattern (achievements/records)
instead of the discovered pattern from the analysis.

 python part3e_b_followup_pattern_scoring_script.py `
>>   --results_dir "results/SR_TOPIC_1_SUCCESS_CONTINUATION/combinatorial_analysis_20250524_230725" `
>>   --combination "RSA_SAE_discriminative" `
>>   --target_pattern "achievement or failure" `
>>   --config_dir "../config" `
>>   --num_examples 15
"""

import argparse
import asyncio
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
import logging
from tqdm.auto import tqdm

def load_scoring_prompt():
    """Load the scoring prompt template."""
    prompt_path = Path("scoring_prompt.txt")
    if not prompt_path.exists():
        prompt_path = Path("prompts/scoring_prompt.txt")

    if not prompt_path.exists():
        logging.warning(f"Scoring prompt template not found. Using default.")
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

async def score_against_target_pattern(examples, target_pattern, claude_api_key, scoring_prompt, output_dir, num_examples=25):
    """Score examples against a target pattern instead of discovered pattern."""
    if not claude_api_key:
        return []
    
    scores = []
    
    # Score each example
    for i, example in enumerate(tqdm(examples[:num_examples], desc="Scoring against target pattern")):
        text_content = example['text'].strip()
        
        if not text_content:
            scores.append(None)
            continue
        
        try:
            scoring_prompt_formatted = scoring_prompt.format(
                pattern=target_pattern,
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
    
    return scores

def create_score_distribution_plot(scores, target_pattern, output_path):
    """Create histogram/distribution plot of target pattern scores."""
    valid_scores = [s for s in scores if s is not None]
    
    if not valid_scores:
        logging.error("No valid scores to plot")
        return
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Basic histogram
    axes[0, 0].hist(valid_scores, bins=4, range=(-0.5, 3.5), alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Target Pattern Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Score Distribution for "{target_pattern}"')
    axes[0, 0].set_xticks([0, 1, 2, 3])
    
    # Add count labels on bars
    counts, bins, patches = axes[0, 0].hist(valid_scores, bins=4, range=(-0.5, 3.5), alpha=0)
    for i, count in enumerate(counts):
        if count > 0:
            axes[0, 0].text(i, count + 0.1, str(int(count)), ha='center')
    
    # Plot 2: Percentage distribution
    score_counts = {i: valid_scores.count(i) for i in range(4)}
    total = len(valid_scores)
    percentages = [score_counts[i] / total * 100 for i in range(4)]
    
    bars = axes[0, 1].bar(range(4), percentages, alpha=0.7, color=['red', 'orange', 'yellow', 'green'])
    axes[0, 1].set_xlabel('Target Pattern Score')
    axes[0, 1].set_ylabel('Percentage')
    axes[0, 1].set_title(f'Score Distribution (%) for "{target_pattern}"')
    axes[0, 1].set_xticks([0, 1, 2, 3])
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        if pct > 0:
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{pct:.1f}%', ha='center')
    
    # Plot 3: Cumulative distribution
    axes[1, 0].hist(valid_scores, bins=4, range=(-0.5, 3.5), cumulative=True, 
                    alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].set_xlabel('Target Pattern Score')
    axes[1, 0].set_ylabel('Cumulative Count')
    axes[1, 0].set_title(f'Cumulative Distribution for "{target_pattern}"')
    axes[1, 0].set_xticks([0, 1, 2, 3])
    
    # Plot 4: Summary statistics as text
    axes[1, 1].axis('off')
    mean_score = np.mean(valid_scores)
    median_score = np.median(valid_scores)
    mode_score = max(set(valid_scores), key=valid_scores.count)
    
    stats_text = f"""Target Pattern: "{target_pattern}"
    
Total Examples Scored: {len(valid_scores)}
Mean Score: {mean_score:.2f}
Median Score: {median_score:.1f}
Mode Score: {mode_score}

Score Distribution:
0 (Irrelevant): {score_counts[0]} ({score_counts[0]/total*100:.1f}%)
1 (Vaguely related): {score_counts[1]} ({score_counts[1]/total*100:.1f}%)
2 (Loosely related): {score_counts[2]} ({score_counts[2]/total*100:.1f}%)
3 (Clearly identifies): {score_counts[3]} ({score_counts[3]/total*100:.1f}%)

High Quality Matches (Score ≥ 2): {score_counts[2] + score_counts[3]} ({(score_counts[2] + score_counts[3])/total*100:.1f}%)
Perfect Matches (Score = 3): {score_counts[3]} ({score_counts[3]/total*100:.1f}%)"""
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Score distribution plot saved to {output_path}")

async def main():
    parser = argparse.ArgumentParser(description="Score examples against target pattern")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing analysis results")
    parser.add_argument("--combination", type=str, required=True, help="Combination name (e.g., RSA_SAE_discriminative)")
    parser.add_argument("--target_pattern", type=str, default="achievements, records, and exceptional accomplishments", 
                       help="Target pattern to score against")
    parser.add_argument("--config_dir", type=str, default="../config", help="Directory with API key config")
    parser.add_argument("--num_examples", type=int, default=25, help="Number of examples to score")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
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
                logging.error(f"Failed to load Claude API key: {e}")
                return
    
    if not claude_api_key:
        logging.error("No Claude API key found")
        return
    
    # Find the analysis directory
    results_path = Path(args.results_dir)
    if not results_path.exists():
        logging.error(f"Results directory {results_path} does not exist")
        return
    
    # Check if the provided path is directly an analysis directory
    if results_path.name.startswith("combinatorial_analysis_"):
        # User provided direct path to analysis directory
        analysis_dir = results_path
        logging.info(f"Using provided analysis directory: {analysis_dir}")
    else:
        # Look for timestamped analysis directories
        analysis_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("combinatorial_analysis_")]
        if not analysis_dirs:
            logging.error(f"No analysis directories found in {results_path}")
            return
        
        # Use the most recent one
        analysis_dir = max(analysis_dirs, key=lambda x: x.stat().st_mtime)
        logging.info(f"Using most recent analysis directory: {analysis_dir}")
    
    # Load combination results
    combination_dir = analysis_dir / args.combination
    if not combination_dir.exists():
        logging.error(f"Combination directory {combination_dir} does not exist")
        return
    
    results_file = combination_dir / "combination_results.json"
    if not results_file.exists():
        logging.error(f"Results file {results_file} does not exist")
        return
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    top_examples = results.get("top_examples", [])
    if not top_examples:
        logging.error("No top examples found in results")
        return
    
    logging.info(f"Found {len(top_examples)} examples to score against target pattern: '{args.target_pattern}'")
    
    # Load scoring prompt
    scoring_prompt = load_scoring_prompt()
    
    # Score examples against target pattern
    target_scores = await score_against_target_pattern(
        top_examples, args.target_pattern, claude_api_key, scoring_prompt, 
        combination_dir, num_examples=args.num_examples
    )
    
    # Calculate score distribution
    valid_scores = [s for s in target_scores if s is not None]
    if valid_scores:
        score_distribution = {}
        for score_val in [0, 1, 2, 3]:
            count = valid_scores.count(score_val)
            score_distribution[score_val] = {
                'count': count,
                'percentage': (count / len(valid_scores)) * 100
            }
        score_distribution['mean'] = np.mean(valid_scores)
        score_distribution['total_scored'] = len(valid_scores)
        
        logging.info(f"Score distribution: {score_distribution}")
    
    # Create distribution plot
    plot_path = combination_dir / f"target_pattern_distribution_{args.combination}.png"
    create_score_distribution_plot(target_scores, args.target_pattern, plot_path)
    
    # Save detailed results
    target_scoring_results = []
    for i, (example, score) in enumerate(zip(top_examples[:args.num_examples], target_scores)):
        target_scoring_results.append({
            'example_index': i,
            'text': example['text'],
            'prediction_prob': example.get('prediction_prob', None),
            'discovered_pattern_score': example.get('pattern_match_score', None),
            'target_pattern_score': score,
            'target_pattern': args.target_pattern
        })
    
    full_target_results = {
        'target_pattern': args.target_pattern,
        'combination': args.combination,
        'individual_scores': target_scoring_results,
        'score_distribution': score_distribution if valid_scores else None,
        'summary': {
            'total_examples': args.num_examples,
            'successfully_scored': len(valid_scores),
            'mean_target_score': np.mean(valid_scores) if valid_scores else None,
            'high_quality_matches': len([s for s in valid_scores if s >= 2]) if valid_scores else 0,
            'perfect_matches': len([s for s in valid_scores if s == 3]) if valid_scores else 0
        }
    }
    
    # Save results
    output_file = combination_dir / f"target_pattern_scoring_{args.combination}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_target_results, f, indent=2)
    
    logging.info(f"Target pattern scoring complete!")
    logging.info(f"Results saved to: {output_file}")
    logging.info(f"Distribution plot saved to: {plot_path}")
    
    if valid_scores:
        logging.info(f"Summary: {len([s for s in valid_scores if s >= 2])}/{len(valid_scores)} high-quality matches (≥2)")
        logging.info(f"Perfect matches (score=3): {len([s for s in valid_scores if s == 3])}/{len(valid_scores)}")

if __name__ == "__main__":
    asyncio.run(main())