# LLM Interpretability Project

This project focuses on understanding and analyzing the internal workings of Large Language Models (LLMs) through various interpretability techniques and tools.

## Overview

The project aims to provide insights into how LLMs process and generate text by examining their internal representations, attention patterns, and neuron activations. It includes tools for analyzing model behavior, visualizing attention patterns, and studying the role of individual neurons in the model's decision-making process.

Final project: ![ProjectSummary.md](ProjectSummary.md)

## Research Journey

Set up notes:
- I first attempted to load llama so I could train an SAE. Tokenizer is specific for llama
- For some reason, (I spent a lot of time), I couldn't access The Pile from my domain, so I used wiki-small
- Later, realized how small that was, and went up in size
- Attemped Anthropics workflow (but in a top-down approach, starting with a dataset purportedly representing a feature to 'pick a feature') --> scan large dataset for ones that feature activate, token activation visualization, Claude API to 'feature ID' by finding a common pattern, text relevance scoring, clamping, some basic stats on token activations

Perhaps not suprisingly, my overly ambitious semantic concept ("Golden Gate Bridge") couldn't be found on the small model and 50k SAE. I continued to stubbornly persist with neuroscience techniques:
- stimulus-response mapping (used above as well)
- opted for a 'centered' token window on the prompt (probably a mistake, at least without a 'targetting token')
  - could work for 'concrete' semantic concepts (e.g. nouns), but limited when looking at patterns across a sentence
  - bad when your concept is at the beginning and because RSA windowing (in this case) requires a consistent window
  - in the future, could use non-restrictive-windowed distance metrics
- looked at univariate contrast maps, RSA using the univariate contrast as an "ROI" window/feature selection, and RSA *within* SAE features by grabbing the top middle layer activations (this one can be run per token)
  - there are other permutations, as well as optimizations, I was just exploring

This didn't work well, so I tried:
- retraining the SAE on a large dataset
- chunking the wiki dataset for 'Golden Gate' samples
  - kind of 'hacking' my data, but I realized there was only *one* mention of "golden gate" in the entire first dataset, and only ~100 in the second
- eventually dropping the concept for a more 'common' concept: success/achievement (vs failure)
- eventually adding in a scoring mechanisms for the target (regardless of the pattern identified, which is also used to score)
  - e.g. we target 'success', ID feature, scan for that, ID the pattern and get "sequels", score and get some >2s back; but, now, we still score for our target (and may see a couple vaguely-related to 'success' scores)
- retried with a 'left-first' windowing for RSA

Finally, stepped back, and tried a pipeline with variations:
- Still: 1) Uses target concept, 2) ID features, 3) analyzes those features, 4) Scores
- However, for this pipeline, I used an ensemble of features (only 8), assuming concepts are multidimensionally represented
- Variations in pipeline: ['raw' middle layer activations/SAE] - [maximum activations/maximum RSA] - (if RSA: [discriminative RSA/categorical RSA])
  - all combinations were run
- Also, used a simple linear model to ensemble features (only 8, because our sample set was only 50)
- Note: Used RSA simply as feature identification, not for the classifier, although I was thinking about mapping that but the sparsity/smallness of data was too extreme compared to the feature size
- **Result: Essentially, RSA > raw, SAE > layer activations, and, if using RSA, discriminative ~ categorical**
  - Caveat: On a small dataset, one feature, small SAE, small model, small feature ensemble, etc.
![Method Comparison Analysis](results/SR_TOPIC_1_SUCCESS_CONTINUATION/combinatorial_analysis_20250525_012051/method_comparison_analysis.png)

## Features

- Sparse Autoencoder (SAE) model analysis
- Attention pattern visualization
- Neuron activation analysis
- Model behavior analysis tools
- Integration with TransformerLens for detailed model inspection

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd llm_intrepretability_project
```

2. Create and activate a virtual environment:
```bash
python -m venv llm-venv
source llm-venv/bin/activate  # On Windows: llm-venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For Windows users, run the setup script:
```powershell
.\setup-windows\setup-llm-intrepretability.ps1
```

5. To run batch analysis scripts:
```powershell
.\batch_part4_analysis_script.ps1
```

## Project Structure

- `config/` - Configuration files for different experiments
- `models/` - Pre-trained model checkpoints and saved models
- `checkpoints/` - Training checkpoints
- `open_llama_tests/` - Tests and experiments with Open LLaMA models
- `sae_model.pt` - Sparse Autoencoder model weights

## Usage

[Add specific usage instructions based on your project's main functionality]

## Dependencies

Key dependencies include:
- PyTorch
- TransformerLens
- Transformers
- Matplotlib
- Seaborn
- Wandb (for experiment tracking)

For a complete list of dependencies, see `requirements.txt`.

## Contributing

[Add contribution guidelines if applicable]

## License

This project is licensed under the terms included in the LICENSE file.

## Acknowledgments

- TransformerLens library for model interpretability tools
- Open LLaMA project for model weights
- [Add other acknowledgments as needed]