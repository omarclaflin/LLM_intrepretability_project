# LLM Interpretability Project

This project focuses on intrepretability of Large Language Models (LLMs) exploring some cognitive neuroscience approaches.

## Overview

Essentially, I copied the Anthropic Monosemanticity blog/paper, but tried targetting semantic concepts and hunting for features (instead of semantic labelling on discovered features), along with applied cog neuro technqiues. Ran into a lot of challenges using a personal computer and small model.

## tl;dr

I looked at different feature selection approaches that could improve LLM Intrepretability, and I found (caveat: with a very small dataset):
1. RSA (representational similarity approaches) > maximum activation approaches, 
2. SAE > raw layer activations, 
3. and, if using RSA, discriminative > categorical

These results were run on a small dataset, LLM, SAE, etc so very preliminary. However, if true, RSA > max makes sense as in cog neuro, we see this being more sensitive to capturing the geometry of activity over thresholding. SAE > raw layer, this makes sense as we basically leverage the Sparse Autoencoder as a decomposition/mapping method prior to analysis. Discrimniative > categorical, this also makes sense give the pattern seen in cog neuro (usually discriminaitive is used).

Final project summary: ![ProjectSummary.md](ProjectSummary.md)

## Motivation

There are a lot of advantages if we could get some of these cog neuro approaches to work. I understand the scalability of 'top-down' concept-targetting approach to feature, makes less sense than the 'bottom-up' feature-concept discovery approach the field seems to use. However, I think by opposing techniques could be complementary if developed.

I was hoping to steamroll through the basic concept and move on to more advanced (and interesting) applications, but given my small model/single GPU, I ended up having a difficult enough time getting a small signal for the basic approach.

In the future, I think we could quickly design stim-responses (multidimensional ones: factual vs emotion, cross-domain analogies, gender/etc vs description/assumptions, and other potentially interactive semantic axises) and quantitatively visualize the LLM processing on it. 

## Research Journey

Set up notes:
- I loaded llama so I could train an SAE. Tokenizer is specific for llama. Realized batch sizes have to be small given my GPU size.
- For some reason, (I spent a lot of time), I couldn't access The Pile from my domain, so I used wiki-small. Later, realized how small that was, and went up in size.
- Attemped Anthropics workflow (but in a top-down approach, starting with a dataset purportedly representing a feature to 'pick a feature') 
      -Roughly, their workflow --> pick feature, scan large dataset for prompts that maximally activate that feature, token activation visualization, Claude API to 'feature ID' by finding a common pattern, text relevance scoring, clamping, some basic stats on token activations
      -The demonstrative, rather than comprehensive statistical approach initially surprised me, but its very cool since artifical brain science is such a new field

- Training the SAE: 
    - Copied the latest SAE blog post (regularization error terms, adam epsilon, etc) as closely as possible: ![https://transformer-circuits.pub/2024/april-update/index.html#training-saes](https://transformer-circuits.pub/2024/april-update/index.html#training-saes)
    - Ran into a variety of issues eventually solved with clipping, ambitiously large hidden layer (100k), learning parameter tweaking, quantile/winsorization, clamping, regularization parameter reduction, etc; Some of these were successfully removed but I didn't exhuastively test all of the changes after it worked.
    - My loss ~0.24 was comparable or higher to their worst reported loss 
    - Ended up having to stick with 30-50k, compared to their millions
    - Learned about shrinkage which I guessed would be an issue given the L1 regularizattion (I literally argued with Claude during its explanation to me), but the justification makes sense
    - The thought was to leverage this SAE (even if it appears to only capture 2/3 of variance) as a powerful decomposition tool that can enhance downstream analysis

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
  - kind of 'hacking' my data, but I realized, when I actually searched, there was only *one* mention of "golden gate" in the entire first dataset, and only ~100 in the second
  - I attempted to chunk up the bigger dataset into 100k tokens that included the 100 mentions to give my SAE a chance (~0.1%)
- didn't work, eventually dropping the concept for a more 'common' concept: success/achievement (vs failure)
- eventually adding in a scoring mechanisms for the target (regardless of the pattern identified, which was also used to score as a confirmatory test)
  - e.g. we target 'success', ID feature, scan for that, ID the pattern and get "sequels", score and get some >2s back; but, now, we still score for our target (and may see a couple vaguely-related to 'success' scores)
- retried with a 'left-first' windowing for RSA

Finally, stepped back, and tried a *new* pipeline with variations:
- Still: 1) Uses target concept, 2) ID features, 3) analyzes those features, 4) Scores
- However, for this pipeline, I used an ensemble of features (only 8), assuming concepts are multidimensionally represented
- Variations in pipeline: ['raw' middle layer activations/SAE] - [maximum activations/maximum RSA] - (if RSA: [discriminative RSA/categorical RSA])
  - all combinations were run
- Also, used a simple linear model to ensemble features (only 8, because our sample set was only 50)
- Note: Used RSA simply as feature identification, not for the classifier, although I was thinking about mapping that but the sparsity/smallness of data was too extreme compared to the feature size
- **Result: Essentially, RSA > raw, SAE > layer activations, and, if using RSA, discriminative ~> categorical**
  - Caveat: On a small dataset, one feature, small SAE, small model, small feature ensemble, etc.
![Method Comparison Analysis](results/SR_TOPIC_1_SUCCESS_CONTINUATION/combinatorial_analysis_20250525_012051/method_comparison_analysis.png)


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