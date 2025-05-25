# Batch script to run part4 feature analysis on multiple result sets
# Save this as "run_all_part4.ps1" and run with: .\run_all_part4.ps1

Write-Host "Starting batch Part4 analysis..." -ForegroundColor Green
Write-Host "This will run 4 analyses sequentially" -ForegroundColor Green

# Common parameters
$MODEL_PATH = "../models/open_llama_3b"
$SAE_PATH = "../sae_model.pt"
$INPUT_JSON = "data/SR_TOPIC_0_GOLDEN_GATE_BRIDGE/unifactorial-bridge-category.json"
$CONFIG_DIR = "../config"
$NUM_EXAMPLES = 10
$BATCH_SIZE = 4
$NUM_SAMPLES = 75000

# Analysis 1: NeuronInputRSA
Write-Host "`n=== ANALYSIS 1: NeuronInputRSA ===" -ForegroundColor Yellow
$FEATURES_1 = "results/SR_TOPIC_0_GOLDEN_GATE_BRIDGE/NeuronInputRSA_neuroscience_analysis_20250523_184456/rsa_results.json"
$OUTPUT_1 = "results/SR_TOPIC_0_GOLDEN_GATE_BRIDGE/part4_NeuronInputRSA"

Write-Host "Features file: $FEATURES_1" -ForegroundColor Cyan
Write-Host "Output dir: $OUTPUT_1" -ForegroundColor Cyan

python part4b_find_feature_meaning_large_wiki.py `
  --model_path $MODEL_PATH `
  --sae_path $SAE_PATH `
  --output_dir $OUTPUT_1 `
  --input_json $INPUT_JSON `
  --features $FEATURES_1 `
  --num_examples $NUM_EXAMPLES `
  --batch_size $BATCH_SIZE `
  --config_dir $CONFIG_DIR `
  --num_samples $NUM_SAMPLES

if ($LASTEXITCODE -ne 0) {
    Write-Host "Analysis 1 failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Analysis 1 completed successfully!" -ForegroundColor Green

# Analysis 2: univariate_RSA
Write-Host "`n=== ANALYSIS 2: univariate_RSA ===" -ForegroundColor Yellow
$FEATURES_2 = "results/SR_TOPIC_0_GOLDEN_GATE_BRIDGE/univariate_RSA_neuroscience_analysis_20250523_175528/rsa_results.json"
$OUTPUT_2 = "results/SR_TOPIC_0_GOLDEN_GATE_BRIDGE/part4_univariate_RSA"

Write-Host "Features file: $FEATURES_2" -ForegroundColor Cyan
Write-Host "Output dir: $OUTPUT_2" -ForegroundColor Cyan

python part4b_find_feature_meaning_large_wiki.py `
  --model_path $MODEL_PATH `
  --sae_path $SAE_PATH `
  --output_dir $OUTPUT_2 `
  --input_json $INPUT_JSON `
  --features $FEATURES_2 `
  --num_examples $NUM_EXAMPLES `
  --batch_size $BATCH_SIZE `
  --config_dir $CONFIG_DIR `
  --num_samples $NUM_SAMPLES

if ($LASTEXITCODE -ne 0) {
    Write-Host "Analysis 2 failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Analysis 2 completed successfully!" -ForegroundColor Green

# Analysis 3: RSA only
Write-Host "`n=== ANALYSIS 3: RSA_only ===" -ForegroundColor Yellow
$FEATURES_3 = "results/SR_TOPIC_0_GOLDEN_GATE_BRIDGE/neuroscience_analysis_20250523_175422/rsa_results.json"
$OUTPUT_3 = "results/SR_TOPIC_0_GOLDEN_GATE_BRIDGE/part4_RSA_only"

Write-Host "Features file: $FEATURES_3" -ForegroundColor Cyan
Write-Host "Output dir: $OUTPUT_3" -ForegroundColor Cyan

python part4b_find_feature_meaning_large_wiki.py `
  --model_path $MODEL_PATH `
  --sae_path $SAE_PATH `
  --output_dir $OUTPUT_3 `
  --input_json $INPUT_JSON `
  --features $FEATURES_3 `
  --num_examples $NUM_EXAMPLES `
  --batch_size $BATCH_SIZE `
  --config_dir $CONFIG_DIR `
  --num_samples $NUM_SAMPLES

if ($LASTEXITCODE -ne 0) {
    Write-Host "Analysis 3 failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Analysis 3 completed successfully!" -ForegroundColor Green

# Analysis 4: T-test results
Write-Host "`n=== ANALYSIS 4: T-test results ===" -ForegroundColor Yellow
$FEATURES_4 = "results/SR_TOPIC_0_GOLDEN_GATE_BRIDGE/neuroscience_analysis_20250523_175422/top_features.json"
$OUTPUT_4 = "results/SR_TOPIC_0_GOLDEN_GATE_BRIDGE/part4_ttest"

Write-Host "Features file: $FEATURES_4" -ForegroundColor Cyan
Write-Host "Output dir: $OUTPUT_4" -ForegroundColor Cyan

python part4b_find_feature_meaning_large_wiki.py `
  --model_path $MODEL_PATH `
  --sae_path $SAE_PATH `
  --output_dir $OUTPUT_4 `
  --input_json $INPUT_JSON `
  --features $FEATURES_4 `
  --num_examples $NUM_EXAMPLES `
  --batch_size $BATCH_SIZE `
  --config_dir $CONFIG_DIR `
  --num_samples $NUM_SAMPLES

if ($LASTEXITCODE -ne 0) {
    Write-Host "Analysis 4 failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Analysis 4 completed successfully!" -ForegroundColor Green

Write-Host "`n=== ALL ANALYSES COMPLETED ===" -ForegroundColor Green
Write-Host "Results saved to:" -ForegroundColor Green
Write-Host "  1. $OUTPUT_1" -ForegroundColor Cyan
Write-Host "  2. $OUTPUT_2" -ForegroundColor Cyan  
Write-Host "  3. $OUTPUT_3" -ForegroundColor Cyan
Write-Host "  4. $OUTPUT_4" -ForegroundColor Cyan