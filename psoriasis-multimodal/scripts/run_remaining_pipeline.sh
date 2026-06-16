#!/bin/bash
# Exit on error
set -e

echo "=== Pipeline Runner: Monitoring VGG-16 and Running Remaining Models ==="

# Wait for VGG-16 metrics JSON to be written
echo "Waiting for VGG-16 training to finish (waiting for results/vgg16_metrics.json)..."
while [ ! -f results/vgg16_metrics.json ]; do
    sleep 30
done
echo "VGG-16 training completed successfully!"

# 1. Train EfficientNet-B3
echo "------------------------------------------------------------"
echo "Training EfficientNet-B3..."
echo "------------------------------------------------------------"
./venv/bin/python scripts/train_primary.py --config configs/config.yaml --model efficientnet_b3 --mode train

# 2. Train ViT-B/16
echo "------------------------------------------------------------"
echo "Training ViT-B/16 with LoRA..."
echo "------------------------------------------------------------"
./venv/bin/python scripts/train_primary.py --config configs/config.yaml --model vit_b16 --mode train

# 3. Run XAI for EfficientNet-B3
echo "------------------------------------------------------------"
echo "Generating XAI for EfficientNet-B3..."
echo "------------------------------------------------------------"
./venv/bin/python scripts/run_xai.py --config configs/config.yaml --model efficientnet_b3 --method all

# 4. Run XAI for ViT-B/16
echo "------------------------------------------------------------"
echo "Generating XAI for ViT-B/16..."
echo "------------------------------------------------------------"
./venv/bin/python scripts/run_xai.py --config configs/config.yaml --model vit_b16 --method attention_rollout

# 5. Run Full Evaluation and generate figures
echo "------------------------------------------------------------"
echo "Running full evaluation and generating final comparison..."
echo "------------------------------------------------------------"
./venv/bin/python scripts/evaluate_all.py --config configs/config.yaml

echo "------------------------------------------------------------"
echo "Generating publication-quality figures..."
echo "------------------------------------------------------------"
./venv/bin/python scripts/generate_figures.py --config configs/config.yaml

echo "============================================================"
echo "Pipeline Runner: All tasks successfully finished!"
echo "============================================================"
