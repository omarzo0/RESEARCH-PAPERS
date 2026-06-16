# Deep Learning for Predicting Biologic Response in Psoriasis Patients from Skin Images

This repository contains the complete implementation of the image-only deep learning study to predict **biologic treatment response** in psoriasis patients using clinical skin photographs.

By mapping severity categories (Mild, Moderate, Severe) to a biologic response proxy grounded in clinical **PASI (Psoriasis Area and Severity Index)** thresholds, this framework evaluates multiple convolutional neural networks (CNNs) and Vision Transformers (ViTs) on their capacity to identify candidates for biologic therapies.

---

## 🌟 Features

- **End-to-End Pipeline**: Includes automated dataset downloading, advanced image quality filtering (blur detection and perceptual hash near-duplicate checking), and stratified splitting.
- **State-of-the-Art Architectures**:
  - **EfficientNet-B3** (Primary CNN model, partially unfrozen)
  - **ViT-B/16** (Primary ViT model, using **LoRA (Low-Rank Adaptation)** parameter-efficient fine-tuning)
  - **Baselines**: ResNet-50, DenseNet-121, and VGG-16.
- **Explainable AI (XAI)**:
  - **Grad-CAM++** for CNN spatial attention mapping.
  - **Attention Rollout** for ViT token-to-token attention visualization.
  - **LIME** for superpixel-based classification explanations.
- **Robust Evaluation**:
  - 5-fold stratified cross-validation.
  - Hyperparameter optimization using **Optuna** with median pruning.
  - Pairwise statistical significance testing via **McNemar's test**.
  - Automatic publication-quality figure generation (300 DPI, PDF + PNG formats).

---

## 📂 Project Structure

```
psoriasis-multimodal/
├── configs/
│   └── config.yaml               # Central hyperparameter and path config
├── data/
│   ├── raw/                      # Downloaded raw images
│   ├── processed/                # Normalized and resized images
│   └── splits/                   # Train/Val/Test CSV splits
├── scripts/
│   ├── download_data.py          # Automatic Kaggle dataset downloader
│   ├── run_preprocessing.py      # End-to-end preprocessing runner
│   ├── train_baselines.py        # CLI for baseline CNN training
│   ├── train_primary.py          # CLI for EfficientNet-B3 and ViT-B/16 training
│   ├── run_xai.py                # Visualizations (Grad-CAM++, LIME, Rollout)
│   ├── evaluate_all.py           # Overall comparison and significance testing
│   └── generate_figures.py       # 300 DPI publication figure exporter
├── src/
│   ├── models/
│   │   ├── baselines.py          # ResNet-50, DenseNet-121, VGG-16 definitions
│   │   ├── efficientnet.py       # Custom EfficientNet-B3 Classifier
│   │   ├── lora.py               # Custom PyTorch LoRA wrapper
│   │   └── vit.py                # ViT-B/16 with LoRA integration
│   ├── config.py                 # Reproducibility and config parser
│   ├── dataset.py                # PyTorch Dataset and DataLoader builder
│   ├── preprocessing.py          # Blur, duplicates, and split logic
│   ├── augmentation.py           # Albumentations + CutMix/MixUp collations
│   ├── train.py                  # Core PyTorch training engine
│   ├── evaluate.py               # ML metrics and plots
│   ├── explain.py                # XAI implementation (LIME, Grad-CAM++, Rollout)
│   └── utils.py                  # Logger and early stopping utils
└── requirements.txt              # Standardized dependencies list
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/psoriasis-multimodal.git
   cd psoriasis-multimodal
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Kaggle API Key Configuration**:
   To download the dataset automatically, make sure your Kaggle credentials are set up at `~/.kaggle/kaggle.json`. You can download the key from your Kaggle Account Settings.

---

## 🚀 Step-by-Step Execution Guide

### 1. Data Acquisition
Download the raw Psoriasis Skin Dataset from Kaggle:
```bash
python scripts/download_data.py
```
*If your Kaggle API key is not configured, follow the printed manual download instructions and place the images in `data/raw/`.*

### 2. Preprocessing & Quality Control
Filter low-quality blurry images, detect and remove near-duplicates, map labels to biologic response proxies (`Mild` $\to 0$, `Moderate`/`Severe` $\to 1$), and create stratified splits:
```bash
python scripts/run_preprocessing.py --config configs/config.yaml
```

### 3. Train Baseline CNNs
Train the ResNet-50, DenseNet-121, and VGG-16 baseline classifiers:
```bash
python scripts/train_baselines.py --config configs/config.yaml --model resnet50
python scripts/train_baselines.py --config configs/config.yaml --model densenet121
python scripts/train_baselines.py --config configs/config.yaml --model vgg16
```

### 4. Primary Model Optimization & Training
For **EfficientNet-B3** and **ViT-B/16**:

- **Hyperparameter Optimization (Optuna)**:
  Runs 50 trials (with median pruning) to search for the best learning rate, dropout, weight decay, and batch size:
  ```bash
  python scripts/train_primary.py --config configs/config.yaml --model efficientnet_b3 --mode hyperopt
  ```

- **5-Fold Stratified Cross-Validation**:
  Assess robustness and prevent overfitting:
  ```bash
  python scripts/train_primary.py --config configs/config.yaml --model efficientnet_b3 --mode cv
  ```

- **Full Dataset Training**:
  Train on the combined training/validation split and save the final best model checkpoint:
  ```bash
  python scripts/train_primary.py --config configs/config.yaml --model efficientnet_b3 --mode train
  python scripts/train_primary.py --config configs/config.yaml --model vit_b16 --mode train
  ```

### 5. Explainability (XAI) Generation
Generate Grad-CAM++, LIME, and Attention Rollout visualizations:
```bash
python scripts/run_xai.py --config configs/config.yaml --model efficientnet_b3 --method all
python scripts/run_xai.py --config configs/config.yaml --model vit_b16 --method attention_rollout
```
*Visualizations are exported directly to `results/figures/`.*

### 6. System-Wide Evaluation & Figure Export
Evaluate all models on the held-out test set, execute McNemar's test for significance, export a LaTeX-ready table, and output high-DPI publication figures:
```bash
python scripts/evaluate_all.py --config configs/config.yaml
python scripts/generate_figures.py --config configs/config.yaml
```

---

## 📊 Expected Performance Target

| Metric | Target |
| :--- | :--- |
| **AUC-ROC** | $> 0.87$ |
| **F1 Score** | $> 0.82$ |
| **Sensitivity** | $> 0.80$ |
| **Specificity** | $> 0.84$ |
| **Accuracy** | $> 0.83$ |

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
