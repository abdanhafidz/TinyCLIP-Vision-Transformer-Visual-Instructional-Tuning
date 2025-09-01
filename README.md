# Vision Instruction Tuning with CLIP-based Training
A comprehensive implementation of vision-language instruction tuning following CLIP training procedures with two-stage training methodology and extensive ablation studies.


You can read the full experimental paper of this project as a part of study ablation on : [📄 AUTHOR PAPER](https://drive.google.com/file/d/1jxY_rsAO2VfTHD-GnGLG5YKOZCGumRGN/view?usp=sharing)

## Research Paper References :
[Visual Instructional Tuning by LLava](https://arxiv.org/abs/2304.08485)
[TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance](https://arxiv.org/abs/2309.12314)
[Dense Connector for MLLMs](https://arxiv.org/abs/2405.13800)
## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset Preparation](#dataset-preparation)
- [Training Pipeline](#training-pipeline)
- [Ablation Studies](#ablation-studies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## 🔍 Overview

This project implements a vision instruction tuning system that follows CLIP training procedures with a two-stage approach:

1. **Stage 1: Vision Projection Training** - Training the vision projection layer to align visual features with language embeddings
2. **Stage 2: LLM Fine-tuning** - Fine-tuning the language model for instruction following with visual context

### Key Features
- **Base Model**: TinyCLIP-ViT-40M for efficient vision encoding
<img width="1058" height="457" alt="image" src="https://github.com/user-attachments/assets/52eb144f-4012-4237-b290-053c66d99697" />

- **Two-stage Training**: Separate vision projection and LLM fine-tuning stages
- **Comprehensive Ablation Studies**: Testing various configurations and hyperparameters
- **Flexible Architecture**: Support for dense connectors and different scheduling strategies

## 🏗️ Architecture

### Training Pipeline Flow

<img width="555" height="693" alt="image" src="https://github.com/user-attachments/assets/a906e2a7-4181-4f36-b845-38a1eeac9b5c" />


**Stage 1: Vision Projection Training**
- Vision Encoder (TinyCLIP-ViT-40M) ❄️ (frozen)
- Trainable Vision Projection Layer 🔥
- Llama 3.2 1B ❄️ (frozen)

**Stage 2: Visual Instruction Tuning**
- Vision Encoder (TinyCLIP-ViT-40M) ❄️ (frozen) 
- Trained Vision Projection Layer 🔥
- Llama 3.2 1B 🔥 (fine-tuned)
- Urna 1B ❄️ (frozen)

### Data Processing Pipeline

<img width="860" height="346" alt="image" src="https://github.com/user-attachments/assets/5ec08a8e-30c8-46ef-8bc8-71a9bb209e43" />


## 📊 Dataset Preparation

### Source Dataset
- **Primary Dataset**: ShareGPT4V Dataset (English)
- **Filtering**: COCO images only for consistency
- **Size**: 50,027 conversation pairs after filtering

### Preprocessing Steps
1. **Data Sampling**: Strategic sampling from the original dataset
2. **Translation**: Indonesian language support (ID)
3. **Text Embedding**: Human conversation text embedding (384 dimensions)
4. **Dimensionality Reduction**: UMAP reduction to 10 dimensions for clustering
5. **Cluster Sampling**: Random sampling per cluster for balanced dataset creation

### Dataset Variants
- **10kRandom**: 10k samples with random sampling strategy
- **20kUmap**: 20k samples with UMAP-based clustering and sampling

### Experimental Configurations
- **Learning Rate**: Fixed at 2e-5 across all experiments
- **Schedulers**: Linear with Warmup, Cosine with Warmup
- **Epochs**: 3, 6, 9 epochs tested
- **Batch Sizes**: 1, 8, 12 (varying based on GPU memory)
- **Dense Connector**: 4-layer dense projection when enabled
- **Chat Template**: Applied to alpindale/Llama-3.2-1B-Instruct
- **Hardware**: NVIDIA A100 with 10GB, 26.4GB, or 37GB configurations

## 🚀 Training Pipeline

### Stage 1: Vision Projection Training
```python
# Configuration
- Base Model: TinyCLIP-ViT-40M (frozen)
- Projection Layer: Trainable dense layer
- Target: Llama 3.2 1B embeddings (frozen)
- Objective: Align visual features with language space
```

### Stage 2: Visual Instruction Tuning
```python
# Configuration  
- Vision Encoder: TinyCLIP-ViT-40M (frozen)
- Vision Projection: Pre-trained from Stage 1 (trainable)
- Language Model: Llama 3.2 1B (fine-tuned)
- Task: Instruction following with visual context
```

## 🧪 Ablation Studies

### Comprehensive Experimental Design
This study includes 19 different experimental configurations testing various combinations of:

### Dataset Sampling Strategy
- [x] **10kRandom**: Random sampling from filtered dataset (10k samples)
- [x] **20kUmap**: UMAP-based clustering with strategic sampling (20k samples)

### Architecture Variants
- [x] **Dense Connector**: 4-layer dense projection network
- [x] **No Dense Connector**: Direct feature mapping
- [x] **Chat Template**: Structured conversation formatting (alpindale/Llama-3.2-1B-Instruct)
- [x] **No Chat Template**: Raw instruction-response pairs

### Training Configurations
- [x] **Learning Rate Scheduling**:
  - Linear Scheduler with Warmup
  - Cosine Scheduler with Warmup
- [x] **Training Duration**: 3, 6, 9 epochs
- [x] **Learning Rate**: 2e-5 (consistent across all experiments)
- [x] **Batch Sizes**: 1, 8, 12 (adapted to GPU memory constraints)
- [x] **Hardware Configurations**: 
  - A100 10GB RAM
  - A100 26.4GB RAM  
  - A100 37GB RAM

### Experimental Matrix
| Exp ID | Dataset Type | Dense Connector | Chat Template | Scheduler | Epochs | Learning Rate | Batch Size | GPU Config |
|--------|--------------|-----------------|---------------|-----------|--------|---------------|------------|------------|
| 1      | 20kUmap      | ❌              | ❌            | Linear + Warmup | 3 | 2e-5 | 1 | A100 10GB |
| 2      | 20kUmap      | ❌              | ❌            | Linear + Warmup | 3 | 2e-5 | 12 | A100 37GB |
| 3      | 20kUmap      | ❌              | ❌            | Linear + Warmup | 6 | 2e-5 | 12 | A100 37GB |
| 4      | 20kUmap      | ❌              | ❌            | Cosine + Warmup | 3 | 2e-5 | 12 | A100 37GB |
| 5      | 20kUmap      | ❌              | ❌            | Cosine + Warmup | 6 | 2e-5 | 12 | A100 37GB |
| 6      | 20kUmap      | ❌              | ✅            | Cosine + Warmup | 6 | 2e-5 | 8 | A100 10GB |
| 7      | 20kUmap      | ❌              | ✅            | Cosine + Warmup | 9 | 2e-5 | 8 | A100 10GB |
| 8      | 10kRandom    | ❌              | ❌            | Linear + Warmup | 3 | 2e-5 | 8 | A100 26.4GB |
| 9      | 10kRandom    | ❌              | ❌            | Linear + Warmup | 6 | 2e-5 | 8 | A100 26.4GB |
| 10     | 10kRandom    | ❌              | ❌            | Cosine + Warmup | 3 | 2e-5 | 12 | A100 37GB |
| 11     | 10kRandom    | ❌              | ❌            | Cosine + Warmup | 6 | 2e-5 | 12 | A100 37GB |
| 12     | 10kRandom    | ❌              | ✅            | Linear + Warmup | 6 | 2e-5 | 8 | A100 37GB |
| 13     | 10kRandom    | ❌              | ✅            | Cosine + Warmup | 6 | 2e-5 | 8 | A100 37GB |
| 14     | 10kRandom    | ✅ (4 layers)   | ✅            | Linear + Warmup | 3 | 2e-5 | 8 | A100 37GB |
| 15     | 10kRandom    | ✅ (4 layers)   | ✅            | Linear + Warmup | 6 | 2e-5 | 8 | A100 37GB |
| 16     | 10kRandom    | ✅ (4 layers)   | ✅            | Linear + Warmup | 9 | 2e-5 | 8 | A100 37GB |
| 17     | 10kRandom    | ✅ (4 layers)   | ✅            | Cosine + Warmup | 3 | 2e-5 | 8 | A100 26.4GB |
| 18     | 10kRandom    | ✅ (4 layers)   | ✅            | Cosine + Warmup | 6 | 2e-5 | 8 | A100 26.4GB |
| 19     | 10kRandom    | ✅ (4 layers)   | ✅            | Cosine + Warmup | 9 | 2e-5 | 8 | A100 26.4GB |


## 📈 Experimental Results

### Stage 1: Vision Projection Training
Complete experimental matrix with 19 configurations testing various combinations of dataset sampling, architecture variants, and training configurations.

### Stage 2: Fine-Tuning Experiments
| Exp ID | Dataset Type | Dense Connector | Chat Template | Scheduler | Epochs | Learning Rate | Batch Size | GPU Config |
|--------|--------------|-----------------|---------------|-----------|--------|---------------|------------|------------|
| 3      | 20kUmap      | ❌              | ❌            | Linear + Warmup | 3 (Early Stop) | 2e-5 | 3 | A100 28GB |
| 7      | 20kUmap      | ❌              | ✅            | Cosine + Warmup | 3 (Early Stop) | 2e-5 | 3 | A100 28GB |
| 9      | 10kRandom    | ❌              | ❌            | Linear + Warmup | 3 (Early Stop) | 2e-5 | 3 | A100 28GB |
| 13     | 10kRandom    | ❌              | ✅            | Cosine + Warmup | 3 (Early Stop) | 2e-5 | 3 | A100 28GB |
| 16     | 10kRandom    | ✅ (4 layers)   | ✅            | Linear + Warmup | 3 | 2e-5 | 3 | A100 28GB |
| 19     | 10kRandom    | ✅ (4 layers)   | ✅            | Cosine + Warmup | 3 | 2e-5 | 3 | A100 28GB |

### Experiment Tracking
All experiments are tracked using Weights & Biases (wandb):
- **Project**: `multimodal-llama` and `multimodal-llama-dense-connector`  
- **Organization**: `abdan-hafidz-institut-teknologi-sepuluh-nopember`


*This project is part of ongoing research in vision-language understanding and instruction following capabilities.*
