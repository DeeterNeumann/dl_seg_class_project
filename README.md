# dl_seg_class_project

# MoNuSAC Nucleus Segmentation and Classification

Whole Slide Histopathology Inference with Deep Learning

**Author:** Deeter Neumann

# Project Overview
This project develps and deploys a deep learning pipeline for nucleus segmentation and cell type classification in H&E-stained histopathology images. The model is trained on the MoNuSAC dataset and performs inference on whole slide immages (WSIs) using tiled processing and Gaussian blending.

The system demonstrates a full translational pipeline:
- Data preprocessing and normalization
- Multi-head segmentation (semantic and ternary (boundary))
- Whole slide tiling inference
- Interactive deployment via Hugging Face Spaces

# Live Demo:
https://huggingface.com/spaces/drneumann/CellClassification

# Objectives
- Segment nuclei in histopathology images
- Enable inference on large WSIs via tiling
- Provide an accessible web interface
- Demonstrate end-to-end ML deployment

# Cell Types Detected

Class
- Epithelial
- Lymphocyte
- Neutrophil
- Macrophage

# Model Architecture
- Backbone: ResNet Encoder (via segmentation_models_pytorch)
- Decoder: U-Net style
- Heads:
    - Semantic segmentation (5 classes (including background))
    - Ternary map (inside / boundary / background)

Why multi-head?
Boundary prediction improves instance separation and segmentation accuracy.

# Dataset
Training
- MoNuSAC 2020 - multi-organ nuclei segmentation dataset
- H&E stained histopathology
- 40x magnification
Evaluation / Demo
- MoNuSAC test WSIs

# Pipeline
1) Preprocessing
   - RGB conversion
   - ImageNet normalization
2) Whole Slide Inference
   - Tile size: 256x256
   - Overlap: 64 px
   - Gaussian blending to reduce edge artifacts
3) Post-processing
   - Argmax class prediction
   - Color overlay rendering
   - Interactive visualization

# Deployment
This project is deployed using:
- Docker
- Gradio UI
- Hugging Face Spaces

Hugging Face Spaces provides:
- Reproducible deployment
- Public accessibility
- Model hosting

# Running Locally

1) clone repo
git clone https://github.com/DeeterNeumann/dl_seg_class_project.git
cd dl_seg_class_project/deploy

2) Install dependencies
pip install -r requirements.txt

3) Run app
python app.py

Open: http://localhost:7860

# Docker Deployment
docker build -t cell-seg .
docker run -p 7860:7860 cell-seg

# Example Workflow
1) Upload H&E image
2) Model tiles and processes image
3) Semantic overlay shows cell types
4) Ternary overlay shows nucleus boundaries

# Results & Observations
- Boundary-aware segmentation improves nucleus separation
- Performance generalizes to MoNuSAC with minor degradation
- Whole slide tiling enables scalable inference

# Limitations
- Trained on MoNuSAC -> domain shift on other tissues
- No stain normalization
- CPU inference on Spaces limits WSI size
- Instance segmentation not fully implemented

# Future Work
- GPU deployment for faster WSI inference
- Stain normalization
- Instance segmentation via watershed or HoVer-Net approach
- Quantitative TIL scoring

# Repository Structure

dl_seg_class_project/
├── deploy/                     # Hugging Face Spaces app (Docker + Gradio UI)
│   ├── app.py                  # Web interface for WSI inference
│   ├── inference.py            # Tiled inference & overlay logic
│   ├── model.py                # Model architecture & loading
│   ├── requirements.txt        # Deployment dependencies
│   └── weights/                # Model checkpoints (Git LFS)
│
├── scripts/                    # Data preparation & experiment utilities
│   ├── export_manifest_dataset.py
│   └── other pipeline helpers
│
├── assets/                     # Project assets (configs, metadata)
│
├── dh_train_immune_boost.py    # Primary training script (multi-head model)
├── dh_train_immbst_terwt.py    # Training variant (boundary weighting ablation)
├── summarize_runs.py           # Training metrics aggregation
├── generate_summary_doc.py     # Automated results report generation
├── download_from_lightning.sh  # Artifact retrieval from Lightning runs
├── training_run_summary.pdf    # Capstone results summary
│
├── .gitignore
└── README.md
   
