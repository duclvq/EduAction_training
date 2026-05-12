# EduAction Training

Skeleton-based classroom action recognition using 5-fold cross-validation. Compares three GCN architectures (ST-GCN, MGSAN, DDNet) across multiple keypoint extraction methods on the EduAction dataset.

## Dataset

**EduAction** — 7 classroom action classes: `drinking`, `lecture`, `play_phone`, `sleeping`, `talking`, `watch_computer`, `writing`

**Download:** [Google Drive](https://drive.google.com/drive/folders/1CTIp5lT7Yq3YYPtUhrgvUbx5F-ZAnglT?usp=sharing)

Keypoint formats supported:
- **MediaPipe 133kp** — full body (face + hands + body)
- **MediaPipe 127kp** — upper body
- **MediaPipe 17kp** — body only (COCO-17 format)
- **ViTPose 17kp** — body only (COCO-17 format)

## Models

| Directory | Model | Description |
|-----------|-------|-------------|
| `st-gcn/` | ST-GCN | Spatial Temporal Graph Convolutional Network |
| `MGSAN/` | MGSAN | Multi-Graph Spatial Attention Network |
| `MODEL_DDNET/` | DDNet | Dual-feature Difference Network |

## Results (5-Fold Cross-Validation)

Evaluation: 5-fold CV, seed=42, 64-frame window, 7 classes.

| Rank | Model | Keypoints | Mean Acc | Std | Train Time |
|------|-------|-----------|----------|-----|------------|
| 1 | ST-GCN | MediaPipe 17kp | **80.57%** | ±4.10% | ~103 min |
| 1 | MGSAN | MediaPipe 17kp | **80.57%** | ±3.08% | ~218 min |
| 3 | MGSAN | MediaPipe 127kp | 79.71% | ±1.67% | ~88 min |
| 3 | MGSAN | ViTPose 17kp | 79.71% | ±1.67% | ~237 min |
| 5 | ST-GCN | ViTPose 17kp | 79.43% | ±3.68% | ~114 min |
| 6 | MGSAN | MediaPipe 133kp | 77.14% | ±1.81% | ~147 min |
| 6 | DDNet | ViTPose 17kp | 77.14% | ±4.78% | ~6 min |
| 8 | ST-GCN | MediaPipe 133kp | 76.57% | ±2.65% | ~30 min |
| 9 | DDNet | MediaPipe 127kp | 76.29% | ±5.08% | ~7 min |
| 10 | DDNet | MediaPipe 17kp | 76.00% | ±4.18% | ~3 min |
| 11 | DDNet | MediaPipe 133kp | 74.29% | ±5.92% | ~7 min |
| 12 | ST-GCN | MediaPipe 127kp | 71.14% | ±4.73% | ~28 min |

## Project Structure

```
EduAction_training/
├── MGSAN/
│   ├── config/eduaction/       # Training configs (kfold, upper body, etc.)
│   ├── feeders/                # Data loaders
│   ├── graph/                  # Skeleton graph definitions
│   ├── model/                  # MGSAN model implementation
│   ├── train_kfold.py          # K-fold training script
│   └── work_dir/               # Results and checkpoints
├── st-gcn/
│   ├── config/st_gcn/eduaction/
│   ├── feeder/
│   ├── net/
│   ├── processor/
│   ├── train_kfold.py
│   └── work_dir/
├── MODEL_DDNET/
│   ├── ddnet.py
│   ├── train_kfold.py
│   └── work_dir/
├── kfold_visualization/        # Generated comparison charts
├── visualize_kfold_results.py  # Visualization script
└── visualize_all_results.py
```

## Training

Each model directory contains a `train_kfold.py` script. Example:

```bash
# ST-GCN
cd st-gcn
python train_kfold.py --config config/st_gcn/eduaction/train_kfold.yaml

# MGSAN (joint stream)
cd MGSAN
python train_kfold.py --config config/eduaction/kfold.yaml

# MGSAN (upper body)
python train_kfold.py --config config/eduaction/kfold_upper.yaml

# DDNet
cd MODEL_DDNET
python train_kfold.py
```

## Visualization

After training all models, generate comparison charts:

```bash
python visualize_kfold_results.py
```

Outputs saved to `kfold_visualization/`:
- Accuracy comparison bar chart
- Per-fold result lines
- Per-class accuracy breakdown
- Confusion matrices
- Training time comparison

## Per-Class Accuracy (Best Configurations)

| Class | ST-GCN 17kp | MGSAN 17kp | DDNet ViTPose |
|-------|-------------|------------|---------------|
| drinking | ~88% | ~88% | ~80% |
| lecture | ~84% | ~84% | — |
| watch_computer | ~84% | ~84% | ~100% |
| talking | ~68% | ~72% | — |
