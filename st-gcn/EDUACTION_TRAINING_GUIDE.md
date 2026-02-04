# ST-GCN Training Guide for EduAction Dataset

This guide explains how to train ST-GCN (Spatial Temporal Graph Convolutional Networks) on the EduAction dataset.

## Dataset Configuration

The EduAction dataset uses **COCO-WholeBody format with 133 keypoints**:
- **Body**: 0-16 (17 points)
- **Feet**: 17-22 (6 points)  
- **Face**: 23-90 (68 points)
- **Left Hand**: 91-111 (21 points)
- **Right Hand**: 112-132 (21 points)

### Data Split (Same as DDNet and MGSAN)
- **Train/Test Ratio**: 70% / 30%
- **Random Seed**: 42
- **Stratified Split**: Ensures balanced class distribution

### Classes (7 total)
1. drinking
2. lecture
3. play_phone
4. sleeping
5. talking
6. watch_computer
7. writing

## Available Configurations

### 1. Full Body (133 keypoints)
```bash
python main.py recognition -c config/st_gcn/eduaction/train.yaml
```
Uses all 133 keypoints including face, body, hands, and feet.

### 2. Upper Body (127 keypoints) - Recommended
```bash
python main.py recognition -c config/st_gcn/eduaction/train_upper.yaml
```
Body + face + hands, without feet. Same as MGSAN `upper_body` configuration.
Best for classroom actions where feet are often not visible.

### 3. Body Only (23 keypoints)
```bash
python main.py recognition -c config/st_gcn/eduaction/train_body.yaml
```
Fastest training with only body and feet keypoints.

### 4. Body + Hands (65 keypoints)
```bash
python main.py recognition -c config/st_gcn/eduaction/train_hands.yaml
```
Good for gesture-based actions, includes body and both hands.

## Quick Start

### 1. Update Data Path
Edit the config file to set your data path:
```yaml
train_feeder_args:
  data_dir: /path/to/EduAction_pose_data  # Your data path
```

### 2. Train the Model
```bash
# Using the wrapper script
python train_eduaction.py --config config/st_gcn/eduaction/train.yaml

# Or directly with main.py
python main.py recognition -c config/st_gcn/eduaction/train.yaml
```

### 3. Override Settings via Command Line
```bash
# Change batch size and learning rate
python main.py recognition -c config/st_gcn/eduaction/train.yaml \
    --batch_size 32 \
    --base_lr 0.005

# Use specific GPU
python main.py recognition -c config/st_gcn/eduaction/train.yaml \
    --device 0

# Quick test with debug mode
python main.py recognition -c config/st_gcn/eduaction/train.yaml \
    --train_feeder_args "debug=True"
```

### 4. Evaluate a Trained Model
```bash
python main.py recognitionUpper Body | Body Only | Body+Hands |
|-----------|-----------|------------|-----------|------------|
| Keypoints | 133 | 127 | 23 | 65 |
| Batch Size | 16 | 16 | 32 | 24 |
| Base LR | 0.01 | 0.01 | 0.01 | 0.01 |
| Epochs | 100 | 100 | 100 | 100 |
| LR Steps | [40,60,80]
| Parameter | Full Body | Body Only | Body+Hands |
|-----------|-----------|-----------|------------|
| Keypoints | 133 | 23 | 65 |
| Batch Size | 16 | 32 | 24 |
| Base LR | 0.01 | 0.01 | 0.01 |
| Epochs | 100 | 100 | 100 |
| LR Steps | [40,60,80] | [40,60,80] | [40,60,80] |

## File Structure

```
st-gcn/
├── config/st_gcn/eduaction/
│   ├── train.yaml        # Full body training (133 kpts)
│   ├── train_upper.yaml  # Upper body training (127 kpts) - Recommended
│   ├── train_body.yaml   # Body only training (23 kpts)
│   ├── train_hands.yaml  # Body + hands training (65 kpts)
│   └── test.yaml         # Evaluation config
├── feeder/
│   └── feeder_eduaction.py  # EduAction data loader
├── net/utils/
│   └── graph.py          # Graph definitions (includes eduaction layouts)
├── train_eduaction.py    # Training wrapper script
└── main.py               # Main entry point
```

## Graph Layouts

Four graph layouts are available for EduAction:

| Layout | Keypoints | Description |
|--------|-----------|-------------|
| `eduaction` | 133 | Full body with all connections |
| `eduaction_upper` | 127 | Body + face + hands (no feet) |
| `eduaction_body` | 23 | Body + feet only |
| `eduaction_hands` | 65 | Body + both hands (remapped indices) |

## Comparison with Other 127/65/23 | Original GCN-based method |
| MGSAN | PyTorch | 133/127 | Multi-scale attention |
| DD-Net | TensorFlow | 133 | Lightweight, fast |

All models use the **same data split** (70/30, seed=42) for fair comparison.

### Keypoint Subset Mapping

| Subset Name | ST-GCN Config | MGSAN Config | Keypoints |
|-------------|---------------|--------------|-----------|
| Full Body | `train.yaml` | `full_body` | 133 |
| Upper Body | `train_upper.yaml` | `upper_body` | 127 |
| Body Only | `train_body.yaml` | `body_only` | 23 |
| Body + Hands | `train_hands.yaml` | `body_hands` | 65 |
| MGSAN | PyTorch | 133 | Multi-scale attention |
| DD-Net | TensorFlow | 133 | Lightweight, fast |

All models use the **same data split** (70/30, seed=42) for fair comparison.

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in config
- Use `body_only` or `body_hands` layout with fewer keypoints

### Slow Training
- Use `body_only` config (23 keypoints vs 133)
- Enable `mmap=True` in feeder (default)

### Import Errors
Make sure you're in the st-gcn directory:
```bash
cd st-gcn
python main.py recognition -c config/st_gcn/eduaction/train.yaml
```

## Expected Results

Training on EduAction typically achieves:
- **Full Body (133 kpts)**: ~85-90% accuracy
- **Body Only (23 kpts)**: ~80-85% accuracy  
- **Body+Hands (65 kpts)**: ~85-88% accuracy

Results may vary based on hyperparameters and hardware.
