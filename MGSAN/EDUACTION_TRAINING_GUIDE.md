# MGSAN Training on EduAction Dataset

## Dataset Info
- **Classes**: 7 (drinking, lecture, play_phone, sleeping, talking, watch_computer, writing)
- **Samples**: 350 total (50 per class)
- **Keypoints**: 133 (COCO-WholeBody format)
- **Coordinates**: 2D (x, y)

## Train/Test Split
| Parameter | Value | Note |
|-----------|-------|------|
| train_ratio | 0.7 | Same as DDNet |
| seed | 42 | Same as DDNet |
| Split method | **Stratified** | Balanced classes, same as DDNet |

**Result**: 245 train (35/class) / 105 test (15/class)

## Available Configurations

| Config | Keypoints | Command |
|--------|-----------|---------|
| Full body | 133 | `--config ./config/eduaction/default.yaml` |
| Upper body | 127 | `--config ./config/eduaction/upper_body.yaml` |
| Body + Hands | 65 | `--config ./config/eduaction/body_hands.yaml` |
| Face + Hands | 110 | `--config ./config/eduaction/face_hands.yaml` |
| Hands only | 42 | `--config ./config/eduaction/hands_only.yaml` |
| Body only | 23 | `--config ./config/eduaction/body_only.yaml` |

## Quick Start

```bash
cd MGSAN

# Full body (133 keypoints)
python train_eduaction.py --config ./config/eduaction/default.yaml --phase train

# Body + Hands (65 keypoints) - recommended for classroom actions
python train_eduaction.py --config ./config/eduaction/body_hands.yaml --phase train

# Hands only (42 keypoints)
python train_eduaction.py --config ./config/eduaction/hands_only.yaml --phase train
```

## Keypoint Subsets

```
full_body (133):     Body(0-16) + Feet(17-22) + Face(23-90) + Hands(91-132)
upper_body (127):    Body(0-16) + Face(23-90) + Hands(91-132)
body_only (23):      Body(0-16) + Feet(17-22)
body_hands (65):     Body(0-22) + Hands(91-132)
face_hands (110):    Face(23-90) + Hands(91-132)
hands_only (42):     Left hand(91-111) + Right hand(112-132)
```

## Custom Data Path

Edit config file or use command line:
```bash
python train_eduaction.py --config ./config/eduaction/default.yaml \
  --train-feeder-args data_dir=/your/path/to/EduAction_pose_data \
  --test-feeder-args data_dir=/your/path/to/EduAction_pose_data
```

## Training Parameters

| Parameter | Value |
|-----------|-------|
| batch_size | 16 |
| num_epoch | 120 |
| base_lr | 0.01 |
| weight_decay | 0.0004 |
| optimizer | Adam |
| window_size | 64 frames |
| warm_up_epoch | 5 |

## Output

Training results saved to `./work_dir/eduaction/mgsan_<config>/`:
- `runs-<epoch>-<step>.pt` - Model weights
- `log.txt` - Training log
- `config.yaml` - Config backup
- `epoch*_each_class_acc.csv` - Per-class accuracy
