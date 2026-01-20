# EduAction Training Configuration Guide

## Tổng quan

File `train_eduaction.py` hỗ trợ **cấu hình linh hoạt** để thử nghiệm với các tổ hợp keypoint khác nhau.

## Data Split Strategy (Chiến lược chia dữ liệu)

**NEW** - Random Split với Fixed Seed:
- **70% Training** (245 videos từ tất cả các class)
- **30% Testing** (105 videos từ tất cả các class)
- **15% Validation** (từ 70% training → ~37 videos)
- **Random seed = 42** (đảm bảo split giống nhau mỗi lần chạy)
- **Stratified split** (giữ tỷ lệ class đồng đều)

Mỗi class có 50 videos → Split thành:
- Train: 35 videos (70%)
- Test: 15 videos (30%)
- Validation: ~5 videos (từ 35 train videos)

**OLD** (deprecated):
- Train: Videos 1-40
- Test: Videos 41-50
- Vấn đề: Một số class không có đủ 40 videos trong phạm vi 1-40

### Các nhóm keypoint có thể bật/tắt:

1. **Legs** (10 keypoints)
   - Knees (2), Ankles (2), Feet (6)
   - `use_leg_keypoints = True/False`

2. **Face** (68 keypoints)
   - Face landmarks chi tiết
   - `use_face_keypoints = True/False`

3. **Hands** (42 keypoints)
   - Left hand (21) + Right hand (21)
   - `use_hand_keypoints = True/False`

4. **Body** (13-23 keypoints)
   - Luôn được bao gồm (nose, eyes, ears, shoulders, elbows, wrists, hips)
   - Số lượng thay đổi tùy theo `use_leg_keypoints`

---

## Các cấu hình phổ biến

| Config Name | Legs | Face | Hands | Total Keypoints | Use Case |
|-------------|------|------|-------|-----------------|----------|
| **Full Body** | ✅ | ✅ | ✅ | 133 | Baseline, sử dụng toàn bộ thông tin |
| **Upper Body** | ❌ | ✅ | ✅ | 123 | Loại bỏ noise từ chân |
| **Body + Hands** | ✅ | ❌ | ✅ | 65 | Tập trung vào động tác tay |
| **Body + Hands (no legs)** | ❌ | ❌ | ✅ | 55 | Động tác tay, không cần face/legs |
| **Body Only** | ✅ | ❌ | ❌ | 23 | Minimal config, pose đơn giản |
| **Body Only (no legs)** | ❌ | ❌ | ❌ | 13 | Ultra minimal, chỉ upper body |

---

## Cấu trúc Keypoints (133 keypoints)

```
0-16:   Body (COCO format)
        0: nose
        1-2: eyes (left, right)
        3-4: ears (left, right)
        5-6: shoulders (left, right)
        7-8: elbows (left, right)
        9-10: wrists (left, right)
        11-12: hips (left, right)
        13-14: knees (left, right) ❌ EXCLUDED in upper body config
        15-16: ankles (left, right) ❌ EXCLUDED in upper body config

17-22:  Feet ❌ EXCLUDED in upper body config

23-90:  Face (68 keypoints) ✅ INCLUDED

91-111: Left hand (21 keypoints) ✅ INCLUDED

112-132: Right hand (21 keypoints) ✅ INCLUDED
```

---

## Cách sử dụng

### Option 1: Sửa trực tiếp trong code

Mở file `train_eduaction.py` và tìm phần configuration:

```python
# ==================== CONFIGURATION ====================
USE_LEG_KEYPOINTS = False   # Include legs?
USE_FACE_KEYPOINTS = False  # Include face?
USE_HAND_KEYPOINTS = True   # Include hands?
# =======================================================
```

**Ví dụ các cấu hình:**

```python
# Full body (all 133 keypoints)
USE_LEG_KEYPOINTS = True
USE_FACE_KEYPOINTS = True
USE_HAND_KEYPOINTS = True

# Upper body only (no legs)
USE_LEG_KEYPOINTS = False
USE_FACE_KEYPOINTS = True
USE_HAND_KEYPOINTS = True

# Body + Hands only (no face)
USE_LEG_KEYPOINTS = True
USE_FACE_KEYPOINTS = False
USE_HAND_KEYPOINTS = True

# Minimal (body only, no legs/face/hands)
USE_LEG_KEYPOINTS = False
USE_FACE_KEYPOINTS = False
USE_HAND_KEYPOINTS = False
```

### Option 2: Chạy từ Python script

```python
from train_eduaction import train_eduaction_model

# Full body
model, history, accuracy = train_eduaction_model(
    use_leg_keypoints=True,
    use_face_keypoints=True,
    use_hand_keypoints=True
)

# Upper body only
model, history, accuracy = train_eduaction_model(
    use_leg_keypoints=False,
    use_face_keypoints=True,
    use_hand_keypoints=True
)

# Body + hands (no face)
model, history, accuracy = train_eduaction_model(
    use_leg_keypoints=False,
    use_face_keypoints=False,
    use_hand_keypoints=True
)
```

---

## Output Files

Các file output sẽ được **tự động đặt tên** theo config đã chọn:

### Naming Convention

Format: `EduAction_DDNet_{config_suffix}.h5`

Suffix được tạo từ các phần đã bật:
- `legs` - nếu `use_leg_keypoints = True`
- `face` - nếu `use_face_keypoints = True`
- `hands` - nếu `use_hand_keypoints = True`

### Ví dụ:

| Config | Suffix | Files |
|--------|--------|-------|
| Legs + Face + Hands | `legs_face_hands` | `EduAction_DDNet_legs_face_hands.h5` |
| Face + Hands | `face_hands` | `EduAction_DDNet_face_hands.h5` |
| Legs + Hands | `legs_hands` | `EduAction_DDNet_legs_hands.h5` |
| Hands only | `hands` | `EduAction_DDNet_hands.h5` |
| Body only | `body_only` | `EduAction_DDNet_body_only.h5` |

**Các file được tạo:**
- Model: `EduAction_DDNet_{suffix}.h5`
- History: `EduAction_training_history_{suffix}.pkl`
- Plots: `EduAction_training_plots_{suffix}.png`
- Confusion Matrix: `EduAction_confusion_matrix_{suffix}.png`

---

## So sánh Performance

| Config | Keypoints | feat_d | Khi nào dùng? |
|--------|-----------|--------|---------------|
| **Full Body** | 133 | 8,778 | Baseline, có đủ dữ liệu tốt |
| **No Legs** | 123 | 7,503 | Leg bị occlusion nhiều |
| **No Face** | 65 | 2,080 | Face không quan trọng cho action |
| **Hands Only** | 55 | 1,485 | Tập trung vào hand gestures |
| **Body Only** | 13-23 | 78-253 | Dataset rất nhỏ, cần tránh overfit |

### Gợi ý theo loại action

#### Actions cần Face (sleeping, lecture):
```python
USE_FACE_KEYPOINTS = True  # Face orientation quan trọng
```

#### Actions cần Hands (drinking, play_phone, writing):
```python
USE_HAND_KEYPOINTS = True  # Hand gestures quan trọng
```

#### Actions cần Legs (standing, walking):
```python
USE_LEG_KEYPOINTS = True  # Lower body position quan trọng
```

### Nguyên tắc chung:

1. **Dataset lớn (>1000 samples/class)**: Dùng full body
2. **Dataset nhỏ (<500 samples/class)**: Loại bỏ features không cần thiết
3. **High occlusion**: Loại bỏ phần bị che nhiều
4. **Specific actions**: Chỉ giữ lại keypoints liên quan

---

## Ví dụ chạy

```bash
# Train với upper body only
python train_eduaction.py

# Output sẽ hiển thị:
# >>> Training with UPPER BODY ONLY (excluding legs)
# Using upper body only (no legs): 123 keypoints
```

---

## Parameters khác có thể điều chỉnh

Trong class `EduActionConfig`:

```python
# Temporal
self.frame_l = 40  # Độ dài chuỗi frames (thử: 30, 40, 50, 60)

# Training
self.batch_size = 16  # Batch size (thử: 8, 16, 32)
self.epochs = 150  # Số epochs
self.learning_rate = 0.001  # Learning rate

# Data split
self.train_split = 0.7
self.val_split = 0.15
self.test_split = 0.15

# Oversampling
self.apply_oversampling = True  # Có balance classes không
```

---

## Tips

1. **Chạy cả 2 configs** để so sánh performance
2. **Monitor validation accuracy** - nếu upper body tốt hơn, chứng tỏ leg keypoints gây noise
3. **Kiểm tra confusion matrix** - xem class nào được cải thiện khi loại bỏ legs
4. **Dataset nhỏ** - upper body có thể tốt hơn do ít overfitting

---

## Troubleshooting

**Q: Model báo lỗi shape mismatch?**
A: Kiểm tra `feat_d` được tính đúng: `feat_d = joint_n * (joint_n - 1) / 2`

**Q: Tại sao accuracy thấp?**
A: Thử điều chỉnh `frame_l`, `batch_size`, hoặc enable/disable `apply_oversampling`

**Q: Muốn loại bỏ face/hand keypoints?**
A: Tương tự, sửa `keypoint_indices` trong `EduActionConfig.__init__()`
