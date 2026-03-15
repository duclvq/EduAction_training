import torch
import cv2
import numpy as np
import math
import torchvision.transforms as transforms
from ultralytics import YOLO

from model import ViTPose
from debug_model import smart_load_weights


CHECKPOINT_PATH = "New folder/vitpose_base_coco_aic_mpii (1).pth"
INPUT_IMAGE_PATH = "datasets/my_classroom_data/image/img01A_0-19-40.jpg"
OUTPUT_IMAGE_PATH = "ket_qua_final.jpg"

# Tham số Model
MODEL_SIZE = 'base'
YOLO_MODEL = 'yolo11x-pose.pt'
INPUT_SIZE = (192, 256)

CONF_THRESHOLD = 0.15

VIS_THRESHOLD = 0.25

IOU_THRESHOLD = 0.6
CONTAINMENT_THRESH = 0.85


FAKE_SCORE = 0.55
COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (169, 169, 169)
COLOR_POINT = (0, 0, 255)

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]





def impute_missing_legs(keypoints, scores, threshold=0.3):

    imputed_kpts = list(keypoints)
    imputed_scores = list(scores)

    limbs = [
        {'side': 'left', 's': 5, 'h': 11, 'k': 13, 'a': 15},
        {'side': 'right', 's': 6, 'h': 12, 'k': 14, 'a': 16}
    ]

    for l in limbs:
        if scores[l['h']] > threshold and scores[l['s']] > threshold:

            sx, sy = keypoints[l['s']]
            hx, hy = keypoints[l['h']]
            torso_len = math.sqrt((sx - hx) ** 2 + (sy - hy) ** 2)
            direction = 1 if l['side'] == 'left' else -1  # 1: Mở sang phải ảnh, -1: Mở sang trái ảnh

            # --- SUY LUẬN GỐI  ---
            if scores[l['k']] < threshold:

                spread_factor = 0.2

                thigh_len_factor = 0.6
                kx_pred = hx + (torso_len * spread_factor * direction)
                ky_pred = hy + (torso_len * thigh_len_factor)

                imputed_kpts[l['k']] = (int(kx_pred), int(ky_pred))
                imputed_scores[l['k']] = FAKE_SCORE

            # --- SUY LUẬN CỔ CHÂN (ANKLE) ---
            kx, ky = imputed_kpts[l['k']]
            if scores[l['a']] < threshold:
                # Cổ chân: Hơi thu vào
                tuck_factor = 0.05

                shin_len_factor = 0.8

                # direction * -1 nghĩa là đi ngược hướng mở gối -> Thu vào trong
                ax_pred = kx + (torso_len * tuck_factor * (direction * -1))
                ay_pred = ky + (torso_len * shin_len_factor)

                imputed_kpts[l['a']] = (int(ax_pred), int(ay_pred))
                imputed_scores[l['a']] = FAKE_SCORE

    return imputed_kpts, imputed_scores


def draw_skeleton(img, keypoints, scores):

    for p1, p2 in SKELETON_CONNECTIONS:
        if p1 < 5 or p2 < 5:
            continue
        if scores[p1] > 0.1 and scores[p2] > 0.1:
            is_fake = (abs(scores[p1] - FAKE_SCORE) < 0.01 or abs(scores[p2] - FAKE_SCORE) < 0.01)

            if not is_fake and (scores[p1] < VIS_THRESHOLD or scores[p2] < VIS_THRESHOLD):
                continue

            color = COLOR_FAKE if is_fake else COLOR_REAL
            thickness = 2 if is_fake else 2
            cv2.line(img, keypoints[p1], keypoints[p2], color, thickness)

    for i, (x, y) in enumerate(keypoints):
        if i < 5:
            continue
        # Kiểm tra ngưỡng cho khớp
        thresh = 0.15 if i > 10 else VIS_THRESHOLD  # Chân chấp nhận thấp hơn

        is_fake = abs(scores[i] - FAKE_SCORE) < 0.01

        if scores[i] > thresh or is_fake:
            color = COLOR_FAKE if is_fake else COLOR_POINT
            radius = 3 if color == COLOR_FAKE else 4
            cv2.circle(img, (x, y), radius, color, -1)

def filter_boxes(boxes, iou_thresh=0.6, contain_thresh=0.85):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        remaining = []
        for b in boxes:
            x1 = max(current[0], b[0])
            y1 = max(current[1], b[1])
            x2 = min(current[2], b[2])
            y2 = min(current[3], b[3])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = (current[2] - current[0]) * (current[3] - current[1])
            box2_area = (b[2] - b[0]) * (b[3] - b[1])
            union_area = box1_area + box2_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            intersection_ratio = inter_area / min(box1_area, box2_area) if min(box1_area, box2_area) > 0 else 0
            if iou < iou_thresh and intersection_ratio < contain_thresh:
                remaining.append(b)
        boxes = remaining
    return keep


def box_to_center_scale(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    box_aspect_ratio = h / w
    new_w = w * 1.25
    height_multiplier = 2.2 if box_aspect_ratio < 1.5 else 3.0
    estimated_full_height = new_w * height_multiplier
    new_h = max(h * 1.1, estimated_full_height)
    new_w = new_h * 0.75
    pad_top = new_h * 0.08
    new_y1 = y1 - pad_top
    new_y2 = new_y1 + new_h
    cx = x1 + w / 2
    new_x1, new_x2 = cx - new_w / 2, cx + new_w / 2
    return max(0, int(new_x1)), max(0, int(new_y1)), min(img_w, int(new_x2)), int(new_y2)


def process_pose(model, crop_img):
    input_tensor = cv2.resize(crop_img, INPUT_SIZE)
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(input_tensor).unsqueeze(0)
    if torch.cuda.is_available(): input_tensor = input_tensor.cuda()
    with torch.no_grad(): heatmaps = model(input_tensor)
    return heatmaps


def get_keypoints_global(heatmaps, box_coords, crop_size):
    x1, y1 = box_coords
    crop_w, crop_h = crop_size
    heatmaps_np = heatmaps.detach().cpu().numpy()
    keypoints, scores = [], []
    for i in range(17):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmaps_np[0, i, :, :])
        px, py = max_loc[0] * 4, max_loc[1] * 4
        real_cx, real_cy = px * (crop_w / INPUT_SIZE[0]), py * (crop_h / INPUT_SIZE[1])
        keypoints.append((int(real_cx + x1), int(real_cy + y1)))
        scores.append(max_val)
    return keypoints, scores


def pose_nms(all_results, dist_threshold=30):
    if not all_results: return []
    all_results = sorted(all_results, key=lambda x: x['avg_score'], reverse=True)
    keep = []
    for current in all_results:
        is_duplicate = False
        curr_kpts = current['keypoints']
        curr_center = np.mean([curr_kpts[5], curr_kpts[6], curr_kpts[11], curr_kpts[12]], axis=0)
        for kept_item in keep:
            kept_kpts = kept_item['keypoints']
            kept_center = np.mean([kept_kpts[5], kept_kpts[6], kept_kpts[11], kept_kpts[12]], axis=0)
            if np.linalg.norm(curr_center - kept_center) < dist_threshold:
                is_duplicate = True
                break
        if not is_duplicate: keep.append(current)
    return keep



def main():
    print(" Running Classsroom Pose Estimation ")
    yolo = YOLO(YOLO_MODEL)
    vit_model = ViTPose(model_size=MODEL_SIZE, num_joints=17, img_size=(256, 192))
    vit_model = smart_load_weights(vit_model, CHECKPOINT_PATH)
    vit_model.eval()
    if torch.cuda.is_available(): vit_model.cuda()

    frame = cv2.imread(INPUT_IMAGE_PATH)
    if frame is None: return
    H, W = frame.shape[:2]

    results = yolo(frame, verbose=False, conf=CONF_THRESHOLD)
    raw_boxes = []
    for b in results[0].boxes:
        if int(b.cls[0]) == 0:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            raw_boxes.append([x1, y1, x2, y2, float(b.conf[0])])

    final_boxes = filter_boxes(raw_boxes, iou_thresh=IOU_THRESHOLD, contain_thresh=CONTAINMENT_THRESH)
    print(f" Found {len(final_boxes)} people.")

    all_skeletons = []
    for box_data in final_boxes:
        box = box_data[:4]
        nx1, ny1, nx2, ny2 = box_to_center_scale(box, W, H)
        crop_img = frame[ny1:min(ny2, H), nx1:nx2]
        if crop_img.size == 0: continue

        heatmaps = process_pose(vit_model, crop_img)
        kpts, scores = get_keypoints_global(heatmaps, (nx1, ny1), (crop_img.shape[1], crop_img.shape[0]))
        kpts, scores = impute_missing_legs(kpts, scores,
                                           threshold=VIS_THRESHOLD)  # Dùng ngưỡng VIS_THRESHOLD để quyết định có impute hay không

        avg_score = np.mean([scores[5], scores[6], scores[11], scores[12]])
        all_skeletons.append({'keypoints': kpts, 'scores': scores, 'avg_score': avg_score})

    unique_skeletons = pose_nms(all_skeletons, dist_threshold=40)
    for item in unique_skeletons:
        draw_skeleton(frame, item['keypoints'], item['scores'])

    cv2.imwrite(OUTPUT_IMAGE_PATH, frame)
    print(f" Done! Result saved to {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()