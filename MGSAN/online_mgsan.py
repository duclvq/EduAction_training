import os
import sys
import random
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import yaml
from collections import OrderedDict
from numpy.lib.stride_tricks import as_strided

# Seed for reproducibility
SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

POSE_PATH = r"pose_new_v2"


def import_class(import_str):
    """Import class từ string"""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found in {mod_str}')


def load_mgsan_model(config_path, weights_path, device='cpu'):
    """
    Load MGSAN model từ config và weights file
    """
    # Đọc config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Import model class
    Model = import_class(config['model'])
    model_args = config['model_args']
    
    # Khởi tạo model
    print(f"Khởi tạo model: {config['model']}")
    print(f"Model args: {model_args}")
    model = Model(**model_args)
    
    # Load weights
    print(f"Loading weights từ: {weights_path}")
    weights = torch.load(weights_path, map_location=device)
    weights = OrderedDict([[k.split('module.')[-1], v] for k, v in weights.items()])
    
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    
    return model, config


def create_windows(np_pose, window_size=64):
    """Tạo sliding windows từ pose data"""
    if len(np_pose) < window_size:
        # Padding nếu video ngắn hơn window_size
        pad_length = window_size - len(np_pose)
        np_pose = np.concatenate([np_pose, np.zeros((pad_length,) + np_pose.shape[1:])])
    
    stride = np_pose.strides[0]
    shape = (np_pose.shape[0] - window_size + 1, window_size) + np_pose.shape[1:]
    strides = (stride,) + np_pose.strides
    windows = as_strided(np_pose, shape=shape, strides=strides)
    return windows


def read_txt(filename):
    """Đọc file txt line by line"""
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def preprocess_pose_for_mgsan(pose_windows, config):
    """
    Chuyển đổi pose data sang format phù hợp với MGSAN
    Input: pose_windows shape (num_windows, frames, joints, coords)
    Output: tensor shape (num_windows, channels, frames, joints, persons)
    """
    num_windows = len(pose_windows)
    num_frames = pose_windows.shape[1]
    num_joints = pose_windows.shape[2]
    num_coords = pose_windows.shape[3] if len(pose_windows.shape) > 3 else 3
    
    # MGSAN expects: (batch, channels=3, frames, joints, persons)
    # Reshape: (num_windows, frames, joints, coords) -> (num_windows, coords, frames, joints, 1)
    mgsan_input = np.transpose(pose_windows, (0, 3, 1, 2))  # (N, coords, frames, joints)
    mgsan_input = np.expand_dims(mgsan_input, axis=-1)  # (N, coords, frames, joints, 1)
    
    return torch.from_numpy(mgsan_input).float()


def predict_mgsan(model, device, pose_windows, batch_size=16):
    """
    Chạy inference MGSAN trên pose windows
    """
    model.eval()
    pred_list = []
    
    num_windows = len(pose_windows)
    
    with torch.no_grad():
        for i in tqdm(range(0, num_windows, batch_size), desc="Inference"):
            batch = pose_windows[i:i+batch_size].to(device)
            
            # Forward pass
            output = model(batch)
            
            # Get predictions
            pred = output.argmax(dim=1)
            pred_list.extend(pred.cpu().numpy().tolist())
    
    return pred_list


def main():
    # Cấu hình
    CONFIG_PATH = "./config/cobot/default.yaml"
    WEIGHTS_PATH = "./work_dir/cobot/ctrgcn_joint/runs-97-10282.pt"
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    
    print(f"Sử dụng device: {device}")
    
    # Load MGSAN model
    print("\n=== Loading MGSAN model ===")
    model, config = load_mgsan_model(CONFIG_PATH, WEIGHTS_PATH, device)
    print("Model loaded successfully!")
    
    # Get window size from config or use default
    window_size = 64  # MGSAN typically uses 64 frames
    batch_size = 1
    
    # Tạo output folder
    exp_name = 'mgsan_online'
    output_dir = f'mgsan_predictions/{exp_name}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Đọc danh sách test files
    test_list = read_txt('test_person.txt')
    pose_filename_list = os.listdir(POSE_PATH)
    
    print(f"\nTest list: {len(test_list)} subjects")
    print(f"Total pose files: {len(pose_filename_list)}")
    
    # Process each video
    count = 0
    for name in pose_filename_list:
        # Chỉ xử lý files trong test_list
        if not name[:name.rindex('_')] in test_list:
            continue
        
        print(f"\n[{count+1}] Processing: {name}")
        
        # Load pose data
        pose_path = os.path.join(POSE_PATH, name)
        np_pose = np.load(pose_path)
        print(f"  Pose shape: {np_pose.shape}")
        
        # Tạo sliding windows
        sample_windows = create_windows(np_pose, window_size=window_size)
        print(f"  Windows created: {len(sample_windows)}")
        
        # Preprocess cho MGSAN
        mgsan_input = preprocess_pose_for_mgsan(sample_windows, config)
        print(f"  MGSAN input shape: {mgsan_input.shape}")
        
        # Chạy inference
        pred_results = predict_mgsan(model, device, mgsan_input, batch_size=batch_size)
        print(f"  Predictions: {len(pred_results)}")
        
        # Lưu kết quả
        # Save as txt
        txt_path = os.path.join(output_dir, f"{name[:-4]}.txt")
        with open(txt_path, "w") as f:
            for pred in pred_results:
                f.write(str(pred) + '\n')
        
        # Save as npy
        npy_path = os.path.join(output_dir, f"{name[:-4]}.npy")
        np.save(npy_path, np.array(pred_results))
        
        print(f"  ✓ Saved to {output_dir}/{name[:-4]}.*")
        count += 1
    
    print(f"\n=== Completed! Processed {count} videos ===")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
