import os
import sys
import time
import torch
import yaml
from collections import OrderedDict


def import_class(import_str):
    """Import class từ string (ví dụ: 'model.MGSAN.Model')"""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found in {mod_str}')


def load_model_from_config(config_path, weights_path):
    """
    Load model từ config file và weights file (giống cách trong main.py)
    
    Args:
        config_path: đường dẫn đến file config (.yaml)
        weights_path: đường dẫn đến file weights (.pt)
    
    Returns:
        model: PyTorch model đã load weights
    """
    # Đọc config file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file không tồn tại: {config_path}")
    
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
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file không tồn tại: {weights_path}")
    
    print(f"Loading weights từ: {weights_path}")
    weights = torch.load(weights_path, map_location='cpu')
    
    # Xử lý weights (loại bỏ 'module.' prefix nếu có)
    weights = OrderedDict([[k.split('module.')[-1], v] for k, v in weights.items()])
    
    model.load_state_dict(weights)
    model.eval()
    
    return model, config


def load_sample_data(num_person, num_point, in_channels=3, frames=64, batch_size=1):
    """
    Tạo dữ liệu mẫu phù hợp với input của model
    
    Args:
        num_person: số người trong skeleton
        num_point: số khớp (joints) trong skeleton
        in_channels: số kênh input (thường là 3 cho x,y,z)
        frames: số frame
        batch_size: batch size
    
    Returns:
        Tensor với shape (batch_size, in_channels, frames, num_point, num_person)
    """
    return torch.randn(batch_size, in_channels, frames, num_point, num_person)


def inference(model, data, device='cpu'):
    """
    Chạy inference trên dữ liệu
    
    Args:
        model: PyTorch model
        data: input data tensor
        device: 'cpu' hoặc 'cuda'
    
    Returns:
        output: prediction từ model
        inference_time: thời gian chạy inference (giây)
    """
    model = model.to(device)
    model.eval()  # Đảm bảo model ở eval mode
    data = data.to(device)
    
    # Warm-up GPU nếu dùng CUDA
    if device == 'cuda':
        with torch.no_grad():
            _ = model(data)
        torch.cuda.synchronize()
    
    with torch.no_grad():
        start = time.time()
        output = model(data)
        if device == 'cuda':
            torch.cuda.synchronize()  # Đợi GPU hoàn thành
        end = time.time()
        inference_time = end - start
    
    return output, inference_time


if __name__ == "__main__":
    # Cấu hình
    config_path = "./config/cobot/default.yaml"
    weights_path = "./work_dir/cobot/ctrgcn_joint/runs-97-10282.pt"
    
    # Có thể đổi sang config khác nếu cần:
    # config_path = "./config/nturgbd120-cross-subject/default.yaml"
    # weights_path = "./test_weights/test/NTU120_csub_joint/runs-94-46248.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device ='cpu'  # Force sử dụng CPU cho ví dụ này
    print(f"Sử dụng device: {device}")
    
    
    try:
        # Load model
        model, config = load_model_from_config(config_path, weights_path)
        print("Model loaded thành công!")
        
        # Đếm số parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Số parameters: {num_params:,}")
        
        # Tạo dữ liệu mẫu
        model_args = config['model_args']
        data = load_sample_data(
            num_person=model_args['num_person'],
            num_point=model_args['num_point'],
            in_channels=3,  # x, y, z coordinates
            frames=64,
            batch_size=1
        )
        print(f"Input shape: {data.shape}")
        print(f"Input range: [{data.min().item():.3f}, {data.max().item():.3f}]")
        start_time = time.time()
        for i in range(10):
            # Chạy inference
            print(f"\nĐang chạy inference trên {device}...")
            output, inference_time = inference(model, data, device)
            
            # Debug output
            print(f"\nDebug output:")
            print(f"  - Output shape: {output.shape}")
            print(f"  - Output dtype: {output.dtype}")
            print(f"  - Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            print(f"  - Output first 5 values: {output[0, :5].detach().cpu().numpy()}")
            
            # Hiển thị kết quả
            print(f"\nKết quả inference:")
            print(f"  - Thời gian: {inference_time:.6f} giây")
            
            # Kiểm tra output có hợp lệ không
            if output.shape[1] != model_args['num_class']:
                print(f"  ⚠️ WARNING: Output shape không khớp với num_class!")
                print(f"     Expected: {model_args['num_class']}, Got: {output.shape[1]}")
            
            # Predicted class
            predicted_class = output.argmax(dim=1).item()
            print(f"  - Predicted class: {predicted_class}")
            
            # Hiển thị top 5 predictions với softmax probabilities
            probs = torch.softmax(output, dim=1)
            top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
            
            print(f"  - Top 5 predictions:")
            for i in range(5):
                cls_idx = top5_idx[0][i].item()
                cls_prob = top5_prob[0][i].item()
                print(f"      Class {cls_idx}: {cls_prob:.4f} ({cls_prob*100:.2f}%)")
            
            # Hiển thị raw scores (logits) của top 5
            top5_logits, top5_logits_idx = torch.topk(output, 5, dim=1)
            print(f"  - Top 5 raw scores (logits):")
            for i in range(5):
                cls_idx = top5_logits_idx[0][i].item()
                cls_logit = top5_logits[0][i].item()
                print(f"      Class {cls_idx}: {cls_logit:.4f}")
        total_time = time.time() - start_time
        print(f"\nTổng thời gian cho 10 lần inference: {total_time:.6f} giây")
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()
