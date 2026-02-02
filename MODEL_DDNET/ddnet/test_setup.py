"""
Simple script to test data loading and verify training setup
"""
import os
import sys
import pickle
import numpy as np

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """Test loading and examining the skeleton data"""
    data_dir = r'D:\py_source\ddnet_classroom\CStudentAct_processed_pose'
    class_names = ['raising_hand', 'sleeping', 'standing', 'using_phone', 'writing']
    
    print("Testing data loading...")
    print("=" * 50)
    
    total_samples = 0
    for class_idx, class_name in enumerate(class_names):
        file_path = os.path.join(data_dir, f"{class_name}_pose.pkl")
        
        if not os.path.exists(file_path):
            print(f"❌ {file_path} not found")
            continue
            
        print(f"📁 Loading {class_name}...")
        with open(file_path, 'rb') as f:
            class_data = pickle.load(f)
        
        print(f"   - Number of sequences: {len(class_data)}")
        
        # Examine a sample
        sample_key = list(class_data.keys())[0]
        sample_data = class_data[sample_key]
        print(f"   - Sample shape: {sample_data.shape}")
        print(f"   - Data type: {sample_data.dtype}")
        print(f"   - Value range: [{sample_data.min():.2f}, {sample_data.max():.2f}]")
        
        total_samples += len(class_data)
        print()
    
    print(f"Total samples across all classes: {total_samples}")
    print("=" * 50)

def test_model_import():
    """Test importing the model and utilities"""
    print("Testing model imports...")
    print("=" * 50)
    
    try:
        from ddnet import build_DD_Net, Config
        print("✅ Successfully imported build_DD_Net and Config")
        
        from utils import zoom, get_CG, sampling_frame
        print("✅ Successfully imported utility functions")
        
        # Test config
        config = Config()
        print(f"✅ Config created successfully")
        print(f"   - Frame length: {config.frame_l}")
        print(f"   - Joint number: {config.joint_n}")
        print(f"   - Joint dimensions: {config.joint_d}")
        print(f"   - Feature dimensions: {config.feat_d}")
        
        # Test model building
        model = build_DD_Net(config)
        print(f"✅ Model built successfully")
        print(f"   - Model input shapes: {[input.shape for input in model.inputs]}")
        print(f"   - Model output shape: {model.output.shape}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("=" * 50)

def test_data_preprocessing():
    """Test a small sample of data preprocessing"""
    print("Testing data preprocessing...")
    print("=" * 50)
    
    try:
        from ddnet import Config
        from utils import zoom, get_CG
        
        # Load a small sample
        data_dir = r'D:\py_source\ddnet_classroom\CStudentAct_processed_pose'
        file_path = os.path.join(data_dir, "raising_hand_pose.pkl")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Get first sample
        sample_key = list(data.keys())[0]
        pose = data[sample_key]
        
        print(f"Original pose shape: {pose.shape}")
        
        # Initialize config
        config = Config()
        
        # Test resizing to match expected dimensions
        if pose.shape[1] != config.joint_n:
            if pose.shape[1] > config.joint_n:
                pose = pose[:, :config.joint_n, :]
            else:
                padding = np.zeros((pose.shape[0], config.joint_n - pose.shape[1], pose.shape[2]))
                pose = np.concatenate([pose, padding], axis=1)
        
        # Handle coordinate dimensions
        if pose.shape[2] < config.joint_d:
            padding = np.zeros((pose.shape[0], pose.shape[1], config.joint_d - pose.shape[2]))
            pose = np.concatenate([pose, padding], axis=2)
        elif pose.shape[2] > config.joint_d:
            pose = pose[:, :, :config.joint_d]
        
        print(f"Adjusted pose shape: {pose.shape}")
        
        # Test zoom function
        pose_resized = zoom(pose, target_l=config.frame_l, 
                          joints_num=config.joint_n, joints_dim=config.joint_d)
        print(f"Resized pose shape: {pose_resized.shape}")
        
        # Test motion feature generation
        motion_features = get_CG(pose_resized, config)
        print(f"Motion features shape: {motion_features.shape}")
        
        print("✅ Data preprocessing test successful!")
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 50)

if __name__ == "__main__":
    print("DD-Net Training Setup Test")
    print("=" * 60)
    
    # Run tests
    test_data_loading()
    test_model_import()
    test_data_preprocessing()
    
    print("Test completed!")
