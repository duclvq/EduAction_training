"""
Setup script for DD-Net training environment
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Setting up DD-Net training environment...")
    print("=" * 50)
    
    try:
        # Check if pip is available
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        print("✅ pip is available")
    except subprocess.CalledProcessError:
        print("❌ pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(requirements_file):
        print("❌ requirements.txt not found")
        return False
    
    try:
        print("📦 Installing requirements from requirements.txt...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def verify_installation():
    """Verify that all required packages are installed"""
    print("\n🧪 Verifying installation...")
    print("=" * 50)
    
    packages_to_check = [
        ('numpy', 'np'),
        ('tensorflow', 'tf'),
        ('keras', 'keras'),
        ('sklearn', 'sklearn'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns'),
        ('tqdm', 'tqdm'),
        ('scipy', 'scipy'),
        ('pandas', 'pd')
    ]
    
    all_good = True
    for package, alias in packages_to_check:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            
            # Try to get version
            try:
                if package == 'sklearn':
                    version = eval(f"{alias}.__version__")
                else:
                    version = eval(f"{alias}.__version__")
                print(f"✅ {package:<15} - version {version}")
            except:
                print(f"✅ {package:<15} - imported successfully")
                
        except ImportError:
            print(f"❌ {package:<15} - not available")
            all_good = False
    
    print("=" * 50)
    if all_good:
        print("🎉 All dependencies are installed and working!")
        return True
    else:
        print("❌ Some dependencies are missing or not working properly.")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️  Checking GPU availability...")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            
            # Test GPU configuration
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                    c = tf.matmul(a, b)
                print("✅ GPU computation test passed")
            except:
                print("⚠️  GPU found but computation test failed")
        else:
            print("ℹ️  No GPU found, will use CPU for training")
            print("   (Training will be slower but still functional)")
    except:
        print("❌ Cannot check GPU status")

def create_sample_config():
    """Create a sample configuration file"""
    config_content = '''"""
Sample configuration file for DD-Net training
Modify these parameters according to your needs
"""

class CustomConfig:
    def __init__(self):
        # Model parameters
        self.frame_l = 30       # Length of input frames
        self.joint_n = 48       # Number of joints
        self.joint_d = 3        # Joint dimensions (x, y, z)
        self.clc_num = 5        # Number of classes
        self.feat_d = 1128      # Feature dimensions
        self.filters = 16       # Base number of filters
        
        # Training parameters
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 0.001
        self.validation_split = 0.2
        self.test_split = 0.2
        
        # Data paths
        self.data_dir = r'D:\\py_source\\ddnet_classroom\\CStudentAct_processed_pose'
        
        # Output paths
        self.model_save_path = 'DD_Net_trained.h5'
        self.history_save_path = 'training_history.pkl'
        
        # Class names
        self.class_names = ['raising_hand', 'sleeping', 'standing', 'using_phone', 'writing']
        
        # Data augmentation
        self.use_data_augmentation = True
        self.augmentation_probability = 0.5
'''
    
    config_file = os.path.join(os.path.dirname(__file__), "config_sample.py")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"\n📄 Sample configuration created: {config_file}")
    print("   You can modify this file to customize training parameters")

def main():
    print("DD-Net Training Environment Setup")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Verify installation
    if not verify_installation():
        print("❌ Setup failed during verification")
        return
    
    # Check GPU
    check_gpu()
    
    # Create sample config
    create_sample_config()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Verify your data is in the correct directory")
    print("2. Run 'python test_setup.py' to test data loading")
    print("3. Run 'python training.py' to start training")
    print("4. Monitor training progress and adjust parameters as needed")

if __name__ == "__main__":
    main()
