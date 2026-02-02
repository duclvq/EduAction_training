"""
Simple training script that works around import issues
"""
import os
import sys
import numpy as np
import pickle
import random
from tqdm import tqdm

# Set environment before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging
# Remove problematic CUDA setting for now
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model
    import math
    print("✅ TensorFlow/Keras imported successfully")
except ImportError as e:
    print(f"❌ Failed to import TensorFlow/Keras: {e}")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    print("✅ scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ Failed to import scikit-learn: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✅ Matplotlib/Seaborn imported successfully")
except ImportError as e:
    print(f"⚠️  Plotting libraries not available: {e}")
    plt = None
    sns = None

# Set random seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def cosine_scheduler(epoch, lr, total_epochs, min_lr=1e-7, max_lr=1e-3):
    """Cosine annealing learning rate scheduler"""
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))

class SimpleConfig:
    """Simple configuration class"""
    def __init__(self):
        # Model parameters
        self.frame_l = 30
        self.joint_n = 48  # Will be adjusted based on data
        self.joint_d = 3   # Will be adjusted based on data  
        self.feat_d = 1128  # Will be calculated
        self.filters = 16
        
        # Training parameters
        self.batch_size = 8  # Smaller batch size for stability
        self.epochs = 50     # Fewer epochs for testing
        self.learning_rate = 0.001
        self.validation_split = 0.2
        self.test_split = 0.2
        
        # Learning rate scheduler parameters
        self.use_cosine_scheduler = True  # Set to False to use ReduceLROnPlateau
        self.min_lr = 1e-7
        self.max_lr = self.learning_rate
        
        # Paths
        self.data_dir = r'D:\py_source\ddnet_classroom\CStudentAct_processed_pose'
        self.model_save_path = 'simple_ddnet_model.h5'
        
        # Classes
        self.class_names = ['raising_hand', 'sleeping', 'standing', 'using_phone', 'writing']
        self.clc_num = len(self.class_names)

def load_and_prepare_data(config):
    """Load and prepare skeleton data"""
    print("Loading skeleton data...")
    
    all_poses = []
    all_labels = []
    
    for class_idx, class_name in enumerate(config.class_names):
        file_path = os.path.join(config.data_dir, f"{class_name}_pose.pkl")
        
        if not os.path.exists(file_path):
            print(f"⚠️  {file_path} not found, skipping...")
            continue
        
        print(f"📁 Loading {class_name}...")
        with open(file_path, 'rb') as f:
            class_data = pickle.load(f)
        
        for person_id, pose_sequence in class_data.items():
            all_poses.append(pose_sequence)
            all_labels.append(class_idx)
        
        print(f"   - Loaded {len(class_data)} sequences")
    
    print(f"Total: {len(all_poses)} samples across {len(config.class_names)} classes")
    return all_poses, all_labels

def simple_preprocess(poses, config):
    """Simple preprocessing that handles variable input dimensions"""
    print("Preprocessing poses...")
    
    # First, examine the data to determine actual dimensions
    sample_pose = poses[0]
    actual_joints = sample_pose.shape[1]
    actual_dims = sample_pose.shape[2]
    
    print(f"Data dimensions: joints={actual_joints}, coords={actual_dims}")
    
    # Update config to match data
    config.joint_n = min(actual_joints, 48)  # Use up to 48 joints
    config.joint_d = actual_dims
    
    # Calculate feature dimensions for distance matrix
    config.feat_d = int(config.joint_n * (config.joint_n - 1) / 2)
    
    print(f"Using: joints={config.joint_n}, dims={config.joint_d}, feat_d={config.feat_d}")
    
    processed_poses = []
    motion_features = []
    
    for pose in tqdm(poses, desc="Processing"):
        # Resize to target frame length using simple interpolation
        original_frames = pose.shape[0]
        
        if original_frames != config.frame_l:
            # Simple linear interpolation for frame resizing
            indices = np.linspace(0, original_frames - 1, config.frame_l)
            pose_resized = np.zeros((config.frame_l, config.joint_n, config.joint_d))
            
            for i, idx in enumerate(indices):
                idx_floor = int(np.floor(idx))
                idx_ceil = min(int(np.ceil(idx)), original_frames - 1)
                
                if idx_floor == idx_ceil:
                    pose_resized[i] = pose[idx_floor, :config.joint_n, :config.joint_d]
                else:
                    weight = idx - idx_floor
                    pose_resized[i] = ((1 - weight) * pose[idx_floor, :config.joint_n, :config.joint_d] + 
                                     weight * pose[idx_ceil, :config.joint_n, :config.joint_d])
        else:
            pose_resized = pose[:, :config.joint_n, :config.joint_d]
        
        # Normalize pose (simple min-max normalization)
        pose_resized = (pose_resized - pose_resized.min()) / (pose_resized.max() - pose_resized.min() + 1e-8)
        
        # Generate simple motion features (pairwise distances)
        motion_feat = []
        for frame in pose_resized:
            distances = []
            for i in range(config.joint_n):
                for j in range(i + 1, config.joint_n):
                    dist = np.linalg.norm(frame[i] - frame[j])
                    distances.append(dist)
            motion_feat.append(distances)
        
        motion_feat = np.array(motion_feat)
        
        processed_poses.append(pose_resized)
        motion_features.append(motion_feat)
    
    return np.array(processed_poses), np.array(motion_features)

def build_simple_model(config):
    """Build a simplified version of DD-Net"""
    print("Building simplified DD-Net model...")
    
    # Motion input
    motion_input = Input(shape=(config.frame_l, config.feat_d), name='motion')
    # Pose input  
    pose_input = Input(shape=(config.frame_l, config.joint_n, config.joint_d), name='pose')
    
    # Motion stream
    x_motion = Conv1D(config.filters * 2, 3, padding='same', activation='relu')(motion_input)
    x_motion = MaxPooling1D(2)(x_motion)
    x_motion = Conv1D(config.filters * 4, 3, padding='same', activation='relu')(x_motion)
    x_motion = GlobalMaxPooling1D()(x_motion)
    
    # Pose stream - flatten and process
    x_pose = Reshape((config.frame_l, -1))(pose_input)
    x_pose = Conv1D(config.filters * 2, 3, padding='same', activation='relu')(x_pose)
    x_pose = MaxPooling1D(2)(x_pose)
    x_pose = Conv1D(config.filters * 4, 3, padding='same', activation='relu')(x_pose)
    x_pose = GlobalMaxPooling1D()(x_pose)
    
    # Fusion
    x = concatenate([x_motion, x_pose])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(config.clc_num, activation='softmax')(x)
    
    model = Model(inputs=[motion_input, pose_input], outputs=x)
    return model

def train_simple_model():
    """Main training function"""
    print("🚀 Starting Simple DD-Net Training")
    print("=" * 50)
    
    # Initialize config
    config = SimpleConfig()
    
    # Load data
    poses, labels = load_and_prepare_data(config)
    
    if len(poses) == 0:
        print("❌ No data found!")
        return
    
    # Preprocess data
    X_pose, X_motion = simple_preprocess(poses, config)
    Y = np.array(labels)
    
    print(f"Data shapes: X_pose={X_pose.shape}, X_motion={X_motion.shape}, Y={Y.shape}")
    
    # Create train/test splits
    X_motion_train, X_motion_test, X_pose_train, X_pose_test, Y_train, Y_test = train_test_split(
        X_motion, X_pose, Y, test_size=0.3, stratify=Y, random_state=42
    )
    
    # Convert labels to categorical
    Y_train_cat = to_categorical(Y_train, num_classes=config.clc_num)
    Y_test_cat = to_categorical(Y_test, num_classes=config.clc_num)
    
    print(f"Train samples: {len(X_motion_train)}, Test samples: {len(X_motion_test)}")
    
    # Build model
    model = build_simple_model(config)
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model summary:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    # Add learning rate scheduler
    if config.use_cosine_scheduler:
        print("🔄 Using cosine annealing learning rate scheduler")
        lr_scheduler = LearningRateScheduler(
            lambda epoch, lr: cosine_scheduler(epoch, lr, config.epochs, config.min_lr, config.max_lr),
            verbose=1
        )
        callbacks.append(lr_scheduler)
    else:
        print("📉 Using ReduceLROnPlateau scheduler")
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=config.min_lr))
    
    # Train model
    print("🎯 Starting training...")
    history = model.fit(
        [X_motion_train, X_pose_train], Y_train_cat,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("📊 Evaluating model...")
    test_loss, test_acc = model.evaluate([X_motion_test, X_pose_test], Y_test_cat, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Predictions and classification report
    predictions = model.predict([X_motion_test, X_pose_test])
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("\n📋 Classification Report:")
    print(classification_report(Y_test, predicted_classes, target_names=config.class_names))
    
    # Save model
    model.save(config.model_save_path)
    print(f"✅ Model saved to {config.model_save_path}")
    
    # Plot training history if matplotlib is available
    if plt is not None:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(history.history['accuracy'], label='Training')
            ax1.plot(history.history['val_accuracy'], label='Validation')
            ax1.set_title('Accuracy')
            ax1.legend()
            
            ax2.plot(history.history['loss'], label='Training')
            ax2.plot(history.history['val_loss'], label='Validation')
            ax2.set_title('Loss')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('simple_training_history.png')
            print("📈 Training plots saved to simple_training_history.png")
        except Exception as e:
            print(f"⚠️  Could not save plots: {e}")
    
    return model, history

if __name__ == "__main__":
    # Check GPU
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("🎮 GPU detected")
    else:
        print("💻 Using CPU")
    
    # Run training
    try:
        model, history = train_simple_model()
        print("🎉 Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
