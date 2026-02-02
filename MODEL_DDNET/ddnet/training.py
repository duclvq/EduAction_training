import os
import sys
import copy
import random
from collections import Counter

# Check for required dependencies
def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import tensorflow as tf
    except ImportError:
        missing_deps.append('tensorflow')
    
    try:
        import keras
    except ImportError:
        missing_deps.append('keras')
    
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        missing_deps.append('scikit-learn')
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing_deps.append('matplotlib')
    
    try:
        import seaborn as sns
    except ImportError:
        missing_deps.append('seaborn')
    
    try:
        from tqdm import tqdm
    except ImportError:
        missing_deps.append('tqdm')
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n📦 To install missing dependencies, run:")
        print("   pip install -r requirements.txt")
        print("\n   Or install individually:")
        for dep in missing_deps:
            print(f"   pip install {dep}")
        return False
    
    return True

# Only import if dependencies are available
if check_dependencies():
    import numpy as np
    import pickle
    import random
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm

    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from keras.utils import to_categorical
    from keras.callbacks import LearningRateScheduler
    import math
    import keras.backend as K
    import tensorflow as tf

    from ddnet import build_DD_Net, Config
    from utils import zoom, get_CG, sampling_frame
else:
    print("❌ Cannot proceed without required dependencies.")
    print("Please install them and try again.")
    sys.exit(1)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def cosine_scheduler(epoch, lr, total_epochs, min_lr=1e-7, max_lr=1e-3):
    """
    Cosine annealing learning rate scheduler
    """
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))

def cosine_scheduler_with_restarts(epoch, lr, total_epochs, min_lr=1e-7, max_lr=1e-3, restart_period=50):
    """
    Cosine annealing with warm restarts
    """
    t_cur = epoch % restart_period
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t_cur / restart_period))

class TrainingConfig(Config):
    def __init__(self):
        super().__init__()
        # Training specific parameters
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 0.001
        self.validation_split = 0.1
        self.test_split = 0.2
        self.model_save_path = 'DD_Net_trained.h5'
        self.history_save_path = 'training_history.pkl'
        
        # Data augmentation parameters
        self.use_data_augmentation = True
        self.augmentation_probability = 0.5
        
        # Learning rate scheduler parameters
        self.use_cosine_scheduler = True  # Set to False to use ReduceLROnPlateau instead
        self.min_lr = 1e-6
        self.max_lr = self.learning_rate
        self.cosine_restarts = False  # Set to True for cosine annealing with warm restarts
        
        # Class mapping
        self.class_names = ['raising_hand', 'sleeping', 'standing', 'using_phone', 'writing', 'sitting']
        self.clc_num = len(self.class_names)
        
        # Data source configuration
        self.use_tracking_data = True  # True: use tracking results, False: use processed pose
        self.train_data_file = r'D:\py_source\ddnet_classroom\tracking\ByteTrack\train_data_continuous.pkl'
        self.val_data_file = r'D:\py_source\ddnet_classroom\tracking\ByteTrack\val_data_continuous.pkl'
        
        # Oversampling parameters
        self.apply_oversampling = True
        self.oversampling_strategy = 'balanced'  # 'balanced' or 'target_count'
        self.target_samples_per_class = None  # None for balanced, or specific number
        self.augmentation_noise_factor = 0.01  # 1% noise for augmented samples

def load_skeleton_data(data_dir, config):
    """
    Load skeleton pose data from separate train and validation files
    Returns train and validation data separately
    """
    train_poses = []
    train_labels = []
    val_poses = []
    val_labels = []
    
    if config.use_tracking_data:
        print("Loading skeleton data from separate train/val files...")
        
        # Check if files exist
        if not os.path.exists(config.train_data_file):
            print(f"Error: {config.train_data_file} not found!")
            print("Falling back to processed pose data...")
            config.use_tracking_data = False
        elif not os.path.exists(config.val_data_file):
            print(f"Error: {config.val_data_file} not found!")
            print("Falling back to processed pose data...")
            config.use_tracking_data = False
        else:
            print(f"Loading training data from: {config.train_data_file}")
            with open(config.train_data_file, 'rb') as f:
                train_data = pickle.load(f)
            
            print(f"Loading validation data from: {config.val_data_file}")
            with open(config.val_data_file, 'rb') as f:
                val_data = pickle.load(f)
            
            # Process train data
            if 'X' in train_data and 'y' in train_data:
                print("\nProcessing training data (X, y structure)")
                
                # Show action map if available
                if 'action_map' in train_data:
                    print(f"Action map: {train_data['action_map']}")
                
                # Load train data directly (no sampling limit for separate files)
                train_poses = list(train_data['X'])
                train_labels = list(train_data['y'])
                
                print(f"\nTraining set loaded: {len(train_poses)} samples")
                
                # Show train class distribution
                train_counts = Counter(train_labels)
                for class_idx, count in sorted(train_counts.items()):
                    class_name = config.class_names[class_idx]
                    print(f"  {class_name}: {count} samples")
                
                # Process validation data
                if 'X' in val_data and 'y' in val_data:
                    print("\nProcessing validation data (X, y structure)")
                    val_poses = list(val_data['X'])
                    val_labels = list(val_data['y'])
                    
                    print(f"\nValidation set loaded: {len(val_poses)} samples")
                    
                    # Show val class distribution
                    val_counts = Counter(val_labels)
                    for class_idx, count in sorted(val_counts.items()):
                        class_name = config.class_names[class_idx]
                        print(f"  {class_name}: {count} samples")
                else:
                    print("Warning: Validation data format not recognized")
                    val_poses = []
                    val_labels = []
            
            else:
                # Old format: class-based structure
                # Extract by class
                for class_idx, class_name in enumerate(config.class_names):
                    print(f"Processing {class_name}...")
                    count = 0
                    
                    # Check different possible key formats
                    possible_keys = [
                        f'train_{class_name}',
                        f'{class_name}',
                        class_name
                    ]
                    
                    class_data = None
                    for key in possible_keys:
                        if key in tracking_data:
                            class_data = tracking_data[key]
                            break
                    
                    if class_data is None:
                        print(f"  Warning: No data found for {class_name}")
                        continue
                    
                    # Iterate through video IDs
                    for video_id, segments in class_data.items():
                        # Each video can have multiple segments (continuous sequences)
                        for segment in segments:
                            # Extract keypoints from segment
                            keypoints_sequence = []
                            for frame in segment:
                                if 'keypoints' in frame:
                                    keypoints_sequence.append(frame['keypoints'])
                            
                            if len(keypoints_sequence) > 0:
                                pose_array = np.array(keypoints_sequence)
                                all_poses.append(pose_array)
                                all_labels.append(class_idx)
                                count += 1
                    
                    print(f"  Loaded {count} samples")
    
    if not config.use_tracking_data:
        print("Loading skeleton data from processed pose files...")
        
        for class_idx, class_name in enumerate(config.class_names):
            file_path = os.path.join(data_dir, f"{class_name}_pose.pkl")
            
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping...")
                continue
                
            print(f"Loading {class_name} data...")
            with open(file_path, 'rb') as f:
                class_data = pickle.load(f)
            
            for person_id, pose_sequence in class_data.items():
                # pose_sequence shape: (frames, joints, coords)
                all_poses.append(pose_sequence)
                all_labels.append(class_idx)
            
            print(f"  Loaded {sum(1 for l in all_labels if l == class_idx)} samples")
    
    print(f"\nTotal samples loaded:")
    print(f"  Training: {len(train_poses)}")
    print(f"  Validation: {len(val_poses)}")
    
    return train_poses, train_labels, val_poses, val_labels

def apply_oversampling_to_training_data(X_motion_train, X_pose_train, Y_train_raw, config):
    """
    Apply oversampling to training data only (not validation or test)
    """
    if not config.apply_oversampling:
        print("Oversampling disabled, using original data distribution")
        return X_motion_train, X_pose_train, Y_train_raw
    
    print("\n" + "="*50)
    print("APPLYING OVERSAMPLING TO TRAINING DATA")
    print("="*50)
    
    # Check current class distribution
    class_counts = Counter(Y_train_raw)
    print("Current training class distribution:")
    for class_idx, count in sorted(class_counts.items()):
        class_name = config.class_names[class_idx]
        print(f"  {class_name}: {count} samples")
    
    # Determine target sample count for each class
    if config.oversampling_strategy == 'balanced':
        target_count = max(class_counts.values())
        print(f"\nBalancing all classes to {target_count} samples")
    elif config.target_samples_per_class:
        target_count = config.target_samples_per_class
        print(f"\nTargeting {target_count} samples per class")
    else:
        target_count = max(class_counts.values())
        print(f"\nBalancing all classes to {target_count} samples (default)")
    
    # Separate data by class
    class_data = {}
    for class_idx in range(config.clc_num):
        mask = Y_train_raw == class_idx
        class_data[class_idx] = {
            'X_motion': X_motion_train[mask],
            'X_pose': X_pose_train[mask],
            'indices': np.where(mask)[0]
        }
    
    # Apply oversampling to each class
    oversampled_X_motion = []
    oversampled_X_pose = []
    oversampled_Y = []
    
    for class_idx in range(config.clc_num):
        current_count = len(class_data[class_idx]['X_motion'])
        class_name = config.class_names[class_idx]
        
        print(f"\nProcessing {class_name}:")
        print(f"  Current: {current_count} samples")
        
        if current_count >= target_count:
            print(f"  No oversampling needed")
            # Use existing samples
            oversampled_X_motion.extend(class_data[class_idx]['X_motion'])
            oversampled_X_pose.extend(class_data[class_idx]['X_pose'])
            oversampled_Y.extend([class_idx] * current_count)
        else:
            samples_needed = target_count - current_count
            print(f"  Need to generate {samples_needed} additional samples")
            
            # Add original samples
            oversampled_X_motion.extend(class_data[class_idx]['X_motion'])
            oversampled_X_pose.extend(class_data[class_idx]['X_pose'])
            oversampled_Y.extend([class_idx] * current_count)
            
            # Generate additional samples through augmentation
            for i in range(samples_needed):
                # Randomly select a source sample
                source_idx = random.randint(0, current_count - 1)
                source_motion = class_data[class_idx]['X_motion'][source_idx].copy()
                source_pose = class_data[class_idx]['X_pose'][source_idx].copy()
                
                # Add small random noise for variation
                motion_noise = np.random.normal(0, config.augmentation_noise_factor, source_motion.shape)
                pose_noise = np.random.normal(0, config.augmentation_noise_factor, source_pose.shape)
                
                augmented_motion = source_motion + motion_noise
                augmented_pose = source_pose + pose_noise
                
                oversampled_X_motion.append(augmented_motion)
                oversampled_X_pose.append(augmented_pose)
                oversampled_Y.append(class_idx)
            
            print(f"  Final: {target_count} samples (added {samples_needed})")
    
    # Convert back to numpy arrays
    oversampled_X_motion = np.array(oversampled_X_motion)
    oversampled_X_pose = np.array(oversampled_X_pose)
    oversampled_Y = np.array(oversampled_Y)
    
    # Shuffle the oversampled data
    shuffle_indices = np.random.permutation(len(oversampled_Y))
    oversampled_X_motion = oversampled_X_motion[shuffle_indices]
    oversampled_X_pose = oversampled_X_pose[shuffle_indices]
    oversampled_Y = oversampled_Y[shuffle_indices]
    
    print("\n" + "="*50)
    print("OVERSAMPLING RESULTS")
    print("="*50)
    print(f"Original training samples: {len(Y_train_raw)}")
    print(f"Oversampled training samples: {len(oversampled_Y)}")
    
    final_counts = Counter(oversampled_Y)
    print("\nFinal training class distribution:")
    for class_idx, count in sorted(final_counts.items()):
        class_name = config.class_names[class_idx]
        percentage = count / len(oversampled_Y) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    return oversampled_X_motion, oversampled_X_pose, oversampled_Y

def preprocess_pose_data(poses, labels, config):
    """
    Preprocess pose data for training
    """
    print("Preprocessing pose data...")
    X_motion = []  # Motion features (M)
    X_pose = []    # Raw pose data (P)
    Y = []         # Labels
    
    for i, pose in enumerate(tqdm(poses, desc="Processing poses")):
        try:
            # Normalize pose dimensions to match config
            # Current data seems to have shape (frames, joints, 2)
            # We need to adapt it to (frames, joint_n, joint_d)
            
            # First, resize the pose to match expected dimensions
            if pose.shape[1] != config.joint_n:
                # If we have more joints than expected, take the first joint_n joints
                if pose.shape[1] > config.joint_n:
                    pose = pose[:, :config.joint_n, :]
                # If we have fewer joints, pad with zeros
                else:
                    padding = np.zeros((pose.shape[0], config.joint_n - pose.shape[1], pose.shape[2]))
                    pose = np.concatenate([pose, padding], axis=1)
            
            # Handle coordinate dimensions
            if pose.shape[2] < config.joint_d:
                # If we have fewer dimensions (e.g., 2D instead of 3D), pad with zeros
                padding = np.zeros((pose.shape[0], pose.shape[1], config.joint_d - pose.shape[2]))
                pose = np.concatenate([pose, padding], axis=2)
            elif pose.shape[2] > config.joint_d:
                # If we have more dimensions, take the first joint_d dimensions
                pose = pose[:, :, :config.joint_d]
            
            # Resize to target frame length
            pose_resized = zoom(pose, target_l=config.frame_l, 
                              joints_num=config.joint_n, joints_dim=config.joint_d)
            
            # Apply data augmentation if enabled
            if config.use_data_augmentation and random.random() < config.augmentation_probability:
                pose_resized = sampling_frame(pose_resized, config)
            
            # Generate motion features
            motion_features = get_CG(pose_resized, config)
            
            X_motion.append(motion_features)
            X_pose.append(pose_resized)
            Y.append(labels[i])
            
        except Exception as e:
            print(f"Error processing pose {i}: {e}")
            continue
    
    # Convert to numpy arrays
    X_motion = np.array(X_motion)
    X_pose = np.array(X_pose)
    Y = np.array(Y)
    
    print(f"Processed data shapes:")
    print(f"X_motion: {X_motion.shape}")
    print(f"X_pose: {X_pose.shape}")
    print(f"Y: {Y.shape}")
    
    return X_motion, X_pose, Y

def create_data_splits(X_motion, X_pose, Y, config):
    """
    Create train/validation/test splits
    """
    print("Creating data splits...")
    
    # First split: separate test set
    X_motion_temp, X_motion_test, X_pose_temp, X_pose_test, Y_temp, Y_test = train_test_split(
        X_motion, X_pose, Y, test_size=config.test_split, stratify=Y, random_state=42
    )
    
    # Second split: separate train and validation from remaining data
    val_size = config.validation_split / (1 - config.test_split)
    X_motion_train, X_motion_val, X_pose_train, X_pose_val, Y_train, Y_val = train_test_split(
        X_motion_temp, X_pose_temp, Y_temp, test_size=val_size, stratify=Y_temp, random_state=42
    )
    
    # Convert labels to categorical
    Y_train_cat = to_categorical(Y_train, num_classes=config.clc_num)
    Y_val_cat = to_categorical(Y_val, num_classes=config.clc_num)
    Y_test_cat = to_categorical(Y_test, num_classes=config.clc_num)
    
    print(f"Train set: {len(X_motion_train)} samples")
    print(f"Validation set: {len(X_motion_val)} samples") 
    print(f"Test set: {len(X_motion_test)} samples")
    
    return (X_motion_train, X_pose_train, Y_train_cat), \
           (X_motion_val, X_pose_val, Y_val_cat), \
           (X_motion_test, X_pose_test, Y_test_cat), \
           (Y_train, Y_val, Y_test)

def plot_training_history(history, save_path='training_plots.png'):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_motion_test, X_pose_test, Y_test_cat, Y_test_raw, config):
    """
    Evaluate the trained model
    """
    print("Evaluating model...")
    
    # Predict on test set
    predictions = model.predict([X_motion_test, X_pose_test])
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    test_accuracy = np.mean(predicted_classes == Y_test_raw)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(Y_test_raw, predicted_classes, 
                              target_names=config.class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(Y_test_raw, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.class_names, yticklabels=config.class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_accuracy, predicted_classes

def train_model():
    """
    Main training function
    """
    # Initialize configuration
    config = TrainingConfig()
    
    print("DD-Net Training Script")
    print("=" * 50)
    print(f"Configuration:")
    print(f"- Frame length: {config.frame_l}")
    print(f"- Joint number: {config.joint_n}")
    print(f"- Joint dimensions: {config.joint_d}")
    print(f"- Number of classes: {config.clc_num}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Learning rate: {config.learning_rate}")
    print(f"- Epochs: {config.epochs}")
    print("=" * 50)
    
    # Load data (now returns separate train and val sets)
    train_poses, train_labels, val_poses, val_labels = load_skeleton_data(config.data_dir, config)
    
    if len(train_poses) == 0:
        print("No training data found! Please check your data files.")
        return
    
    if len(val_poses) == 0:
        print("No validation data found! Please check your data files.")
        return
    
    # Preprocess training data
    print("\nPreprocessing training data...")
    X_motion_train, X_pose_train, Y_train_raw = preprocess_pose_data(train_poses, train_labels, config)
    
    # Preprocess validation data
    print("\nPreprocessing validation data...")
    X_motion_val, X_pose_val, Y_val_raw = preprocess_pose_data(val_poses, val_labels, config)
    
    # Create test set from validation data (20% of validation)
    print("\nCreating test set from validation data...")
    X_motion_val, X_motion_test, X_pose_val, X_pose_test, Y_val_raw, Y_test_raw = train_test_split(
        X_motion_val, X_pose_val, Y_val_raw, test_size=0.2, stratify=Y_val_raw, random_state=42
    )
    
    # Convert labels to categorical
    Y_train_cat = to_categorical(Y_train_raw, num_classes=config.clc_num)
    Y_val_cat = to_categorical(Y_val_raw, num_classes=config.clc_num)
    Y_test_cat = to_categorical(Y_test_raw, num_classes=config.clc_num)
    
    # Apply oversampling to training data only
    X_motion_train, X_pose_train, Y_train_raw = apply_oversampling_to_training_data(
        X_motion_train, X_pose_train, Y_train_raw, config
    )
    
    # Convert oversampled training labels to categorical
    Y_train_cat = to_categorical(Y_train_raw, num_classes=config.clc_num)
    
    print(f"\nFinal data splits:")
    print(f"- Training: {len(X_motion_train)} samples")
    print(f"- Validation: {len(X_motion_val)} samples") 
    print(f"- Test: {len(X_motion_test)} samples")
    
    # Build model
    print("Building DD-Net model...")
    model = build_DD_Net(config)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model compiled successfully!")
    model.summary()
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            config.model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Add learning rate scheduler
    if config.use_cosine_scheduler:
        print("Using cosine annealing learning rate scheduler")
        if config.cosine_restarts:
            # Cosine annealing with warm restarts
            lr_scheduler = LearningRateScheduler(
                lambda epoch, lr: cosine_scheduler_with_restarts(
                    epoch, lr, config.epochs, config.min_lr, config.max_lr
                ),
                verbose=1
            )
        else:
            # Standard cosine annealing
            lr_scheduler = LearningRateScheduler(
                lambda epoch, lr: cosine_scheduler(
                    epoch, lr, config.epochs, config.min_lr, config.max_lr
                ),
                verbose=1
            )
        callbacks.append(lr_scheduler)
    else:
        print("Using ReduceLROnPlateau scheduler")
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=config.min_lr,
                verbose=1
            )
        )
    
    # Train model
    print("Starting training...")
    history = model.fit(
        [X_motion_train, X_pose_train], Y_train_cat,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=([X_motion_val, X_pose_val], Y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    with open(config.history_save_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model
    print("Loading best model for evaluation...")
    model.load_weights(config.model_save_path)
    
    # Evaluate model
    test_accuracy, predictions = evaluate_model(
        model, X_motion_test, X_pose_test, Y_test_cat, Y_test_raw, config
    )
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {config.model_save_path}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    
    return model, history, test_accuracy

if __name__ == "__main__":
    # Set GPU memory growth (if using GPU)
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {len(gpus)} device(s)")
        else:
            print("No GPU found, using CPU")
    except:
        print("GPU configuration failed, using default settings")
    
    # Run training
    model, history, test_accuracy = train_model()
