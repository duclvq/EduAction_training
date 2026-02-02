"""
Run training with multiple configurations and compare results.
This script trains models with different keypoint combinations and saves comparison results.
"""

import os
import pickle
import pandas as pd
from MODEL_DDNET.train_eduaction import train_eduaction_model
import tensorflow as tf


def run_config_comparison(configs, output_csv='config_comparison.csv'):
    """
    Run training with multiple configurations and compare results.

    Args:
        configs: list of tuples (name, use_legs, use_face, use_hands)
        output_csv: path to save comparison CSV
    """
    results = []

    for config_name, use_legs, use_face, use_hands in configs:
        print("\n" + "="*80)
        print(f"RUNNING CONFIGURATION: {config_name}")
        print("="*80)
        print(f"  Legs: {use_legs}")
        print(f"  Face: {use_face}")
        print(f"  Hands: {use_hands}")
        print("="*80 + "\n")

        try:
            # Run training
            model, history, test_accuracy = train_eduaction_model(
                use_leg_keypoints=use_legs,
                use_face_keypoints=use_face,
                use_hand_keypoints=use_hands
            )

            # Get best validation accuracy
            best_val_acc = max(history.history['val_accuracy'])
            final_train_acc = history.history['accuracy'][-1]

            # Calculate number of keypoints
            num_keypoints = 0
            if use_legs:
                num_keypoints += 23  # full body
            else:
                num_keypoints += 13  # upper body only

            if use_face:
                num_keypoints += 68

            if use_hands:
                num_keypoints += 42

            # feat_d calculation
            feat_d = num_keypoints * (num_keypoints - 1) // 2

            # Store results
            results.append({
                'Config': config_name,
                'Legs': use_legs,
                'Face': use_face,
                'Hands': use_hands,
                'Keypoints': num_keypoints,
                'feat_d': feat_d,
                'Best_Val_Acc': best_val_acc,
                'Final_Train_Acc': final_train_acc,
                'Test_Acc': test_accuracy,
                'Epochs': len(history.history['accuracy'])
            })

            # Clear memory
            del model
            tf.keras.backend.clear_session()

        except Exception as e:
            print(f"\n❌ Error running config {config_name}: {e}\n")
            results.append({
                'Config': config_name,
                'Legs': use_legs,
                'Face': use_face,
                'Hands': use_hands,
                'Error': str(e)
            })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to {output_csv}")

    # Print summary table
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    return df


def main():
    """Main function with predefined configurations"""

    # Define configurations to test
    # Format: (name, use_legs, use_face, use_hands)
    configs = [
        # Full combinations
        ("Full_Body", True, True, True),
        ("No_Legs", False, True, True),
        ("No_Face", True, False, True),
        ("No_Hands", True, True, False),

        # Minimal combinations
        ("Body_Hands", False, False, True),
        ("Body_Face", False, True, False),
        ("Body_Only", False, False, False),

        # Hands focused (for drinking, play_phone, writing)
        ("Hands_Only", False, False, True),
    ]

    print("="*80)
    print("MULTI-CONFIGURATION TRAINING")
    print("="*80)
    print(f"Total configurations to test: {len(configs)}\n")

    for i, (name, legs, face, hands) in enumerate(configs, 1):
        keypoint_count = (23 if legs else 13) + (68 if face else 0) + (42 if hands else 0)
        print(f"{i}. {name:20s} → {keypoint_count:3d} keypoints (L:{legs} F:{face} H:{hands})")

    print("="*80)
    input("\nPress Enter to start training (or Ctrl+C to cancel)...")

    # Run comparison
    results_df = run_config_comparison(configs)

    # Print best configuration
    if 'Test_Acc' in results_df.columns:
        best_idx = results_df['Test_Acc'].idxmax()
        best_config = results_df.iloc[best_idx]

        print("\n" + "="*80)
        print("🏆 BEST CONFIGURATION")
        print("="*80)
        print(f"Config: {best_config['Config']}")
        print(f"Keypoints: {best_config['Keypoints']}")
        print(f"Test Accuracy: {best_config['Test_Acc']:.4f}")
        print(f"Val Accuracy: {best_config['Best_Val_Acc']:.4f}")
        print("="*80)


if __name__ == "__main__":
    # GPU configuration
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {len(gpus)} device(s)\n")
        else:
            print("No GPU found, using CPU\n")
    except:
        print("GPU configuration failed, using default settings\n")

    main()
