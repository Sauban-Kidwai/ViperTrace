"""
ViperTrace Data Processing Script

Generates synthetic training data for malware detection.
In production, replace with real malware samples and clean files.
"""

import numpy as np
import torch
import math
from pathlib import Path


def calculate_entropy(data: bytes) -> float:
    """
    Calculate Shannon entropy of byte data
    
    Args:
        data: Byte sequence to analyze
    
    Returns:
        Entropy value (0-8 bits)
    """
    if not data:
        return 0.0
    
    # Count byte frequencies
    byte_counts = [0] * 256
    for byte in data:
        byte_counts[byte] += 1
    
    # Calculate entropy
    entropy = 0.0
    data_len = len(data)
    
    for count in byte_counts:
        if count > 0:
            probability = count / data_len
            entropy -= probability * math.log2(probability)
    
    return entropy


def generate_synthetic_chunk(is_malware: bool, chunk_size: int = 512) -> tuple[np.ndarray, float]:
    """
    Generate a synthetic file chunk with corresponding entropy
    
    Args:
        is_malware: Whether to generate malware-like or clean data
        chunk_size: Size of chunk in bytes
    
    Returns:
        Tuple of (normalized_bytes, entropy)
    """
    if is_malware:
        # Malware: High entropy (encrypted/packed) or very low entropy (repetitive)
        if np.random.random() > 0.5:
            # High entropy - encrypted/packed malware
            chunk = np.random.randint(0, 256, chunk_size, dtype=np.uint8)
        else:
            # Low entropy - repetitive patterns
            pattern = np.random.randint(0, 256, 10, dtype=np.uint8)
            chunk = np.tile(pattern, chunk_size // len(pattern) + 1)[:chunk_size]
    else:
        # Clean files: Medium entropy (structured but varied)
        chunk = np.random.randint(32, 200, chunk_size, dtype=np.uint8)
        # Add some structure
        chunk[::10] = np.random.randint(0, 32, chunk_size // 10, dtype=np.uint8)
    
    # Calculate entropy
    entropy = calculate_entropy(chunk.tobytes())
    
    # Normalize bytes to [0, 1]
    chunk_normalized = chunk.astype(np.float32) / 255.0
    
    return chunk_normalized, entropy


def generate_dataset(n_samples: int = 5000, chunk_size: int = 512, seed: int = 42):
    """
    Generate complete training dataset
    
    Args:
        n_samples: Total number of samples to generate
        chunk_size: Size of each chunk in bytes
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X_bytes, X_entropy, y_labels)
    """
    np.random.seed(seed)
    
    X_bytes = []
    X_entropy = []
    y_labels = []
    
    print(f"🔧 Generating {n_samples} synthetic samples...")
    
    for i in range(n_samples):
        # 50% malware, 50% clean
        is_malware = i >= n_samples // 2
        
        chunk, entropy = generate_synthetic_chunk(is_malware, chunk_size)
        
        X_bytes.append(chunk)
        X_entropy.append(entropy)
        y_labels.append(1.0 if is_malware else 0.0)
        
        if (i + 1) % 1000 == 0:
            print(f"  ✓ Generated {i + 1}/{n_samples} samples")
    
    # Convert to numpy arrays
    X_bytes = np.array(X_bytes, dtype=np.float32)
    X_entropy = np.array(X_entropy, dtype=np.float32).reshape(-1, 1)
    y_labels = np.array(y_labels, dtype=np.float32).reshape(-1, 1)
    
    return X_bytes, X_entropy, y_labels


def save_dataset(X_bytes, X_entropy, y_labels, filepath: str = "training_data.npz"):
    """Save dataset to disk"""
    np.savez_compressed(
        filepath,
        X_bytes=X_bytes,
        X_entropy=X_entropy,
        y_labels=y_labels
    )
    print(f"✅ Dataset saved to: {filepath}")


def load_dataset(filepath: str = "training_data.npz"):
    """Load dataset from disk"""
    data = np.load(filepath)
    return data['X_bytes'], data['X_entropy'], data['y_labels']


if __name__ == "__main__":
    print("=" * 60)
    print("      ViperTrace Data Processing")
    print("=" * 60)
    
    # Generate training data
    X_bytes_train, X_entropy_train, y_train = generate_dataset(
        n_samples=5000,
        chunk_size=512,
        seed=42
    )
    
    # Generate validation data
    print("\n🔧 Generating validation data...")
    X_bytes_val, X_entropy_val, y_val = generate_dataset(
        n_samples=1000,
        chunk_size=512,
        seed=123
    )
    
    # Save datasets
    print("\n💾 Saving datasets...")
    save_dataset(X_bytes_train, X_entropy_train, y_train, "train_data.npz")
    save_dataset(X_bytes_val, X_entropy_val, y_val, "val_data.npz")
    
    print("\n📊 Dataset Statistics:")
    print(f"Training samples: {len(X_bytes_train)}")
    print(f"Validation samples: {len(X_bytes_val)}")
    print(f"Bytes shape: {X_bytes_train.shape}")
    print(f"Entropy shape: {X_entropy_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    print(f"Malware ratio: {y_train.mean():.2%}")
    
    print("\n" + "=" * 60)
    print("✅ Data processing complete!")
    print("=" * 60)
