"""
ViperTrace Model Training Script

This script creates and trains a simple CNN model for malware detection.
In production, you would train this on a real malware dataset.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

def create_model(input_shape=(512, 1)):
    """
    Create a 1D CNN model for malware detection
    
    Args:
        input_shape: Shape of input chunks (512 bytes, 1 channel)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # 1D Convolutional layers
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer (binary classification: malware or not)
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def generate_synthetic_data(n_samples=1000, chunk_size=512):
    """
    Generate synthetic training data for demonstration
    
    In production, replace this with real malware samples
    """
    X_train = []
    y_train = []
    
    for i in range(n_samples):
        # Generate random bytes
        if i < n_samples // 2:
            # "Clean" files - more structured patterns
            chunk = np.random.randint(0, 128, chunk_size, dtype=np.uint8)
        else:
            # "Malware" files - more random/encrypted patterns
            chunk = np.random.randint(0, 256, chunk_size, dtype=np.uint8)
        
        # Normalize
        chunk_normalized = chunk.astype(np.float32) / 255.0
        X_train.append(chunk_normalized)
        
        # Label: 0 = clean, 1 = malware
        y_train.append(0 if i < n_samples // 2 else 1)
    
    X_train = np.array(X_train).reshape(-1, chunk_size, 1)
    y_train = np.array(y_train)
    
    return X_train, y_train

def train_model(epochs=10, batch_size=32):
    """Train and save the model"""
    print("🔧 Creating model architecture...")
    model = create_model()
    
    print("\n📊 Model Summary:")
    model.summary()
    
    print("\n📦 Generating synthetic training data...")
    print("⚠️  NOTE: In production, use real malware datasets!")
    X_train, y_train = generate_synthetic_data(n_samples=2000)
    X_val, y_val = generate_synthetic_data(n_samples=400)
    
    print(f"\n✓ Training samples: {len(X_train)}")
    print(f"✓ Validation samples: {len(X_val)}")
    
    print("\n🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate
    print("\n📈 Final Evaluation:")
    loss, accuracy, precision, recall = model.evaluate(X_val, y_val, verbose=0)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Save model
    model_path = "vipertrace.keras"
    model.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    return model, history

if __name__ == "__main__":
    print("=" * 60)
    print("      ViperTrace Model Training")
    print("=" * 60)
    
    # Train the model
    model, history = train_model(epochs=15, batch_size=32)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. The model is saved as 'vipertrace.keras'")
    print("2. Run 'python main.py' or 'uvicorn main:app --reload' to start the API")
    print("3. The API will automatically load this model")
