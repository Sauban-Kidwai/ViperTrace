from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
from typing import List
import math
import os

app = FastAPI(title="ViperTrace Malware Detection API")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ViperNet(nn.Module):
    """
    Dual-input CNN architecture for malware detection
    
    Inputs:
        - x_bytes: (batch_size, 512) - Normalized byte sequences
        - x_entropy: (batch_size, 1) - Shannon entropy values
    
    Output:
        - (batch_size, 1) - Malware probability [0-1]
    """
    
    def __init__(self, chunk_size: int = 512):
        super(ViperNet, self).__init__()
        
        # 1D CNN for byte sequence processing
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_conv = nn.Dropout(0.3)
        
        # Calculate flattened size after convolutions
        # After 3 pooling layers: 512 -> 256 -> 128 -> 64
        self.cnn_output_size = 128 * (chunk_size // 8)
        
        # Fully connected layers (CNN output + entropy)
        self.fc1 = nn.Linear(self.cnn_output_size + 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.dropout_fc = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_bytes, x_entropy):
        """
        Forward pass with dual inputs
        
        Args:
            x_bytes: (batch_size, 512) - Byte sequences
            x_entropy: (batch_size, 1) - Entropy values
        
        Returns:
            (batch_size, 1) - Malware probability
        """
        # Reshape for 1D CNN: (batch, channels=1, sequence_length=512)
        x = x_bytes.unsqueeze(1)
        
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout_conv(x)
        
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout_conv(x)
        
        # Flatten CNN output
        x = x.view(x.size(0), -1)
        
        # Concatenate with entropy
        x = torch.cat([x, x_entropy], dim=1)
        
        # Fully connected layers
        x = self.dropout_fc(self.relu(self.fc1(x)))
        x = self.dropout_fc(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        
        return x


# Global variables
model = None
device = None
MODEL_PATH = "vipertrace.pth"


class ThreatDetection(BaseModel):
    offset: int
    confidence: float
    threat_type: str
    entropy_score: float


class ScanResponse(BaseModel):
    status: str
    threats: List[ThreatDetection]
    total_chunks: int
    scan_time: float


def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of byte data"""
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


def process_file_chunks(file_bytes: bytes, chunk_size: int = 512):
    """Process file in chunks and extract features"""
    chunks = []
    entropies = []
    
    # Split file into chunks
    for i in range(0, len(file_bytes), chunk_size):
        chunk = file_bytes[i:i + chunk_size]
        
        # Pad chunk if necessary
        if len(chunk) < chunk_size:
            chunk = chunk + b'\x00' * (chunk_size - len(chunk))
        
        # Calculate entropy
        entropy = calculate_entropy(chunk)
        entropies.append(entropy)
        
        # Convert to normalized array for CNN
        chunk_array = np.frombuffer(chunk, dtype=np.uint8)
        chunk_normalized = chunk_array.astype(np.float32) / 255.0
        chunks.append(chunk_normalized)
    
    return chunks, entropies


@app.on_event("startup")
async def load_model():
    """Load the PyTorch model on startup"""
    global model, device
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        if os.path.exists(MODEL_PATH):
            model = ViperNet(chunk_size=512)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()  # Set to evaluation mode
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            print(f"✓ Using device: {device}")
        else:
            print(f"⚠ Warning: Model file {MODEL_PATH} not found. Using mock predictions.")
            model = None
    except Exception as e:
        print(f"⚠ Error loading model: {e}. Using mock predictions.")
        model = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "ViperTrace Malware Detection API",
        "status": "online",
        "model_loaded": model is not None
    }


@app.post("/scan", response_model=ScanResponse)
async def scan_file(file: UploadFile = File(...)):
    """
    Scan uploaded file for malware signatures
    
    - Accepts any file type
    - Processes in 512-byte chunks
    - Returns threats with confidence > 85% (offset + confidence score)
    """
    import time
    start_time = time.time()
    
    try:
        # Read file contents
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Process file in chunks
        chunks, entropies = process_file_chunks(file_bytes, chunk_size=512)
        total_chunks = len(chunks)
        
        # Prepare data for model prediction
        threats = []
        
        if model is not None:
            # Real model prediction with PyTorch
            model.eval()
            
            with torch.no_grad():
                # Convert to tensors
                chunks_tensor = torch.FloatTensor(np.array(chunks)).to(device)
                entropies_tensor = torch.FloatTensor(entropies).reshape(-1, 1).to(device)
                
                # Get predictions
                predictions = model(chunks_tensor, entropies_tensor)
                predictions = predictions.cpu().numpy()
            
            # Analyze predictions
            for idx, (pred, entropy) in enumerate(zip(predictions, entropies)):
                confidence = float(pred[0]) * 100  # Convert to percentage
                
                if confidence > 85.0:
                    # Determine threat type based on entropy
                    threat_type = "Malware Detected"
                    if entropy > 7.5:
                        threat_type = "Encrypted Malware"
                    elif entropy < 3.0:
                        threat_type = "Suspicious Pattern"
                    
                    # Calculate byte offset
                    offset = idx * 512
                    
                    threats.append(ThreatDetection(
                        offset=offset,
                        confidence=round(confidence, 2),
                        threat_type=threat_type,
                        entropy_score=round(entropy, 3)
                    ))
        else:
            # Mock prediction for testing (when model not available)
            avg_entropy = np.mean(entropies)
            
            # Simulate threat detection based on entropy
            if avg_entropy > 7.0 or avg_entropy < 2.0:
                mock_confidence = min(95.0, 80.0 + (avg_entropy * 2))
                threat_type = "Encrypted Malware" if avg_entropy > 7.0 else "Suspicious Pattern"
                
                threats.append(ThreatDetection(
                    offset=0,
                    confidence=round(mock_confidence, 2),
                    threat_type=threat_type,
                    entropy_score=round(avg_entropy, 3)
                ))
        
        scan_time = time.time() - start_time
        
        return ScanResponse(
            status="clean" if len(threats) == 0 else "threats_detected",
            threats=threats,
            total_chunks=total_chunks,
            scan_time=round(scan_time, 3)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
