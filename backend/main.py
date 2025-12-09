from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from typing import List
import io
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

# Global variable to store the model
model = None
MODEL_PATH = "vipertrace.keras"

class ThreatDetection(BaseModel):
    filename: str
    confidence: float
    threat_type: str
    chunks_analyzed: int
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
    """Load the Keras model on startup"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
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
    - Returns threats with confidence > 85%
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
            # Real model prediction
            chunks_array = np.array(chunks)
            # Reshape for CNN: (batch_size, 512, 1) for 1D CNN
            chunks_array = chunks_array.reshape(-1, 512, 1)
            
            # Get predictions
            predictions = model.predict(chunks_array, verbose=0)
            
            # Analyze predictions
            for idx, (pred, entropy) in enumerate(zip(predictions, entropies)):
                confidence = float(pred[0]) * 100  # Convert to percentage
                
                if confidence > 85.0:
                    threat_type = "Malware Detected"
                    if entropy > 7.5:
                        threat_type = "Encrypted Malware"
                    elif entropy < 3.0:
                        threat_type = "Suspicious Pattern"
                    
                    threats.append(ThreatDetection(
                        filename=file.filename,
                        confidence=round(confidence, 2),
                        threat_type=threat_type,
                        chunks_analyzed=idx + 1,
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
                    filename=file.filename,
                    confidence=round(mock_confidence, 2),
                    threat_type=threat_type,
                    chunks_analyzed=total_chunks,
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
