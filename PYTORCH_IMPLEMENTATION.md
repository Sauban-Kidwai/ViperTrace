# ViperTrace - PyTorch Implementation Summary

## ✅ What Has Been Updated

The entire backend has been migrated from TensorFlow/Keras to **PyTorch** with a sophisticated dual-input architecture as requested.

---

## 🎯 Key Changes

### **1. Backend Architecture (PyTorch)**

#### **ViperNet Model** (`ViperNet(nn.Module)`)

- ✅ **Dual-input architecture** as specified:
  - `x_bytes`: (batch_size, 512) - Normalized byte sequences
  - `x_entropy`: (batch_size, 1) - Shannon entropy values
- ✅ **1D CNN** for byte sequence processing:

  - 3 convolutional layers (32 → 64 → 128 filters)
  - MaxPooling after each conv layer
  - Dropout for regularization

- ✅ **Feature concatenation**:

  - CNN output is flattened
  - Concatenated with entropy value
  - Fed into fully connected layers

- ✅ **Output**: Sigmoid activation for malware probability [0-1]

---

### **2. Files Created/Updated**

#### **New Files:**

1. **`backend/process_data.py`**

   - Shannon entropy calculation
   - Synthetic data generation (5,000 training + 1,000 validation samples)
   - Dataset save/load functions

2. **`backend/train_model.py`** (Completely rewritten)
   - `ViperNet` class definition
   - PyTorch Dataset and DataLoader
   - Training and validation loops
   - Saves model as `vipertrace.pth`

#### **Updated Files:**

1. **`backend/main.py`** (Completely rewritten)

   - Redefines `ViperNet` class (for loading weights)
   - Loads `vipertrace.pth` on startup
   - POST `/scan` endpoint:
     - Accepts file upload
     - Processes in 512-byte chunks
     - Calculates entropy per chunk
     - Feeds both inputs to model
     - Returns threats with **offset** + confidence > 85%
   - CORS enabled for "\*"

2. **`backend/requirements.txt`**

   ```
   torch==2.1.2
   numpy==1.24.3
   fastapi==0.104.1
   uvicorn[standard]==0.24.0
   python-multipart==0.0.6
   pydantic==2.5.0
   ```

3. **`backend/Dockerfile`**

   - Updated to copy `vipertrace.pth` and `process_data.py`

4. **`frontend/app/page.tsx`**

   - Updated `ThreatDetection` interface:
     - Changed from `filename` and `chunks_analyzed`
     - To `offset` (byte offset in file)
   - Display offset in hex format: `0x0000`, `0x0200`, etc.

5. **`.gitignore` files**
   - Excludes `*.pth`, `*.pt`, `*.npz` (PyTorch models and data files)

---

## 🚀 How to Use

### **Step 1: Backend Setup**

```cmd
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### **Step 2: Train the Model**

```cmd
python train_model.py
```

**What happens:**

- Generates 5,000 training samples (synthetic malware vs clean data)
- Generates 1,000 validation samples
- Trains ViperNet for 25 epochs
- Saves best model as `vipertrace.pth`
- Reports accuracy, loss, etc.

### **Step 3: Run the Backend API**

```cmd
python main.py
```

or with auto-reload:

```cmd
uvicorn main:app --reload
```

Backend runs on `http://localhost:8000`

### **Step 4: Frontend Setup** (In new terminal)

```cmd
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`

---

## 📊 API Response Format

### **POST /scan**

**Request:**

- Multipart form data with file

**Response:**

```json
{
  "status": "threats_detected",
  "threats": [
    {
      "offset": 0,
      "confidence": 92.5,
      "threat_type": "Encrypted Malware",
      "entropy_score": 7.85
    },
    {
      "offset": 512,
      "confidence": 88.3,
      "threat_type": "Malware Detected",
      "entropy_score": 6.42
    }
  ],
  "total_chunks": 10,
  "scan_time": 0.123
}
```

**Key Points:**

- ✅ `offset`: Byte offset where threat was detected (0, 512, 1024, etc.)
- ✅ `confidence`: Percentage (only included if > 85%)
- ✅ `threat_type`: Based on entropy analysis
- ✅ `entropy_score`: Shannon entropy of the chunk

---

## 🔧 Model Architecture Details

```
ViperNet(
  (conv1): Conv1d(1, 32, kernel_size=(3,), padding=(1,))
  (conv2): Conv1d(32, 64, kernel_size=(3,), padding=(1,))
  (conv3): Conv1d(64, 128, kernel_size=(3,), padding=(1,))
  (pool): MaxPool1d(kernel_size=2)
  (dropout_conv): Dropout(p=0.3)
  (fc1): Linear(in_features=8193, out_features=256)
  (fc2): Linear(in_features=256, out_features=128)
  (fc3): Linear(in_features=128, out_features=1)
  (dropout_fc): Dropout(p=0.4)
  (relu): ReLU()
  (sigmoid): Sigmoid()
)
```

**Total Parameters: ~2.1M** (trainable)

---

## ✨ Features Implemented

✅ **Dual-input CNN** (bytes + entropy)  
✅ **512-byte chunk processing**  
✅ **Shannon entropy calculation**  
✅ **Offset-based threat reporting** (instead of chunk numbers)  
✅ **85% confidence threshold**  
✅ **CORS enabled** for frontend  
✅ **Mock predictions** when model not available  
✅ **CPU/GPU support** (auto-detects)  
✅ **Hex offset display** in UI (e.g., `0x0000`)

---

## 🎨 Frontend Updates

- Terminal-style log now shows:

  ```
  > THREAT 1: Encrypted Malware (92.5% confidence)
  >   └─ Offset: 0x0000 | Entropy: 7.850
  ```

- Threat cards display:
  ```
  Encrypted Malware          92.5%
  Offset: 0x0000 | Entropy: 7.850
  ```

---

## 🐋 Docker Deployment

```cmd
cd backend
docker build -t vipertrace-backend .
docker run -p 8000:8000 vipertrace-backend
```

---

## 📝 Next Steps

1. ✅ **All code is ready** - PyTorch implementation complete
2. 🔄 **Push to GitHub** - Use the git commands I provided earlier
3. 🚀 **Train model** - Run `python train_model.py`
4. ▶️ **Start servers** - Backend on 8000, Frontend on 3000
5. 🧪 **Test** - Upload files and see real-time malware detection!

---

## 🎯 Production Considerations

⚠️ **Current implementation uses synthetic data for demonstration.**

For production deployment:

1. Replace `process_data.py` with real malware dataset processing
2. Train on legitimate malware samples (VirusTotal, Malware Bazaar, etc.)
3. Implement additional security measures
4. Add authentication to the API
5. Set up monitoring and logging
6. Configure CORS for specific domains only

---

**You're all set! 🐍🔍**
