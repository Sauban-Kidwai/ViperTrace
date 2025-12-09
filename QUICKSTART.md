# 🚀 Quick Start Guide - ViperTrace (PyTorch)

## Prerequisites

- Python 3.11+
- Node.js 18+
- Git

---

## Setup & Run (5 Steps)

### 1️⃣ **Backend Setup**

```cmd
cd c:\Users\Sauban\Documents\Projects\ViperTrace\backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ **Train the Model**

```cmd
python train_model.py
```

⏱️ Takes ~5-10 minutes. Creates `vipertrace.pth` (~8 MB)

### 3️⃣ **Start Backend API**

```cmd
python main.py
```

✅ Backend running at `http://localhost:8000`

### 4️⃣ **Frontend Setup** (New terminal)

```cmd
cd c:\Users\Sauban\Documents\Projects\ViperTrace\frontend
npm install
npm run dev
```

✅ Frontend running at `http://localhost:3000`

### 5️⃣ **Push to GitHub**

```cmd
cd c:\Users\Sauban\Documents\Projects\ViperTrace
git add .
git commit -m "Complete PyTorch implementation with dual-input ViperNet"
git push -u origin main
```

---

## 🧪 Testing

1. Open `http://localhost:3000`
2. Drag and drop any file (PDF, EXE, ZIP, etc.)
3. Watch the terminal-style log for real-time analysis
4. See threat detections with:
   - Byte offset (hex format)
   - Confidence percentage
   - Entropy score
   - Threat classification

---

## 📂 Project Structure

```
ViperTrace/
├── backend/
│   ├── main.py              # FastAPI server (PyTorch)
│   ├── train_model.py       # ViperNet training script
│   ├── process_data.py      # Data generation & entropy
│   ├── requirements.txt     # PyTorch, FastAPI, etc.
│   ├── Dockerfile          # Container config
│   └── vipertrace.pth      # Trained model (after training)
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx        # Main UI (cyber theme)
│   │   ├── layout.tsx      # Root layout
│   │   └── globals.css     # Terminal styling
│   ├── package.json        # Next.js dependencies
│   └── ...
│
└── data/                   # Training data (optional)
```

---

## 🎯 Key Features

✅ **PyTorch Implementation** - Dual-input CNN architecture  
✅ **512-byte Chunking** - Processes files in small chunks  
✅ **Shannon Entropy** - Detects encrypted/packed malware  
✅ **Offset Reporting** - Shows exact byte location of threats  
✅ **85% Threshold** - Only reports high-confidence threats  
✅ **CORS Enabled** - Frontend can communicate with backend  
✅ **Cyber UI** - Black/green terminal aesthetic

---

## 🔍 API Endpoints

| Endpoint  | Method | Description        |
| --------- | ------ | ------------------ |
| `/`       | GET    | Health check       |
| `/scan`   | POST   | Upload & scan file |
| `/health` | GET    | Detailed status    |

---

## 💡 Common Commands

**Restart Backend:**

```cmd
cd backend
venv\Scripts\activate
python main.py
```

**Restart Frontend:**

```cmd
cd frontend
npm run dev
```

**Retrain Model:**

```cmd
cd backend
venv\Scripts\activate
python train_model.py
```

**Docker Backend:**

```cmd
cd backend
docker build -t vipertrace .
docker run -p 8000:8000 vipertrace
```

---

## ⚠️ Troubleshooting

**"Model not found" warning:**

- Run `python train_model.py` first
- API will use mock predictions if model missing

**"Cannot connect to backend":**

- Ensure backend is running on port 8000
- Check CORS settings in `main.py`

**Import errors:**

- Activate virtual environment: `venv\Scripts\activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**Frontend build errors:**

- Delete `node_modules` and `.next`
- Run `npm install` again

---

## 📊 Model Performance

After training, you should see:

- **Validation Accuracy**: ~70-85% (on synthetic data)
- **Training Time**: 5-10 minutes (CPU)
- **Model Size**: ~8 MB
- **Inference Speed**: <0.1s per file

⚠️ **Note**: Performance will improve significantly when trained on real malware datasets.

---

## 🎨 Customization

**Change confidence threshold:**

```python
# In main.py, line ~250
if confidence > 85.0:  # Change this value
```

**Adjust chunk size:**

```python
# In main.py and train_model.py
chunks, entropies = process_file_chunks(file_bytes, chunk_size=512)  # Change 512
```

**Modify UI colors:**

```css
/* In frontend/app/globals.css */
/* Change green to any color */
```

---

**Happy malware hunting! 🐍🔍**
