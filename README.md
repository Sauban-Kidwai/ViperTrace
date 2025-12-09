# ViperTrace 🐍🔍

Advanced AI-powered malware detection system using deep learning.

## 🚀 Tech Stack

- **Frontend:** Next.js 14 (App Router) + Tailwind CSS + Lucide React
- **Backend:** Python FastAPI
- **ML Engine:** TensorFlow/Keras
- **Deployment:** Vercel (Frontend) + Docker (Backend)

## 📁 Project Structure

```
/ViperTrace
  /backend
    - main.py           # FastAPI server
    - train_model.py    # Model training script
    - Dockerfile        # Container configuration
    - requirements.txt  # Python dependencies
  /frontend
    /app
      - page.tsx        # Main UI
      - layout.tsx      # Root layout
      - globals.css     # Global styles
    - package.json      # Node dependencies
  /data                 # Training data (optional)
```

## 🔧 Setup Instructions

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- pip

### Backend Setup

1. Navigate to backend folder:

```bash
cd backend
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Train the model (optional, creates vipertrace.keras):

```bash
python train_model.py
```

5. Start the FastAPI server:

```bash
python main.py
```

Or with auto-reload:

```bash
uvicorn main:app --reload
```

Backend will run on `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend folder:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Run the development server:

```bash
npm run dev
```

Frontend will run on `http://localhost:3000`

## 🎮 Usage

1. Start the backend server (port 8000)
2. Start the frontend server (port 3000)
3. Open `http://localhost:3000` in your browser
4. Drag and drop any file to scan for malware
5. View real-time analysis logs and threat detection results

## 🐋 Docker Deployment

### Build and run backend container:

```bash
cd backend
docker build -t vipertrace-backend .
docker run -p 8000:8000 vipertrace-backend
```

## 📊 How It Works

1. **File Upload:** User uploads a file through the drag-and-drop interface
2. **Chunking:** File is split into 512-byte chunks
3. **Feature Extraction:** Shannon entropy is calculated for each chunk
4. **CNN Analysis:** 1D Convolutional Neural Network analyzes patterns
5. **Threat Detection:** Results with >85% confidence are flagged as threats
6. **Results Display:** Terminal-style logs show real-time analysis

## 🔐 Security Features

- Entropy analysis for encrypted malware detection
- Deep learning pattern recognition
- Chunk-level threat identification
- Real-time scanning feedback

## 📝 API Endpoints

- `GET /` - Health check
- `POST /scan` - Upload and scan file
- `GET /health` - Detailed health status

## 🚀 Production Deployment

### Frontend (Vercel)

1. Push code to GitHub
2. Connect repository to Vercel
3. Set environment variable: `NEXT_PUBLIC_API_URL=https://your-backend-url.com`
4. Deploy

### Backend (Docker)

1. Build Docker image: `docker build -t vipertrace-backend .`
2. Deploy to your preferred platform (AWS, Azure, GCP)
3. Ensure CORS is configured for your frontend domain

## 📄 License

MIT

## ⚠️ Disclaimer

This is a demonstration project. For production use with real malware detection, train the model on legitimate malware datasets and implement additional security measures.
# ViperTrace
