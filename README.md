# Smart City Traffic Counter & Speed Analyzer

This project is an interactive web application for real-time traffic analysis. It uses state-of-the-art object detection to count vehicles, estimate their speed, and provide a live dashboard for monitoring and calibration.

## Features

- **Real-time Detection & Tracking**: Utilizes **YOLOv12** (Nano and Medium models) for high-accuracy object detection and **ByteTrack** for robust tracking of individual vehicles.
- **Vehicle Counting**: A virtual line across the screen counts vehicles as they pass.
- **Speed Estimation**: Calculates the real-world speed (in km/h) of tracked vehicles using perspective transformation.
- **Interactive Web Dashboard**: Built with **Streamlit**, the UI allows for easy video upload and live visualization.
- **Live Calibration**:
  - Adjust the speed estimation zone in real-time using sliders.
  - Tune model performance by changing the **Confidence** and **IoU (NMS)** thresholds.
- **GPU Accelerated**: Full support for NVIDIA GPUs via PyTorch-CUDA to achieve high FPS for smooth processing.

## Technology Stack

- **Language**: Python
- **AI Model**: YOLOv12 (from Ultralytics)
- **Computer Vision**: OpenCV, Supervision
- **Web Framework**: Streamlit
- **Core Libraries**: NumPy, PyTorch

## Getting Started

### Prerequisites

- Python 3.8+
- An NVIDIA GPU with CUDA installed (recommended for good performance)

### 1. Setup Environment

```bash
# Clone the repository (if you haven't already)
# git clone <your-repo-url>
# cd smart-city-traffic-counter

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python numpy supervision streamlit
```

### 2. Install PyTorch with GPU Support (Recommended)

For a significant performance boost, reinstall PyTorch to use your GPU.

```bash
# First, uninstall the CPU version
pip uninstall torch torchvision torchaudio

# Then, install the CUDA-enabled version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Run the Application

```bash
streamlit run app.py
```

This will open the dashboard in your web browser.

## How to Use the Dashboard

1.  **Upload a Video**: Use the file uploader to select a traffic video (`.mp4`, `.avi`).
2.  **Configure Settings (Sidebar)**:
    - **Model Size**: Choose between `yolo12n` (faster) and `yolo12m` (more accurate).
    - **Confidence Threshold**: Increase to reduce false detections.
    - **IoU Threshold**: Increase this (e.g., to `0.7`) if cars in traffic are overlapping and being missed.
    - **Speed Calibration**: Adjust the 8 sliders to draw a yellow polygon that matches the perspective of the road lanes in your video. This is **critical** for accurate speed estimation.
3.  **Start Processing**: Click the "Start Processing" button and observe the results.
