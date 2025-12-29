# Smart City Traffic Counter

## Project Overview
This project is a demonstration of a **Smart City Traffic Management System** component focused on real-time vehicle counting. Utilizing the state-of-the-art **YOLO12n** (You Only Look Once, version 12 nano) object detection model, this system analyzes live video footage to detect, track, and count vehicles with high accuracy and efficiency.

## Implementation Strategy

### 1. Input Acquisition
- **Source**: The system is designed to ingest live video feeds. For demonstration purposes, we will simulate a live feed using a pre-recorded traffic video file or a webcam stream.
- **Preprocessing**: Frames are resized and normalized to match the input requirements of the YOLO12n model.

### 2. Object Detection (YOLO12n)
- We utilize **YOLO12n**, the latest iteration in the YOLO family, known for its superior speed-to-accuracy ratio.
- The model is configured to detect specific classes relevant to traffic: `car`, `motorcycle`, `bus`, `truck`.

### 3. Object Tracking & Counting
- **Tracking**: To ensure vehicles are counted exactly once as they move across the frame, we implement an object tracking algorithm (e.g., ByteTrack or SORT) that assigns unique IDs to detected objects across consecutive frames.
- **Counting Logic**: A virtual "counting line" or "zone" is defined within the video frame. When a tracked vehicle's centroid crosses this line, the counter is incremented.

### 4. Visualization & Output
- The system outputs a processed video stream with bounding boxes around detected vehicles, unique IDs, and a real-time counter overlay.
- Data can potentially be logged to a database for further traffic analysis.

## Getting Started
*(Instructions for installation and usage will be added here)*
