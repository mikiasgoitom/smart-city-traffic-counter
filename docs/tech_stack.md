# Technology Stack & Rationale

This document outlines the technologies, languages, and packages selected for the Smart City Traffic Counter project, along with the reasoning behind each choice.

## Programming Language

### **Python**
- **Reasoning**: Python is the de facto standard for Machine Learning and Computer Vision. It offers a vast ecosystem of libraries, ease of prototyping, and seamless integration with deep learning frameworks.

## Core AI Model

### **YOLO12n (You Only Look Once v12 - Nano)**
- **Description**: The latest and most advanced iteration of the YOLO object detection architecture. The 'n' (nano) variant is optimized for speed and efficiency.
- **Reasoning**:
    - **Real-time Performance**: For a live traffic management system, low latency is critical. YOLO12n provides high inference speeds suitable for processing live video streams even on edge devices.
    - **Accuracy**: Despite its small size, it offers robust detection capabilities, minimizing false positives and negatives in complex traffic scenes.
    - **Documentation**: As noted, it comes with robust documentation and support, facilitating smoother implementation.

## Key Libraries & Packages

### **1. OpenCV (`opencv-python`)**
- **Usage**: Video stream handling, frame manipulation, and drawing annotations (bounding boxes, text).
- **Reasoning**: OpenCV is the industry standard for image processing. It is highly optimized and provides all necessary tools to read video frames and display the output.

### **2. Ultralytics (or equivalent YOLO12 provider)**
- **Usage**: Loading the YOLO12n model, performing inference.
- **Reasoning**: Provides a high-level API to interact with YOLO models, making it easy to load weights, run predictions, and handle results.

### **3. Supervision (Optional but Recommended)**
- **Usage**: Advanced tracking, filtering, and counting utilities.
- **Reasoning**: The `supervision` library simplifies the implementation of counting lines and zones, abstracting the complex geometry math required to determine when a vehicle crosses a specific line.

### **4. NumPy**
- **Usage**: Matrix operations and array handling.
- **Reasoning**: Essential for efficient image representation (images are essentially NumPy arrays in OpenCV) and numerical calculations required for tracking logic.

## System Architecture

1.  **Video Source** -> **Frame Extraction** (OpenCV)
2.  **Inference** (YOLO12n) -> **Detections** (Bounding Boxes, Classes)
3.  **Tracker** -> **Object IDs**
4.  **Counter Logic** -> **Update Counts**
5.  **Annotator** -> **Display Result**
