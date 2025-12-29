import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import tempfile
import os

# --- UTILS ---
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# --- CONFIG ---
st.set_page_config(page_title="Smart City Traffic Demo", layout="wide")
st.title("🚦 Smart City Traffic Management System")

# Sidebar Configuration
st.sidebar.header("Model Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
model_path = 'yolo12n.pt'

st.sidebar.header("Speed Calibration (Source Polygon)")
st.sidebar.info("Adjust these points to match the road lanes.")
# Default values based on your previous code
pt1_x = st.sidebar.slider("Top-Left X", 0, 1280, 460)
pt1_y = st.sidebar.slider("Top-Left Y", 0, 720, 350)
pt2_x = st.sidebar.slider("Top-Right X", 0, 1280, 820)
pt2_y = st.sidebar.slider("Top-Right Y", 0, 720, 350)
pt3_x = st.sidebar.slider("Bottom-Right X", 0, 1280, 1200)
pt3_y = st.sidebar.slider("Bottom-Right Y", 0, 720, 700)
pt4_x = st.sidebar.slider("Bottom-Left X", 0, 1280, 100)
pt4_y = st.sidebar.slider("Bottom-Left Y", 0, 720, 700)

real_length = st.sidebar.number_input("Real Road Length (meters)", value=25)
real_width = st.sidebar.number_input("Real Road Width (meters)", value=8)

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader("Upload Traffic Video", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Setup
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Layout for video and stats
    col1, col2 = st.columns([3, 1])
    st_frame = col1.empty()
    st_stat_count = col2.empty()
    st_stat_speed = col2.empty()

    # Init Supervision
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Line Zone
    start_point = sv.Point(0, int(frame_height * 0.5))
    end_point = sv.Point(frame_width, int(frame_height * 0.5))
    line_zone = sv.LineZone(start=start_point, end=end_point)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    # Speed Storage
    coordinates = defaultdict(lambda: deque(maxlen=fps))

    start_button = st.button("Start Processing")

    if start_button:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 1. Update Transformer based on Sliders
            SOURCE = np.array([
                [pt1_x, pt1_y], [pt2_x, pt2_y],
                [pt3_x, pt3_y], [pt4_x, pt4_y]
            ])
            TARGET = np.array([
                [0, 0], [real_width, 0],
                [real_width, real_length], [0, real_length]
            ])
            view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

            # 2. Inference
            results = model(frame, conf=conf_threshold, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            line_zone.trigger(detections=detections)

            # 3. Speed Calc
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points_transformed = view_transformer.transform_points(points)
            
            labels = []
            current_speeds = []

            for tracker_id, class_id, point_real in zip(detections.tracker_id, detections.class_id, points_transformed):
                speed = 0
                if tracker_id in coordinates:
                    prev_point = coordinates[tracker_id]
                    distance = np.linalg.norm(point_real - prev_point)
                    speed_kmh = (distance * fps) * 3.6
                    if speed_kmh > 200: speed_kmh = 0
                    speed = int(speed_kmh)
                    current_speeds.append(speed)
                
                coordinates[tracker_id] = point_real
                labels.append(f"#{tracker_id} {speed} km/h")

            # 4. Annotation
            # Draw the calibration polygon so you can see it
            cv2.polylines(frame, [SOURCE.astype(np.int32)], True, (0, 255, 255), 2)
            
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

            # 5. Display in Streamlit
            # Convert BGR to RGB
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, channels="RGB", use_container_width=True)

            # Update Stats
            st_stat_count.metric("Total Vehicles", line_zone.in_count + line_zone.out_count)
            if current_speeds:
                avg_speed = sum(current_speeds) / len(current_speeds)
                st_stat_speed.metric("Avg Speed (Current Frame)", f"{int(avg_speed)} km/h")

        cap.release()