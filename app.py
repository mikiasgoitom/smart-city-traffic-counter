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

@st.cache_resource
def load_model(path):
    return YOLO(path)

# --- CONFIG ---
st.set_page_config(page_title="Smart City Traffic Demo", layout="wide")
st.title("🚦 Smart City Traffic Management System")

# Sidebar Configuration
st.sidebar.header("Model Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
model_path = 'yolo12m.pt' # Using Nano for speed

st.sidebar.header("Speed Calibration")
st.sidebar.info("Adjust points to match road perspective.")
pt1_x = st.sidebar.slider("Top-Left X", 0, 1280, 460)
pt1_y = st.sidebar.slider("Top-Left Y", 0, 720, 350)
pt2_x = st.sidebar.slider("Top-Right X", 0, 1280, 820)
pt2_y = st.sidebar.slider("Top-Right Y", 0, 720, 350)
pt3_x = st.sidebar.slider("Bottom-Right X", 0, 1280, 1200)
pt3_y = st.sidebar.slider("Bottom-Right Y", 0, 720, 700)
pt4_x = st.sidebar.slider("Bottom-Left X", 0, 1280, 100)
pt4_y = st.sidebar.slider("Bottom-Left Y", 0, 720, 700)

real_length = st.sidebar.number_input("Real Road Length (m)", value=25)
real_width = st.sidebar.number_input("Real Road Width (m)", value=8)

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader("Upload Traffic Video", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    try:
        col1, col2 = st.columns([3, 1])
        st_frame = col1.empty()
        
        with col2:
            st.markdown("### Statistics")
            kpi1 = st.empty() # Total Count
            kpi2 = st.empty() # Avg Speed
            
            st.markdown("---")
            st.write("Debug Info:")
            st_debug = st.empty()

        if st.button("Start Processing"):
            # Load Model (Cached)
            model = load_model(model_path)
            
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0: fps = 30
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Init Supervision
            tracker = sv.ByteTrack()
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            
            start_point = sv.Point(0, int(frame_height * 0.5))
            end_point = sv.Point(frame_width, int(frame_height * 0.5))
            line_zone = sv.LineZone(start=start_point, end=end_point)
            line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

            # Use deque correctly for smoothing
            # Store last 0.5 seconds of data (approx 15 frames if 30fps)
            coordinates = defaultdict(lambda: deque(maxlen=int(fps * 0.5)))

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Update Transformer
                SOURCE = np.array([
                    [pt1_x, pt1_y], [pt2_x, pt2_y],
                    [pt3_x, pt3_y], [pt4_x, pt4_y]
                ])
                TARGET = np.array([
                    [0, 0], [real_width, 0],
                    [real_width, real_length], [0, real_length]
                ])
                view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

                # Inference
                results = model(frame, conf=conf_threshold, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                
                # Filter for vehicles only (COCO IDs: 2=car, 3=motorcycle, 5=bus, 7=truck)
                detections = detections[np.isin(detections.class_id, [2, 3, 5, 7])]
                
                detections = tracker.update_with_detections(detections)
                line_zone.trigger(detections=detections)

                # Speed Calc
                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                points_transformed = view_transformer.transform_points(points)
                
                labels = []
                current_speeds = []

                for tracker_id, class_id, point_real in zip(detections.tracker_id, detections.class_id, points_transformed):
                    
                    # Add current point to history
                    coordinates[tracker_id].append(point_real)
                    
                    speed = 0
                    if len(coordinates[tracker_id]) > 1:
                        # Calculate distance between current point and the oldest point in buffer
                        # This smooths out the jitter over several frames
                        last_point = coordinates[tracker_id][-1]
                        first_point = coordinates[tracker_id][0]
                        
                        distance = np.linalg.norm(last_point - first_point)
                        time_gap = len(coordinates[tracker_id]) / fps
                        
                        if time_gap > 0:
                            speed_ms = distance / time_gap
                            speed_kmh = speed_ms * 3.6
                            
                            if speed_kmh < 200: # Filter unrealistic speeds
                                speed = int(speed_kmh)
                                current_speeds.append(speed)

                    labels.append(f"#{tracker_id} {speed} km/h")

                # 4. Annotation
                cv2.polylines(frame, [SOURCE.astype(np.int32)], True, (0, 255, 255), 2)
                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

                # 5. Display
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(annotated_frame, channels="RGB", use_container_width=True)

                total_count = line_zone.in_count + line_zone.out_count
                avg_speed_val = int(sum(current_speeds) / len(current_speeds)) if current_speeds else 0
                
                kpi1.metric("Total Vehicles", total_count)
                kpi2.metric("Avg Speed", f"{avg_speed_val} km/h")
                st_debug.text(f"Tracking {len(detections)} vehicles")

            cap.release()
            
    finally:
        # Cleanup temp file
        try:
            os.unlink(video_path)
        except:
            pass