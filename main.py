import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque

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

def main():
    # Load Model
    model = YOLO('yolo12n.pt')

    # Video Setup
    video_path = "./data/traffic4.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- CONFIGURATION FOR SPEED ESTIMATION ---
    # 1. Define 4 points on the screen that form a rectangle on the road
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    # YOU MUST ADJUST THESE TO MATCH YOUR VIDEO ROAD
    SOURCE = np.array([
        [460, 350],  # Top-Left
        [820, 350],  # Top-Right
        [1200, 700], # Bottom-Right
        [100, 700]   # Bottom-Left
    ])

    # 2. Define real-world dimensions of that area (in meters)
    # e.g., a 25 meter long stretch of road that is 8 meters wide
    TARGET_WIDTH = 8
    TARGET_LENGTH = 25
    
    TARGET = np.array([
        [0, 0], 
        [TARGET_WIDTH, 0], 
        [TARGET_WIDTH, TARGET_LENGTH], 
        [0, TARGET_LENGTH]
    ])

    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    # -------------------------------------------

    # Initialize Supervision Tools
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Speed storage
    coordinates = defaultdict(lambda: deque(maxlen=fps)) # Store history of positions

    # Counting Line
    start_point = sv.Point(0, int(frame_height * 0.5))
    end_point = sv.Point(frame_width, int(frame_height * 0.5))
    line_zone = sv.LineZone(start=start_point, end=end_point)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    print("Starting Tracking, Counting & Speed Estimation...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Inference
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Update Tracker
        detections = tracker.update_with_detections(detections)

        # Update Counting Line
        line_zone.trigger(detections=detections)

        # --- SPEED CALCULATION ---
        # Get center points of bounding boxes
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        
        # Transform points to real-world coordinates (meters)
        points_transformed = view_transformer.transform_points(points)
        
        labels = []
        for tracker_id, class_id, point_real in zip(detections.tracker_id, detections.class_id, points_transformed):
            
            # Calculate speed if we have previous data
            speed = 0
            if tracker_id in coordinates:
                # Get previous position (from 0.5 seconds ago roughly)
                prev_point = coordinates[tracker_id]
                
                # Distance in meters
                distance = np.linalg.norm(point_real - prev_point)
                
                # Speed = Distance / Time
                # Since we update every frame, time is 1/fps. 
                # But calculating per frame is jittery. 
                # Simple smoothing: Speed (m/s) = distance * fps
                speed_ms = distance * fps
                speed_kmh = speed_ms * 3.6
                
                # Simple filtering to remove noise
                if speed_kmh > 200: speed_kmh = 0 
                speed = int(speed_kmh)

            # Update current position
            coordinates[tracker_id] = point_real

            labels.append(f"#{tracker_id} {speed} km/h")

        # Draw Polygon for reference (so you can see where the speed zone is)
        cv2.polylines(frame, [SOURCE.astype(np.int32)], True, (0, 255, 255), 2)

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

        cv2.imshow("Smart City Traffic Counter", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()