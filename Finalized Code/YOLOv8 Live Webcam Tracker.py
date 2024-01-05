# https://github.com/SkalskiP/yolov8-live/tree/master
# https://www.youtube.com/watch?v=QV85eYOb7gk&ab_channel=Roboflow

'''
from ultralytics import YOLO

# Load the YOLO model from a checkpoint file
model = YOLO('best.pt')

# Make predictions on a video stream (source=0 for webcam)
# Set other parameters such as image size (imgsz), confidence threshold (conf), and show predictions (show)
result = model.predict(source=0, imgsz=640, conf=0.6, show=True)
'''

import cv2
import argparse
import time
from ultralytics import YOLO
import supervision as sv
import numpy as np

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Initialize variables for fps calculation
    start_time = time.time()
    frame_count = 0

    source = 0    # 0 denotes Webcam
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while (1):
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        # Calculate fps
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Display fps in the frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)        
        cv2.imshow("YOLO Predictions with FPS", frame)

        # Break the loop if 'q' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

if __name__ == "__main__":
    main()
