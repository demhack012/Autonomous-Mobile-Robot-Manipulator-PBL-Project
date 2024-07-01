import cv2
from ultralytics import YOLO
import numpy as np

# Load your trained YOLOv8 model
model = YOLO('/root/catkin_ws/src/pbl/scripts/best.pt')  # Replace with the path to your model's weights

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame is read correctly
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model(frame)
    print("f")

    # Draw bounding boxes on the frame
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        conf = detection.conf.item()
        cls = int(detection.cls.item())
        label = f"{model.names[cls]} {conf:.2f}"
        print(model.names[cls])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with bounding boxes
    # cv2.imshow('YOLOv8 Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
