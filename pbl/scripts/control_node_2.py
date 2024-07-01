#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from std_msgs.msg import Uint8
from std_msgs.msg import Int16
from geometry_msgs.msg import Twist
import cv2
from ultralytics import YOLO
import numpy as np


# PID Controller class
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def compute(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


# Load your trained YOLOv8 model
model = YOLO(
    "/root/catkin_ws/src/pbl/scripts/best.pt"
)  # Replace with the path to your model's weights

# List of color labels
color_labels = ["Red", "Green", "purple", "Pink", "Yellow"]

# Initialize a variable to store the target color
target_color = None

# Initialize a Twist message for robot movement
cmd_vel_pub = None

# Initialize PID controllers for linear and lateral velocity
linear_pid = PID(Kp=0.5, Ki=0.0, Kd=0.1, setpoint=0.6)  # Target area ratio is 60%
lateral_pid = PID(Kp=0.5, Ki=0.0, Kd=0.2)

linear_pid_rectangle = PID(
    Kp=0.5, Ki=0.0, Kd=0.1, setpoint=0.3
)  # Target area ratio is 40%

moving_average_filter_coeff = 0.5

linear_speed_prev = 0
lateral_speed_prev = 0

done = 0
box = 0
reached_box = 0
distance = 0

lower_hsv = np.array([144, 10, 204])
upper_hsv = np.array([167, 198, 255])


def color_callback(msg):
    global target_color
    global done
    target_color = msg.data
    done = 0


def ultrasonic_callback(msg):
    global distance
    distance = msg.data


def main():
    global cmd_vel_pub
    global target_color
    global linear_speed_prev
    global lateral_speed_prev
    global done
    global box
    global reached_box
    global distance

    rospy.init_node("yolov8_color_detector", anonymous=True)

    # Subscribe to the color topic
    rospy.Subscriber("color", String, color_callback)
    rospy.Subscriber("ultrasonic", Int16, ultrasonic_callback)

    # Publisher for the robot's velocity commands
    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    robotic_arm_pub = rospy.Publisher("/servo", Uint8, queue_size=10)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Check if webcam is opened successfully
    if not cap.isOpened():
        rospy.logerr("Error: Could not open webcam.")
        return

    rospy.loginfo("YOLOv8 color detector node started.")

    while not rospy.is_shutdown():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is read correctly
        if not ret:
            rospy.logerr("Error: Could not read frame.")
            break

        # Perform inference on the frame
        results = model(frame)
        max_area = 0
        best_detection = None
        # Calculate the total frame area
        frame_area = frame.shape[0] * frame.shape[1]

        # Filter detections based on the target color
        if target_color:
            for detection in results[0].boxes:
                conf = detection.conf.item()
                if conf >= 0.8:
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])
                    cls = int(detection.cls.item())
                    label = model.names[cls]

                    if label == target_color:
                        # Calculate the area of the bounding box
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            best_detection = detection

            # Draw the bounding box of the largest detection of the target color
            if best_detection:
                x1, y1, x2, y2 = map(int, best_detection.xyxy[0])
                conf = best_detection.conf.item()
                label = f"{target_color} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

                # Calculate the area of the detected ball
                ball_area = (x2 - x1) * (y2 - y1)
                area_ratio = ball_area / frame_area

                # Calculate the center of the ball
                ball_center_x = (x1 + x2) / 2
                frame_center_x = frame.shape[1] / 2
                error_x = frame_center_x - ball_center_x

                # Compute PID values
                linear_speed = linear_pid.compute(
                    area_ratio
                ) * moving_average_filter_coeff + linear_speed_prev * (
                    1 - moving_average_filter_coeff
                )
                lateral_speed = lateral_pid.compute(
                    error_x / frame.shape[1]
                ) * moving_average_filter_coeff + lateral_speed_prev * (
                    1 - moving_average_filter_coeff
                )
                linear_speed_prev = linear_speed
                lateral_speed_prev = lateral_speed

                # Create Twist message for robot movement
                twist = Twist()

                # Move towards the ball if the area is less than 60% of the frame area
                if area_ratio < 0.6:
                    twist.linear.x = (
                        linear_speed  # Move forward based on PID controller output
                    )

                if abs(error_x) > 0.1 * frame.shape[1]:
                    twist.angular.z = lateral_speed

                # Stop moving and reset target color if the area is 60% or more
                if (area_ratio > 0.6) and (abs(error_x) < 0.1 * frame.shape[1]):
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    best_detection = None
                    target_color = None
                    done = 1

                # Publish the Twist message
                cmd_vel_pub.publish(twist)
            elif done == 0:
                # If no ball of the specified color is detected, rotate slowly
                twist = Twist()
                twist.angular.z = 0.2  # Adjust the rotation speed as needed
                cmd_vel_pub.publish(twist)
            else:
                # Stop the robot if the ball is detected and the area is 60% or more
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                cmd_vel_pub.publish(twist)
                if not box:
                    robotic_arm_pub.publish(0)
                    robotic_arm_pub.publish(1)
                    robotic_arm_pub.publish(0)
                    box = 1
                else:
                    mask = cv2.inRange(frame, lower_hsv, upper_hsv)
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:
                            x, y, w, h = cv2.boundingRect(contour)
                            cx = x + w / 2
                            area_rectangle = w * h
                            area_ratio_rectangle = area_rectangle / frame_area
                            error_x_rectangle = frame_center_x - cx
                            linear_speed_rectangle = linear_pid_rectangle.compute(
                                area_ratio_rectangle
                            ) * moving_average_filter_coeff + linear_speed_prev * (
                                1 - moving_average_filter_coeff
                            )
                            lateral_speed_rectangle = lateral_pid.compute(
                                error_x_rectangle / frame.shape[1]
                            ) * moving_average_filter_coeff + lateral_speed_prev * (
                                1 - moving_average_filter_coeff
                            )
                            linear_speed_prev = linear_speed_rectangle
                            lateral_speed_prev = lateral_speed_rectangle
                            twist = Twist()
                            if area_ratio_rectangle < 0.4:
                                twist.linear.x = linear_speed_rectangle
                            if abs(error_x_rectangle) > 0.1 * frame.shape[1]:
                                twist.angular.z = lateral_speed_rectangle

                            if (area_ratio_rectangle > 0.4) and (
                                abs(error_x_rectangle) < 0.1 * frame.shape[1]
                            ):
                                twist.linear.x = 0.0
                                twist.angular.z = 0.0
                                reached_box = 1
                            cmd_vel_pub.publish(twist)
                            if reached_box:
                                robotic_arm_pub.publish(2)
                                reached_box = 0
                                done = 0
                                box = 0
                                break

        # Display the frame with the bounding box
        # cv2.imshow("YOLOv8 Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
