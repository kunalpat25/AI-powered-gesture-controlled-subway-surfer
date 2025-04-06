import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller, Key
import time

# Initialize MediaPipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize Keyboard Controller
keyboard = Controller()

# Function to get FPS
def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    return fps, prev_time

# Main loop
cap = cv2.VideoCapture(0)
prev_time = 0
last_action_time = 0
action_cooldown = 0.5  # Cooldown time between actions (in seconds)

# Add a variable to store the last action
last_action = None
action_duration = 0.5  # Duration to maintain the last action (in seconds)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Draw quadrant lines
    cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (0, 255, 0), 2)  # Vertical line
    cv2.line(frame, (0, frame_height // 2), (frame_width, frame_height // 2), (0, 255, 0), 2)  # Horizontal line

    # Process pose detection
    results = pose.process(rgb_frame)
    action = None

    if results.pose_landmarks:
        # Extract key landmarks
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        # Convert normalized coordinates to pixel coordinates
        nose_x, nose_y = int(nose.x * frame_width), int(nose.y * frame_height)

        # Draw landmarks
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Determine action based on nose position
        mid_x, mid_y = frame_width // 2, frame_height // 2
        if nose_x < mid_x - 50:  # Move left
            action = "LEFT"
        elif nose_x > mid_x + 50:  # Move right
            action = "RIGHT"
        elif nose_y < mid_y - 50:  # Move up
            action = "UP"
        elif nose_y > mid_y + 50:  # Move down
            action = "DOWN"
        else:
            action = last_action  # Maintain the last action if no movement is detected

    # Handle actions and simulate key presses
    current_time = time.time()
    if action and action != last_action and current_time - last_action_time > action_cooldown:
        print(f"Action performed: {action}")  # Print the action in the terminal
        if action == "LEFT":
            keyboard.press(Key.left)
            keyboard.release(Key.left)
        elif action == "RIGHT":
            keyboard.press(Key.right)
            keyboard.release(Key.right)
        elif action == "UP":
            keyboard.press(Key.up)
            keyboard.release(Key.up)
        elif action == "DOWN":
            keyboard.press(Key.down)
            keyboard.release(Key.down)

        last_action_time = current_time
        last_action = action  # Update the last action

    # Maintain the last action for a short duration
    if current_time - last_action_time <= action_duration:
        display_action = last_action
    else:
        display_action = "None"

    # Update the displayed action
    cv2.putText(
        frame,
        f"Action: {display_action}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Subway Surfers with Pose Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()