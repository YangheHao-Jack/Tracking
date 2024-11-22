import cv2
import numpy as np
from collections import deque

# Load the video
video_path = '/home/yanghehao/Downloads/Tom.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize Lucas-Kanade Optical Flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Variables to keep track of the previous tip position and the previous frame
prev_gray = None
prev_tip = None
all_tip_positions = []
tip_positions_history = deque(maxlen=5)  # Track last 5 positions for smoothing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Initialize tip position text at the start of each frame
    tip_position_text = "Tip Position: Not Detected"

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Apply adaptive threshold to enhance the catheter
    adaptive_thresh = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 9, 1)  # Adjusted block size and C for better contrast

    # Use morphology to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours on the thresholded image
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter to find the longest contour assuming it corresponds to the catheter
    detected_tip = None
    if contours:
        longest_contour = max(contours, key=cv2.contourArea)
        if len(longest_contour) > 1:
            # Extract the leftmost and rightmost endpoints of the longest contour
            leftmost = tuple(longest_contour[longest_contour[:, :, 0].argmin()][0])
            rightmost = tuple(longest_contour[longest_contour[:, :, 0].argmax()][0])

            # Use the leftmost or rightmost point as the initial tip position based on previous frame direction
            if prev_tip is None:
                # Initialize the tip position
                prev_tip = leftmost  # or rightmost based on requirements

            # Draw the longest contour and the endpoints
            cv2.drawContours(frame, [longest_contour], -1, (0, 255, 0), 1)
            cv2.circle(frame, leftmost, 5, (255, 0, 0), -1)  # Blue dot for the leftmost point
            cv2.circle(frame, rightmost, 5, (0, 0, 255), -1)  # Red dot for the rightmost point

            # Determine which point is closer to the previous tip position
            if prev_tip is not None:
                dist_left = np.sqrt((leftmost[0] - prev_tip[0]) ** 2 + (leftmost[1] - prev_tip[1]) ** 2)
                dist_right = np.sqrt((rightmost[0] - prev_tip[0]) ** 2 + (rightmost[1] - prev_tip[1]) ** 2)
                detected_tip = leftmost if dist_left < dist_right else rightmost
            else:
                detected_tip = leftmost

    # If we have a previous tip, track it using optical flow
    if prev_tip is not None:
        p0 = np.array([[prev_tip]], dtype=np.float32)
        if prev_gray is not None:
            # Calculate optical flow to find the new position of the tip
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

            # Update tip if the flow status is good and position change is reasonable
            if st[0][0] == 1:
                next_tip = (int(p1[0][0][0]), int(p1[0][0][1]))
                
                # Ensure the new position doesn't jump too far from the detected_tip
                if detected_tip is not None:
                    tip_dist = np.sqrt((next_tip[0] - detected_tip[0]) ** 2 + (next_tip[1] - detected_tip[1]) ** 2)
                    if tip_dist < 15:  # Tightened distance constraint to stabilize tracking
                        prev_tip = next_tip
                    else:
                        prev_tip = detected_tip  # Fallback to detected position if flow prediction is off
                else:
                    prev_tip = next_tip  # If no detected_tip, use optical flow prediction

                # Save the position in history for smoothing
                tip_positions_history.append(prev_tip)

                # Average the recent positions for smooth motion
                avg_x = int(np.mean([p[0] for p in tip_positions_history]))
                avg_y = int(np.mean([p[1] for p in tip_positions_history]))
                smoothed_tip = (avg_x, avg_y)

                all_tip_positions.append(smoothed_tip)
                tip_position_text = f"Tip Position: {smoothed_tip}"
                cv2.circle(frame, smoothed_tip, 5, (0, 255, 0), -1)  # Green dot for smoothed tip

        prev_gray = gray.copy()  # Update previous frame for the next iteration

    # Display the tip position on the frame
    cv2.putText(frame, tip_position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame with tracking and tip position
    cv2.imshow("Catheter Tip Tracking", frame)

    # Slow down playback for observation
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Display all saved tip positions at the end
print("All Tip Positions:")
for position in all_tip_positions:
    print(position)
