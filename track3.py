import cv2
import numpy as np
from skimage.morphology import skeletonize

# Load the video
video_path = '/home/yanghehao/Downloads/Tom.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize a background subtractor with KNN, adjusted parameters
fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=800, detectShadows=False)

# Initialize Kalman filter with optimized parameters
kalman = cv2.KalmanFilter(6, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0], [0, 1, 0, 1, 0, 0.5], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.002  # Lower process noise for smoother tracking
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.02  # Adjust measurement noise

# Parameters for Shi-Tomasi corner detection
max_corners = 250
quality_level = 0.5 # Higher quality level for stronger features
min_distance = 20

# Parameters for adaptive thresholding and Canny edge detection
adaptive_threshold_block_size = 11
adaptive_threshold_C = 2
canny_low_threshold = 50
canny_high_threshold = 70

# Initialize tracking variables
previous_tip = None
reset_threshold = 30  # Threshold distance to reset Kalman if tracking diverges
all_tip_positions = []  # List to store all tip positions

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Apply median blur to reduce noise
    fgmask = cv2.medianBlur(fgmask, 3)

    # Apply adaptive thresholding for better edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, adaptive_threshold_block_size, adaptive_threshold_C)

    # Combine background subtraction mask with adaptive thresholding
    combined_mask = cv2.bitwise_and(fgmask, adaptive_thresh)

    # Skeletonize the mask to thin the catheter structure
    skeleton = skeletonize(combined_mask // 255).astype(np.uint8) * 255

    # Edge detection using Canny on the skeletonized mask
    edges = cv2.Canny(skeleton, canny_low_threshold, canny_high_threshold)

    # Detect points on the catheter using Shi-Tomasi corner detection
    feature_points = cv2.goodFeaturesToTrack(edges, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    detected_tip = None
    max_distance_to_previous = 0

    # Tip detection logic with direction filtering
    if feature_points is not None:
        for point in feature_points:
            x, y = point.ravel()
            # Filter out points close to the edges
            if x < 10 or y < 10 or x > frame.shape[1] - 10 or y > frame.shape[0] - 10:
                continue
            
            # Identify the farthest point from the previous tip for consistency
            if previous_tip is not None:
                dist_to_previous = np.sqrt((x - previous_tip[0]) ** 2 + (y - previous_tip[1]) ** 2)
                if dist_to_previous > max_distance_to_previous:
                    detected_tip = (int(x), int(y))
                    max_distance_to_previous = dist_to_previous
            else:
                detected_tip = (int(x), int(y))  # Set detected_tip for the first frame

    # Kalman filter correction if a new detection is found
    if detected_tip is not None:
        # Correct Kalman with detected position
        kalman.correct(np.array([[np.float32(detected_tip[0])], [np.float32(detected_tip[1])]]))
        
        # Reset Kalman filter if the new detected tip is far from the predicted location
        if previous_tip is not None:
            dist_to_predicted = np.sqrt((detected_tip[0] - previous_tip[0]) ** 2 + (detected_tip[1] - previous_tip[1]) ** 2)
            if dist_to_predicted > reset_threshold:
                kalman.statePre[:2] = np.array([[np.float32(detected_tip[0])], [np.float32(detected_tip[1])]])
                kalman.statePost[:2] = np.array([[np.float32(detected_tip[0])], [np.float32(detected_tip[1])]])

        previous_tip = detected_tip
        # Draw the detected tip position
        cv2.circle(frame, detected_tip, 5, (255, 0, 0), -1)  # Blue dot for detected tip
        tip_position_text = f"Tip Position: {detected_tip}"  # Display text for detected tip
        all_tip_positions.append(detected_tip)  # Save detected tip position
    else:
        # Predict the next position using Kalman filter if no tip is detected
        predicted = kalman.predict()
        predicted_tip = (int(predicted[0]), int(predicted[1]))
        previous_tip = predicted_tip

        # Draw the predicted tip position
        cv2.circle(frame, predicted_tip, 5, (0, 0, 255), -1)  # Red dot for predicted tip
        tip_position_text = f"Tip Position (Predicted): {predicted_tip}"  # Display text for predicted tip
        all_tip_positions.append(predicted_tip)  # Save predicted tip position

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
