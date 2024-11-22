import cv2
import numpy as np
from skimage.morphology import skeletonize

# Load the video
video_path = '/home/yanghehao/Downloads/Tom.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize a background subtractor with KNN
fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=800, detectShadows=False)

# Initialize Kalman filter
kalman = cv2.KalmanFilter(6, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0], [0, 1, 0, 1, 0, 0.5], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)

# Initialize window and sliders
cv2.namedWindow("Catheter Tip Tracking")

# Tracking Parameters
cv2.createTrackbar("Process Noise", "Catheter Tip Tracking", 2, 100, lambda x: None)
cv2.createTrackbar("Measurement Noise", "Catheter Tip Tracking", 20, 100, lambda x: None)
cv2.createTrackbar("Max Corners", "Catheter Tip Tracking", 50, 500, lambda x: None)
cv2.createTrackbar("Quality Level", "Catheter Tip Tracking", 10, 100, lambda x: None)
cv2.createTrackbar("Min Distance", "Catheter Tip Tracking", 10, 50, lambda x: None)
cv2.createTrackbar("Threshold Block Size", "Catheter Tip Tracking", 11, 50, lambda x: None)
cv2.createTrackbar("Threshold C", "Catheter Tip Tracking", 2, 10, lambda x: None)
cv2.createTrackbar("Canny Low", "Catheter Tip Tracking", 180, 255, lambda x: None)
cv2.createTrackbar("Canny High", "Catheter Tip Tracking", 190, 255, lambda x: None)
cv2.createTrackbar("Crop Top", "Catheter Tip Tracking", 0, 100, lambda x: None)
cv2.createTrackbar("Crop Bottom", "Catheter Tip Tracking", 10, 100, lambda x: None)
cv2.createTrackbar("Crop Left", "Catheter Tip Tracking", 10, 100, lambda x: None)
cv2.createTrackbar("Crop Right", "Catheter Tip Tracking", 0, 100, lambda x: None)

# Tracking history
all_tip_positions = []
all_extra_tracker_positions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current slider values
    process_noise = cv2.getTrackbarPos("Process Noise", "Catheter Tip Tracking") / 1000.0
    measurement_noise = cv2.getTrackbarPos("Measurement Noise", "Catheter Tip Tracking") / 1000.0
    max_corners = cv2.getTrackbarPos("Max Corners", "Catheter Tip Tracking")
    quality_level = cv2.getTrackbarPos("Quality Level", "Catheter Tip Tracking") / 100.0
    min_distance = cv2.getTrackbarPos("Min Distance", "Catheter Tip Tracking")
    adaptive_threshold_block_size = cv2.getTrackbarPos("Threshold Block Size", "Catheter Tip Tracking") * 2 + 1
    adaptive_threshold_C = cv2.getTrackbarPos("Threshold C", "Catheter Tip Tracking")
    canny_low_threshold = cv2.getTrackbarPos("Canny Low", "Catheter Tip Tracking")
    canny_high_threshold = cv2.getTrackbarPos("Canny High", "Catheter Tip Tracking")
    crop_top = cv2.getTrackbarPos("Crop Top", "Catheter Tip Tracking")
    crop_bottom = cv2.getTrackbarPos("Crop Bottom", "Catheter Tip Tracking")
    crop_left = cv2.getTrackbarPos("Crop Left", "Catheter Tip Tracking")
    crop_right = cv2.getTrackbarPos("Crop Right", "Catheter Tip Tracking")

    # Update Kalman filter noise covariance matrices
    kalman.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

    # Determine cropping boundaries
    h, w, _ = frame.shape
    top = int(h * crop_top / 100)
    bottom = int(h * (100 - crop_bottom) / 100)
    left = int(w * crop_left / 100)
    right = int(w * (100 - crop_right) / 100)
    frame = frame[top:bottom, left:right]

    # Background subtraction and noise removal
    fgmask = fgbg.apply(frame)
    fgmask = cv2.medianBlur(fgmask, 3)

    # Adaptive thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, adaptive_threshold_block_size, adaptive_threshold_C)

    # Combine masks and skeletonize
    combined_mask = cv2.bitwise_and(fgmask, adaptive_thresh)
    skeleton = skeletonize(combined_mask // 255).astype(np.uint8) * 255

    # Edge detection using Canny
    edges = cv2.Canny(skeleton, canny_low_threshold, canny_high_threshold)

    # Detect points on the catheter using Shi-Tomasi corner detection
    feature_points = cv2.goodFeaturesToTrack(edges, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)

    detected_tip = None
    extra_tracker = None

    if feature_points is None:
        print("No feature points detected.")
        predicted = kalman.predict()
        detected_tip = (int(predicted[0]), int(predicted[1]))
    else:
        points = sorted([tuple(point.ravel()) for point in feature_points], key=lambda p: p[0])  # Sort by x-coordinate
        detected_tip = points[0]  # Use leftmost point as the catheter tip

        # Determine the fifth corner before the tip
        if len(points) >= 6:  # Ensure at least 6 points are available
            extra_tracker = points[5]
        else:
            extra_tracker = detected_tip  # Fallback to the tip itself if not enough points

        # Draw all feature points
        for point in points:
            cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 255), -1)

    # Kalman filter correction or prediction
    if detected_tip is not None:
        kalman.correct(np.array([[np.float32(detected_tip[0])], [np.float32(detected_tip[1])]]))
        cv2.circle(frame, (int(detected_tip[0]), int(detected_tip[1])), 5, (255, 0, 0), -1)
        tip_position_text = f"Tip Position: {detected_tip}"
        all_tip_positions.append(detected_tip)

    if extra_tracker is not None:
        cv2.circle(frame, (int(extra_tracker[0]), int(extra_tracker[1])), 5, (0, 255, 0), -1)
        extra_tracker_text = f"Extra Tracker: {extra_tracker}"
        all_extra_tracker_positions.append(extra_tracker)

    # Display the tip and extra tracker positions on the frame
    cv2.putText(frame, tip_position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if extra_tracker is not None:
        cv2.putText(frame, extra_tracker_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Catheter Tip Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("All Tip Positions:")
for position in all_tip_positions:
    print(position)

print("All Extra Tracker Positions:")
for position in all_extra_tracker_positions:
    print(position)
