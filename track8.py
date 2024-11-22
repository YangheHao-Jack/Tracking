import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import distance

# Load the video
video_path = '/home/yanghehao/Downloads/Tom.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize a background subtractor with KNN
fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=800, detectShadows=False)

# Initialize Kalman filter
kalman = cv2.KalmanFilter(6, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0], [0, 1, 0, 1, 0, 0.5],
                                     [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1],
                                     [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)

# Initialize window and sliders
cv2.namedWindow("Catheter Tip Tracking")

# Add sliders for Kalman filter, Shi-Tomasi, thresholding, Canny, cropping, and contrast/brightness adjustments
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
cv2.createTrackbar("Contrast", "Catheter Tip Tracking", 50, 100, lambda x: None)  # Contrast control
cv2.createTrackbar("Brightness", "Catheter Tip Tracking", 50, 100, lambda x: None)  # Brightness control

# Initialize CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Tracking history
all_tip_positions = []
all_extra_tracker_positions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current trackbar values
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
    contrast = cv2.getTrackbarPos("Contrast", "Catheter Tip Tracking") / 50.0  # Normalized contrast multiplier
    brightness = cv2.getTrackbarPos("Brightness", "Catheter Tip Tracking") - 50  # Brightness offset (-50 to +50)

    # Update Kalman filter noise covariance matrices
    kalman.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

    # Crop the frame
    h, w, _ = frame.shape
    top = int(h * crop_top / 100)
    bottom = int(h * (100 - crop_bottom) / 100)
    left = int(w * crop_left / 100)
    right = int(w * (100 - crop_right) / 100)
    frame = frame[top:bottom, left:right]

    # Adjust contrast and brightness
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    # Preprocessing: CLAHE for contrast enhancement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_gray = clahe.apply(gray)

    # Background subtraction
    fgmask = fgbg.apply(frame)  # Background subtraction mask
    fgmask = cv2.medianBlur(fgmask, 3)  # Reduce noise in the mask

    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, adaptive_threshold_block_size, adaptive_threshold_C)

    # Morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    refined_mask = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    # Combine masks and skeletonize
    combined_mask = cv2.bitwise_and(fgmask, refined_mask)
    skeleton = skeletonize(combined_mask // 255).astype(np.uint8) * 255

    # Edge detection using Canny
    edges = cv2.Canny(skeleton, canny_low_threshold, canny_high_threshold)

    # Detect points on the catheter using Shi-Tomasi corner detection
    feature_points = cv2.goodFeaturesToTrack(edges, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    detected_tip = None
    extra_tracker = None

    if feature_points is not None:
        # Convert detected points to a list of tuples
        points = [tuple(point.ravel()) for point in feature_points]

        # Sort points by x-coordinate to prioritize the leftmost point
        sorted_points = sorted(points, key=lambda p: p[0])
        detected_tip = sorted_points[0]  # The leftmost point is the tip

        # Define extra tracker (fifth point or fallback to the tip)
        extra_tracker = sorted_points[5] if len(sorted_points) > 5 else detected_tip

        # Draw all feature points
        for point in points:
            cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 255), -1)

    # Fallback to Kalman prediction if no points are detected
    if detected_tip is None:
        predicted = kalman.predict()
        detected_tip = (int(predicted[0]), int(predicted[1]))

    # Kalman filter correction
    kalman.correct(np.array([[np.float32(detected_tip[0])], [np.float32(detected_tip[1])]]))

    # Draw the detected tip and extra tracker
    cv2.circle(frame, (int(detected_tip[0]), int(detected_tip[1])), 5, (255, 0, 0), -1)
    if extra_tracker:
        cv2.circle(frame, (int(extra_tracker[0]), int(extra_tracker[1])), 5, (0, 255, 0), -1)

    # Display results
    cv2.putText(frame, f"Tip: {detected_tip}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if extra_tracker:
        cv2.putText(frame, f"Extra Tracker: {extra_tracker}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Catheter Tip Tracking", frame)

    # Exit the loop on 'q' key press
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
