import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# Function to find the largest connected component
def get_longest_component(binary_image):
    labels = label(binary_image)
    regions = regionprops(labels)
    if len(regions) == 0:
        return binary_image
    largest_region = max(regions, key=lambda region: region.area)
    mask = (labels == largest_region.label).astype(np.uint8) * 255
    return mask

# Load the video
video_path = '/home/yanghehao/Downloads/Tom.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize Kalman filter
kalman = cv2.KalmanFilter(6, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0], [0, 1, 0, 1, 0, 0.5],
                                     [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1],
                                     [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)

# Create a window for parameter adjustment
cv2.namedWindow("Catheter Tip Tracking")

# Add sliders for all adjustable parameters
cv2.createTrackbar("Crop Top", "Catheter Tip Tracking", 0, 100, lambda x: None)
cv2.createTrackbar("Crop Bottom", "Catheter Tip Tracking", 100, 100, lambda x: None)
cv2.createTrackbar("Crop Left", "Catheter Tip Tracking", 0, 100, lambda x: None)
cv2.createTrackbar("Crop Right", "Catheter Tip Tracking", 100, 100, lambda x: None)
cv2.createTrackbar("Process Noise", "Catheter Tip Tracking", 10, 100, lambda x: None)
cv2.createTrackbar("Measurement Noise", "Catheter Tip Tracking", 10, 100, lambda x: None)
cv2.createTrackbar("Canny Low", "Catheter Tip Tracking", 50, 255, lambda x: None)
cv2.createTrackbar("Canny High", "Catheter Tip Tracking", 150, 255, lambda x: None)
cv2.createTrackbar("Threshold Block Size", "Catheter Tip Tracking", 15, 50, lambda x: None)
cv2.createTrackbar("Threshold C", "Catheter Tip Tracking", 2, 10, lambda x: None)
cv2.createTrackbar("Max Corners", "Catheter Tip Tracking", 50, 500, lambda x: None)
cv2.createTrackbar("Quality Level", "Catheter Tip Tracking", 10, 100, lambda x: None)
cv2.createTrackbar("Min Distance", "Catheter Tip Tracking", 10, 50, lambda x: None)
cv2.createTrackbar("Morph Kernel Size", "Catheter Tip Tracking", 3, 10, lambda x: None)
cv2.createTrackbar("CLAHE Clip Limit", "Catheter Tip Tracking", 20, 100, lambda x: None)
cv2.createTrackbar("CLAHE Grid Size", "Catheter Tip Tracking", 8, 50, lambda x: None)

# Storage for tracking positions
all_tip_positions = []
all_extra_tracker_positions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current trackbar values
    crop_top = cv2.getTrackbarPos("Crop Top", "Catheter Tip Tracking")
    crop_bottom = cv2.getTrackbarPos("Crop Bottom", "Catheter Tip Tracking")
    crop_left = cv2.getTrackbarPos("Crop Left", "Catheter Tip Tracking")
    crop_right = cv2.getTrackbarPos("Crop Right", "Catheter Tip Tracking")
    process_noise = cv2.getTrackbarPos("Process Noise", "Catheter Tip Tracking") / 1000.0
    measurement_noise = cv2.getTrackbarPos("Measurement Noise", "Catheter Tip Tracking") / 1000.0
    canny_low_threshold = cv2.getTrackbarPos("Canny Low", "Catheter Tip Tracking")
    canny_high_threshold = cv2.getTrackbarPos("Canny High", "Catheter Tip Tracking")
    threshold_block_size = cv2.getTrackbarPos("Threshold Block Size", "Catheter Tip Tracking") * 2 + 1
    threshold_C = cv2.getTrackbarPos("Threshold C", "Catheter Tip Tracking")
    max_corners = cv2.getTrackbarPos("Max Corners", "Catheter Tip Tracking")
    quality_level = cv2.getTrackbarPos("Quality Level", "Catheter Tip Tracking") / 100.0
    min_distance = cv2.getTrackbarPos("Min Distance", "Catheter Tip Tracking")
    morph_kernel_size = cv2.getTrackbarPos("Morph Kernel Size", "Catheter Tip Tracking")
    clahe_clip_limit = cv2.getTrackbarPos("CLAHE Clip Limit", "Catheter Tip Tracking") / 10.0
    clahe_grid_size = cv2.getTrackbarPos("CLAHE Grid Size", "Catheter Tip Tracking")

    # Update Kalman filter noise covariance matrices
    kalman.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

    # Crop the frame dynamically
    h, w, _ = frame.shape
    top = int(h * crop_top / 100)
    bottom = int(h * crop_bottom / 100)
    left = int(w * crop_left / 100)
    right = int(w * crop_right / 100)
    frame = frame[top:bottom, left:right]

    # Preprocessing for enhanced feature detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_grid_size, clahe_grid_size))
    enhanced_gray = clahe.apply(gray)

    # Adaptive thresholding and cleaning
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, threshold_block_size, threshold_C)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    refined_mask = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    # Skeletonization and longest component isolation
    skeleton = skeletonize(refined_mask // 255).astype(np.uint8) * 255
    catheter_mask = get_longest_component(skeleton)

    # Canny edge detection
    edges = cv2.Canny(catheter_mask, canny_low_threshold, canny_high_threshold)

    # Detect catheter tip using Shi-Tomasi corner detection
    feature_points = cv2.goodFeaturesToTrack(edges, maxCorners=max_corners, qualityLevel=quality_level,
                                             minDistance=min_distance)
    detected_tip = None
    extra_tracker = None

    if feature_points is not None:
        points = sorted([tuple(point.ravel()) for point in feature_points], key=lambda p: p[0])  # Sort by x-coordinate
        detected_tip = points[0]  # Assume the leftmost point is the tip
        extra_tracker = points[5] if len(points) > 5 else detected_tip  # Fallback to tip if fewer points

        # Draw detected points
        for point in points:
            x, y = map(int, point)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

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

    # Display position text
    if detected_tip:
        cv2.putText(frame, tip_position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if extra_tracker:
        cv2.putText(frame, extra_tracker_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Catheter Tip Tracking", frame)

    # Exit on 'q' key
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
