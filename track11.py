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

# Function to create the improved mask
def create_optimized_mask(frame, settings):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=settings["CLAHE Clip Limit"],
                            tileGridSize=(settings["CLAHE Grid Size"], settings["CLAHE Grid Size"]))
    enhanced_gray = clahe.apply(gray)

    # Noise reduction using Gaussian blur
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

    # Thresholding: Combine adaptive and Otsu's thresholding
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, settings["Threshold Block Size"], settings["Threshold C"])
    combined_thresh = cv2.bitwise_and(otsu_thresh, adaptive_thresh)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (settings["Morph Kernel Size"], settings["Morph Kernel Size"]))
    refined_mask = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    # Extract the largest connected component
    largest_component = get_longest_component(refined_mask)

    # Skeletonize the result
    skeleton = skeletonize(largest_component // 255).astype(np.uint8) * 255

    return skeleton

# Function to retrieve trackbar positions
def get_trackbar_settings(window_name):
    settings = {
        "Crop Top": cv2.getTrackbarPos("Crop Top", window_name),
        "Crop Bottom": cv2.getTrackbarPos("Crop Bottom", window_name),
        "Crop Left": cv2.getTrackbarPos("Crop Left", window_name),
        "Crop Right": cv2.getTrackbarPos("Crop Right", window_name),
        "Process Noise": cv2.getTrackbarPos("Process Noise", window_name) / 1000.0,
        "Measurement Noise": cv2.getTrackbarPos("Measurement Noise", window_name) / 1000.0,
        "Canny Low": cv2.getTrackbarPos("Canny Low", window_name),
        "Canny High": cv2.getTrackbarPos("Canny High", window_name),
        "Threshold Block Size": cv2.getTrackbarPos("Threshold Block Size", window_name) * 2 + 1,
        "Threshold C": cv2.getTrackbarPos("Threshold C", window_name),
        "Max Corners": cv2.getTrackbarPos("Max Corners", window_name),
        "Quality Level": cv2.getTrackbarPos("Quality Level", window_name) / 100.0,
        "Min Distance": cv2.getTrackbarPos("Min Distance", window_name),
        "Morph Kernel Size": cv2.getTrackbarPos("Morph Kernel Size", window_name),
        "CLAHE Clip Limit": cv2.getTrackbarPos("CLAHE Clip Limit", window_name) / 10.0,
        "CLAHE Grid Size": cv2.getTrackbarPos("CLAHE Grid Size", window_name),
    }
    return settings

# Initialize settings window
trackbar_window = "Settings"
cv2.namedWindow(trackbar_window)

# Add sliders for adjustable parameters
cv2.createTrackbar("Crop Top", trackbar_window, 0, 100, lambda x: None)
cv2.createTrackbar("Crop Bottom", trackbar_window, 100, 100, lambda x: None)
cv2.createTrackbar("Crop Left", trackbar_window, 0, 100, lambda x: None)
cv2.createTrackbar("Crop Right", trackbar_window, 100, 100, lambda x: None)
cv2.createTrackbar("Process Noise", trackbar_window, 10, 100, lambda x: None)
cv2.createTrackbar("Measurement Noise", trackbar_window, 10, 100, lambda x: None)
cv2.createTrackbar("Canny Low", trackbar_window, 50, 255, lambda x: None)
cv2.createTrackbar("Canny High", trackbar_window, 150, 255, lambda x: None)
cv2.createTrackbar("Threshold Block Size", trackbar_window, 15, 50, lambda x: None)
cv2.createTrackbar("Threshold C", trackbar_window, 2, 10, lambda x: None)
cv2.createTrackbar("Max Corners", trackbar_window, 50, 500, lambda x: None)
cv2.createTrackbar("Quality Level", trackbar_window, 10, 100, lambda x: None)
cv2.createTrackbar("Min Distance", trackbar_window, 10, 50, lambda x: None)
cv2.createTrackbar("Morph Kernel Size", trackbar_window, 3, 10, lambda x: None)
cv2.createTrackbar("CLAHE Clip Limit", trackbar_window, 20, 100, lambda x: None)
cv2.createTrackbar("CLAHE Grid Size", trackbar_window, 8, 50, lambda x: None)

# Kalman filter initialization
kalman = cv2.KalmanFilter(6, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0], [0, 1, 0, 1, 0, 0.5],
                                     [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1],
                                     [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)

# Video capture
video_path = 'Tom.mp4'
cap = cv2.VideoCapture(video_path)

# Storage for tracking positions
all_tip_positions = []
all_extra_tracker_positions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current settings
    settings = get_trackbar_settings(trackbar_window)

    # Update Kalman filter noise covariance matrices
    kalman.processNoiseCov = np.eye(6, dtype=np.float32) * settings["Process Noise"]
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * settings["Measurement Noise"]

    # Crop the frame dynamically
    h, w, _ = frame.shape
    top = int(h * settings["Crop Top"] / 100)
    bottom = int(h * settings["Crop Bottom"] / 100)
    left = int(w * settings["Crop Left"] / 100)
    right = int(w * settings["Crop Right"] / 100)
    frame = frame[top:bottom, left:right]

    # Apply the new masking process
    skeleton = create_optimized_mask(frame, settings)

    # Edge detection
    edges = cv2.Canny(skeleton, settings["Canny Low"], settings["Canny High"])
    cv2.imshow("Edges", edges)  # Visualization: Edge-detected image

    # Detect catheter tip using Shi-Tomasi corner detection
    feature_points = cv2.goodFeaturesToTrack(edges, maxCorners=settings["Max Corners"],
                                             qualityLevel=settings["Quality Level"], minDistance=settings["Min Distance"])
    detected_tip = None
    if feature_points is not None:
        points = sorted([tuple(point.ravel()) for point in feature_points], key=lambda p: p[0])
        detected_tip = points[0]

        for point in points:
            x, y = map(int, point)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

    # Kalman filter correction or prediction
    if detected_tip is not None:
        kalman.correct(np.array([[np.float32(detected_tip[0])], [np.float32(detected_tip[1])]]))
        cv2.circle(frame, (int(detected_tip[0]), int(detected_tip[1])), 5, (255, 0, 0), -1)
        tip_position_text = f"Tip Position: {detected_tip}"
        all_tip_positions.append(detected_tip)

    # Display results
    cv2.imshow("Catheter Tip Tracking", frame)

    # Exit on 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("All Tip Positions:")
for position in all_tip_positions:
    print(position)
