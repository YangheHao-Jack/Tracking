import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Function to find the largest connected component
def get_longest_component(binary_image):
    labels = label(binary_image)
    regions = regionprops(labels)
    if len(regions) == 0:
        return binary_image
    largest_region = max(regions, key=lambda region: region.area)
    mask = (labels == largest_region.label).astype(np.uint8) * 255
    return mask

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
        "Gaussian Blur Kernel Size": cv2.getTrackbarPos("Gaussian Blur Kernel Size", window_name) * 2 + 1,
        "Bilateral Filter Diameter": cv2.getTrackbarPos("Bilateral Filter Diameter", window_name),
        "Bilateral Sigma Color": cv2.getTrackbarPos("Bilateral Sigma Color", window_name),
        "Bilateral Sigma Space": cv2.getTrackbarPos("Bilateral Sigma Space", window_name),
        "Morph Gradient Kernel Size": cv2.getTrackbarPos("Morph Gradient Kernel Size", window_name),
        "Intensity Threshold": cv2.getTrackbarPos("Intensity Threshold", window_name),
        "Intensity Threshold Fine": cv2.getTrackbarPos("Intensity Threshold Fine", window_name) / 100.0,
        "LOF Neighbors": cv2.getTrackbarPos("LOF Neighbors", window_name),
        "LOF Contamination": cv2.getTrackbarPos("LOF Contamination", window_name) / 100.0,
        "Isolation Forest Contamination": cv2.getTrackbarPos("Isolation Forest Contamination", window_name) / 100.0,
        "Isolation Forest Estimators": cv2.getTrackbarPos("Isolation Forest Estimators", window_name),
        "SVM Gamma": cv2.getTrackbarPos("SVM Gamma", window_name) / 100.0,
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
cv2.createTrackbar("Gaussian Blur Kernel Size", trackbar_window, 2, 10, lambda x: None)
cv2.createTrackbar("Bilateral Filter Diameter", trackbar_window, 5, 15, lambda x: None)
cv2.createTrackbar("Bilateral Sigma Color", trackbar_window, 50, 150, lambda x: None)
cv2.createTrackbar("Bilateral Sigma Space", trackbar_window, 50, 150, lambda x: None)
cv2.createTrackbar("Morph Gradient Kernel Size", trackbar_window, 3, 10, lambda x: None)
cv2.createTrackbar("Intensity Threshold", trackbar_window, 0, 255, lambda x: None)
cv2.createTrackbar("Intensity Threshold Fine", trackbar_window, 0, 100, lambda x: None)
cv2.createTrackbar("LOF Neighbors", trackbar_window, 5, 20, lambda x: None)
cv2.createTrackbar("LOF Contamination", trackbar_window, 5, 50, lambda x: None)
cv2.createTrackbar("Isolation Forest Contamination", trackbar_window, 5, 50, lambda x: None)
cv2.createTrackbar("Isolation Forest Estimators", trackbar_window, 10, 100, lambda x: None)
cv2.createTrackbar("SVM Gamma", trackbar_window, 10, 100, lambda x: None)

# Video capture
video_path = 'Tom.mp4'
cap = cv2.VideoCapture(video_path)

# Kalman filter initialization
kalman = cv2.KalmanFilter(6, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0], [0, 1, 0, 1, 0, 0.5],
                                     [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1],
                                     [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)

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

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)  # Visualization 1: Grayscale image

    # CLAHE to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=settings["CLAHE Clip Limit"],
                            tileGridSize=(settings["CLAHE Grid Size"], settings["CLAHE Grid Size"]))
    enhanced_gray = clahe.apply(gray)
    cv2.imshow("CLAHE Enhanced", enhanced_gray)  # Visualization 2: CLAHE enhanced image

    # Bilateral filter for edge-preserving smoothing
    bilateral_filtered = cv2.bilateralFilter(enhanced_gray, d=settings["Bilateral Filter Diameter"],
                                             sigmaColor=settings["Bilateral Sigma Color"],
                                             sigmaSpace=settings["Bilateral Sigma Space"])
    cv2.imshow("Bilateral Filtered", bilateral_filtered)  # Visualization 3: Bilateral filtered image

    # Further enhance contrast using histogram equalization
    equalized_gray = cv2.equalizeHist(bilateral_filtered)
    cv2.imshow("Histogram Equalized", equalized_gray)  # Visualization 4: Histogram equalized image

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(equalized_gray, (settings["Gaussian Blur Kernel Size"], settings["Gaussian Blur Kernel Size"]), 0)
    cv2.imshow("Gaussian Blurred", blurred)  # Visualization 5: Gaussian blurred image

    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, settings["Threshold Block Size"], settings["Threshold C"])
    cv2.imshow("Adaptive Threshold", adaptive_thresh)  # Visualization 6: Adaptive thresholded image

    # Advanced thresholding based on intensity with finer adjustment and smoothing
    intensity_threshold_value = settings["Intensity Threshold"] + (settings["Intensity Threshold Fine"] * (255 - settings["Intensity Threshold"]))
    # Apply a bilateral filter before thresholding to reduce noise while preserving edges
    bilateral_pre_thresh = cv2.bilateralFilter(gray, d=settings["Bilateral Filter Diameter"],
                                               sigmaColor=settings["Bilateral Sigma Color"],
                                               sigmaSpace=settings["Bilateral Sigma Space"])
    _, intensity_thresh = cv2.threshold(bilateral_pre_thresh, int(intensity_threshold_value), 255, cv2.THRESH_BINARY)
    cv2.imshow("Intensity Threshold", intensity_thresh)  # Visualization 7: Intensity thresholded image

    # Use morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (settings["Morph Kernel Size"], settings["Morph Kernel Size"]))
    cleaned_mask = cv2.morphologyEx(intensity_thresh, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Cleaned Mask", cleaned_mask)  # Visualization 8: Cleaned mask

    # Use cleaned mask for tracking
    catheter_mask = get_longest_component(cleaned_mask)
    edges = cv2.Canny(catheter_mask, settings["Canny Low"], settings["Canny High"])
    # Dilation to strengthen edges before corner detection
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (settings["Morph Gradient Kernel Size"], settings["Morph Gradient Kernel Size"]))
    edges = cv2.dilate(edges, dilation_kernel)
    cv2.imshow("Edges", edges)  # Visualization 9: Canny edges

    # Find feature points with improved Shi-Tomasi corner detection
    feature_points = cv2.goodFeaturesToTrack(edges, maxCorners=settings["Max Corners"],
                                             qualityLevel=settings["Quality Level"], minDistance=settings["Min Distance"],
                                             useHarrisDetector=True, k=0.04)
    detected_tip = None
    extra_tracker = None

    if feature_points is not None:
        points = [tuple(point.ravel()) for point in feature_points]

        # Remove outliers using Local Outlier Factor (LOF)
        if len(points) > 2:
            lof = LocalOutlierFactor(n_neighbors=settings["LOF Neighbors"], contamination=settings["LOF Contamination"])
            points_array = np.array(points)
            is_inlier = lof.fit_predict(points_array)
            points = [point for point, inlier in zip(points, is_inlier) if inlier == 1]

        # Additional outlier filtering using Isolation Forest
        if len(points) > 2:
            iso_forest = IsolationForest(n_estimators=settings["Isolation Forest Estimators"],
                                         contamination=settings["Isolation Forest Contamination"],
                                         random_state=42)
            is_inlier = iso_forest.fit_predict(points_array)
            points = [point for point, inlier in zip(points, is_inlier) if inlier == 1]

        # Additional outlier filtering using One-Class SVM
        if len(points) > 2:
            scaler = StandardScaler()
            scaled_points = scaler.fit_transform(points_array)
            one_class_svm = OneClassSVM(kernel='rbf', gamma=settings["SVM Gamma"], nu=0.05)
            one_class_svm.fit(scaled_points)
            labels = one_class_svm.predict(scaled_points)
            points = [point for point, label in zip(points, labels) if label == 1]

        # Sort and select the leftmost point as detected tip
        if points:
            points = sorted(points, key=lambda p: p[0])
            detected_tip = points[0]
            extra_tracker = points[5] if len(points) > 5 else detected_tip

            for point in points:
                x, y = map(int, point)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

    if detected_tip is not None:
        kalman.correct(np.array([[np.float32(detected_tip[0])], [np.float32(detected_tip[1])]]))
        cv2.circle(frame, (int(detected_tip[0]), int(detected_tip[1])), 5, (255, 0, 0), -1)
        tip_position_text = f"Tip Position: {detected_tip}"
        all_tip_positions.append(detected_tip)

    if extra_tracker is not None:
        cv2.circle(frame, (int(extra_tracker[0]), int(extra_tracker[1])), 5, (0, 255, 0), -1)
        extra_tracker_text = f"Extra Tracker: {extra_tracker}"
        all_extra_tracker_positions.append(extra_tracker)

    if detected_tip:
        cv2.putText(frame, tip_position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if extra_tracker:
        cv2.putText(frame, extra_tracker_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Catheter Tip Tracking", frame)

    # Save current settings to a file when 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("Settings:", settings)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("All Tip Positions:")
for position in all_tip_positions:
    print(position)
