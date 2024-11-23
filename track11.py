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

# Function to apply enhanced filtering for background suppression
def filter_catheter(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
    
    # Thresholding (combination of adaptive and Otsu)
    _, otsu_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    
    # Combine both thresholds for a better binary mask
    combined_thresh = cv2.bitwise_and(otsu_thresh, adaptive_thresh)
    
    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
    
    # Skeletonization for the catheter
    skeleton = skeletonize(refined_mask // 255).astype(np.uint8) * 255
    
    # Extract the largest connected component
    catheter_mask = get_longest_component(skeleton)
    
    return enhanced_gray, refined_mask, catheter_mask

# Load the video
video_path = 'Tom.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply filtering to isolate catheter
    enhanced_gray, refined_mask, catheter_mask = filter_catheter(frame)
    
    # Overlay the catheter skeleton on the original frame
    overlay = frame.copy()
    overlay[catheter_mask > 0] = (0, 255, 0)  # Highlight the catheter in green
    
    # Display results at different stages
    cv2.imshow("Original", frame)
    cv2.imshow("Enhanced Gray", enhanced_gray)
    cv2.imshow("Refined Mask", refined_mask)
    cv2.imshow("Catheter Skeleton", catheter_mask)
    cv2.imshow("Overlay", overlay)
    
    # Exit on 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
