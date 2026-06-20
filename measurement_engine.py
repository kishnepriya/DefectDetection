import cv2
import numpy as np

def calculate_pixels_per_metric(ref_box, actual_w_cm=5.0, actual_h_cm=5.0):
    """
    Calculates the pixel-to-metric ratio (pixels per cm).
    ref_box is [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = ref_box
    w_px = x2 - x1
    h_px = y2 - y1
    
    # Calculate ratio for width and height separately and take the average
    ratio_w = w_px / actual_w_cm
    ratio_h = h_px / actual_h_cm
    
    pixels_per_metric = (ratio_w + ratio_h) / 2.0
    return pixels_per_metric

def detect_coin_reference(image, coin_diameter_cm=2.5):
    """
    Fallback: Detects circular objects (like a coin) using OpenCV Hough Circles.
    Standardizes image width to 800 pixels to ensure scale-invariance.
    Returns bounding box [x1, y1, x2, y2] and pixels_per_metric, or (None, None).
    """
    h_img, w_img = image.shape[:2]
    target_width = 800
    scale = target_width / w_img
    new_h = int(h_img * scale)
    resized = cv2.resize(image, (target_width, new_h))
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Run Hough Circles with robust parameters tuned for coins
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=100, 
        param1=50, 
        param2=80,  # Stricter threshold to avoid false positives on rough concrete
        minRadius=40, 
        maxRadius=130
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Take the most prominent circle (first one returned)
        cx, cy, r = circles[0, 0]
        
        # Scale circle properties back to original resolution
        orig_cx = int(cx / scale)
        orig_cy = int(cy / scale)
        orig_r = int(r / scale)
        
        # Bounding box coordinates for the circle
        x1 = int(orig_cx - orig_r)
        y1 = int(orig_cy - orig_r)
        x2 = int(orig_cx + orig_r)
        y2 = int(orig_cy + orig_r)
        
        diameter_px = orig_r * 2.0
        pixels_per_metric = diameter_px / coin_diameter_cm
        return [x1, y1, x2, y2], pixels_per_metric
        
    return None, None

def detect_cracks_opencv(image):
    """
    Classical CV fallback to detect cracks if YOLO finds nothing.
    Returns list of detections: [{'box': [x1, y1, x2, y2], 'conf': 0.95, 'class': 0}]
    """
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold for dark crack shadows (typical values < 65)
    _, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000: # filter out small noise
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)
        
        # A crack is typically vertical, tall, and occupies a significant part of the image
        if h > h_img * 0.15 and aspect_ratio > 1.2:
            detections.append({
                "box": [x, y, x + w, y + h],
                "conf": 0.95,  # mock confidence
                "class": 0,    # crack
                "is_cv_fallback": True
            })
            
    return detections

def measure_defect_opencv(image, bbox, pixels_per_metric):
    """
    Measures the defect using OpenCV contours inside the bounding box.
    bbox is [x1, y1, x2, y2]
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = [int(val) for val in bbox]
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)
    
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return None
        
    # Crop the defect region
    crop = image[y1:y2, x1:x2]
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to isolate the crack (adaptive or binary thresholding)
    # Since background concrete texture is lighter and cracks are darker,
    # we threshold out dark pixels.
    _, thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Fallback to YOLO bounding box if no contours found
        w_px = x2 - x1
        h_px = y2 - y1
        real_w = w_px / pixels_per_metric
        real_h = h_px / pixels_per_metric
        
        # Format box points for drawing
        box_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        return real_w, real_h, box_points

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get oriented bounding box for the contour
    rect = cv2.minAreaRect(largest_contour)
    box_points = cv2.boxPoints(rect)
    box_points = np.int32(box_points)
    
    # Adjust box points from crop coordinate space to original image space
    box_points[:, 0] += x1
    box_points[:, 1] += y1
    
    # Dimensions of oriented rect
    (cx, cy), (w_px, h_px), angle = rect
    
    real_w = w_px / pixels_per_metric
    real_h = h_px / pixels_per_metric
    
    # Standardize: make height (length) the larger dimension
    if real_w > real_h:
        real_w, real_h = real_h, real_w
        
    return real_w, real_h, box_points

def draw_annotations(image, detections, ref_box=None, actual_ref_w=5.0, actual_ref_h=5.0):
    """
    Annotates the image with measurements.
    detections list of dict: {
        'box': [x1, y1, x2, y2],
        'conf': confidence_score,
        'class': class_id (0 for crack, 1 for reference_object)
    }
    """
    annotated = image.copy()
    
    # 1. First, find reference object and calculate ratio
    pixels_per_metric = None
    ref_source = "coin"
    if ref_box is not None:
        pixels_per_metric = calculate_pixels_per_metric(ref_box, actual_ref_w, actual_ref_h)
    else:
        # Check if reference_object is in detections
        for det in detections:
            if det['class'] == 1: # reference_object
                pixels_per_metric = calculate_pixels_per_metric(det['box'], actual_ref_w, actual_ref_h)
                ref_box = det['box']
                break
        
        # Fallback: If YOLO did not detect the reference, try classical CV coin detection
        if pixels_per_metric is None:
            coin_box, coin_ppm = detect_coin_reference(image, coin_diameter_cm=2.5)
            if coin_box is not None:
                ref_box = coin_box
                pixels_per_metric = coin_ppm
                actual_ref_w, actual_ref_h = 2.5, 2.5
                ref_source = "coin (CV)"
                
    # 2. Draw reference object bounding box
    if ref_box is not None:
        x1, y1, x2, y2 = [int(v) for v in ref_box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(annotated, f"Ref: {actual_ref_w}x{actual_ref_h}cm ({ref_source})", 
                    (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 100, 0), 2)
                    
    # 3. Process and draw other detections (cracks)
    for det in detections:
        if det['class'] == 0: # crack
            bbox = det['box']
            conf = det['conf']
            
            if pixels_per_metric is not None:
                # Measure defect size using OpenCV contours
                result = measure_defect_opencv(image, bbox, pixels_per_metric)
                if result is not None:
                    real_w, real_h, box_points = result
                    
                    # Draw oriented bounding box (red)
                    cv2.drawContours(annotated, [box_points], 0, (0, 0, 255), 2)
                    
                    # Get center of bounding box to draw text
                    cx = int(np.mean(box_points[:, 0]))
                    cy = int(np.mean(box_points[:, 1]))
                    
                    # Draw dimensions text
                    text_w = f"W: {real_w:.2f} cm"
                    text_h = f"L: {real_h:.2f} cm"
                    text_conf = f"Crack ({conf:.2f})"
                    
                    cv2.putText(annotated, text_conf, (cx - 50, cy - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(annotated, text_w, (cx - 50, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(annotated, text_h, (cx - 50, cy + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Draw plain bounding box if no reference object was detected
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"Crack ({conf:.2f}) [No Ref]", 
                            (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 255), 2)
                            
    return annotated
