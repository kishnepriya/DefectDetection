import argparse
import cv2
import os
from ultralytics import YOLO
from measurement_engine import draw_annotations, detect_cracks_opencv

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Defect Detection & Size Measurement")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--source", type=str, default="dataset/images/val/crack_40.jpg", help="Path to image, video file, or webcam index (e.g. 0)")
    parser.add_argument("--output", type=str, default="results/output_result.jpg", help="Path to save output result")
    parser.add_argument("--ref-width", type=float, default=5.0, help="Real world width of reference object in cm")
    parser.add_argument("--ref-height", type=float, default=5.0, help="Real world height of reference object in cm")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for YOLO detections")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found. Please train the model first by running train_model.py.")
        return
        
    print(f"Loading YOLO model: {args.weights}")
    model = YOLO(args.weights)
    
    # Check if source is a webcam index
    is_webcam = False
    try:
        source_idx = int(args.source)
        is_webcam = True
    except ValueError:
        source_idx = args.source
        
    # Check if file exists when not webcam
    if not is_webcam and not os.path.exists(source_idx):
        print(f"Error: Source '{source_idx}' does not exist.")
        return
        
    # Process image
    if not is_webcam and source_idx.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        print(f"Running inference on image: {source_idx}")
        img = cv2.imread(source_idx)
        if img is None:
            print("Error: Failed to read image.")
            return
            
        results = model(img, conf=args.conf)[0]
        
        detections = []
        for box in results.boxes:
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append({
                "box": coords,
                "conf": conf,
                "class": cls
            })
            
        # Fallback: If YOLO did not detect any cracks (class 0), use classical CV crack detection
        has_yolo_crack = any(d["class"] == 0 for d in detections)
        if not has_yolo_crack:
            print("YOLO did not detect any cracks. Running classical CV crack detection fallback...")
            cv_detections = detect_cracks_opencv(img)
            detections.extend(cv_detections)
            
        annotated_img = draw_annotations(img, detections, actual_ref_w=args.ref_width, actual_ref_h=args.ref_height)
        
        # Save output
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(args.output, annotated_img)
        print(f"Successfully processed image. Saved result to '{args.output}'")
        
    # Process video/webcam
    else:
        print(f"Running inference on video source: {source_idx}")
        cap = cv2.VideoCapture(source_idx)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
            
        # Prepare video writer if output path is set and input is a video file
        writer = None
        if not is_webcam and args.output:
            # Change output file extension to mp4 if needed
            out_path = args.output
            if out_path.endswith((".jpg", ".png", ".jpeg")):
                out_path = os.path.splitext(out_path)[0] + ".mp4"
            
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            print(f"Saving output video to: {out_path}")
            
        print("Processing frames. Press 'q' to quit...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, conf=args.conf)[0]
            detections = []
            for box in results.boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    "box": coords,
                    "conf": conf,
                    "class": cls
                })
                
            # Fallback: If YOLO did not detect any cracks (class 0), use classical CV crack detection
            has_yolo_crack = any(d["class"] == 0 for d in detections)
            if not has_yolo_crack:
                cv_detections = detect_cracks_opencv(frame)
                detections.extend(cv_detections)
                
            annotated_frame = draw_annotations(frame, detections, actual_ref_w=args.ref_width, actual_ref_h=args.ref_height)
            
            if writer:
                writer.write(annotated_frame)
                
            # Try to show window
            try:
                cv2.imshow("YOLO Defect Detection & Size Measurement", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                # Running in headless environment (no GUI window support)
                pass
                
        cap.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
            
        print("Finished processing stream.")

if __name__ == "__main__":
    main()
