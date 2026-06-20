import os
import shutil
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO Model for Defect Detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (e.g. 320 for fast CPU training, 640 for standard)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained weights to start training (default: yolov8n.pt)")
    return parser.parse_args()

def train():
    args = parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Dataset configuration file '{args.data}' not found. Please generate dataset first.")
        return
        
    print(f"Loading base YOLO model: {args.weights}...")
    model = YOLO(args.weights)
    
    print(f"Starting training on '{args.data}' for {args.epochs} epochs...")
    # Training parameters
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,      # Set to 0 to avoid multi-threading issues on some Windows systems
        device="cpu"    # Force CPU for standard compatibility. Change to device=0 if you have a CUDA GPU
    )
    
    # Locate and copy the best weights to the root directory
    local_runs = os.path.join("runs", "detect")
    home_runs = os.path.join(os.path.expanduser("~"), "runs", "detect")
    
    detect_dir = None
    if os.path.exists(local_runs):
        detect_dir = local_runs
    elif os.path.exists(home_runs):
        detect_dir = home_runs
        
    best_weights_src = None
    if detect_dir:
        train_dirs = [d for d in os.listdir(detect_dir) if d.startswith("train")]
        train_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(detect_dir, d)))
        if train_dirs:
            best_weights_src = os.path.join(detect_dir, train_dirs[-1], "weights", "best.pt")

    if best_weights_src and os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, "best.pt")
        print(f"\nModel training complete! Copied weights from '{best_weights_src}' to './best.pt'")
    else:
        print(f"\nWarning: Could not automatically locate trained weights file. Please check: {detect_dir}")

if __name__ == "__main__":
    train()
