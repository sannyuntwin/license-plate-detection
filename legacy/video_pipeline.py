import cv2
import os
import pandas as pd
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from license_plate_detection.detector import PlateDetector
from license_plate_detection.ocr import PlateOCR

VIDEO_PATH = "data/videos/car.mp4"
OUTPUT_DIR = "output/plates"
CSV_PATH = "output/results.csv"

# Performance optimization settings
FRAME_SKIP = 3  # Process every 3rd frame instead of every frame
DISPLAY_SKIP = 5  # Update display every 5 processed frames
FAST_MODE = True  # Skip display for faster processing
SHOW_PROGRESS = True  # Show progress every N seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = PlateDetector()
ocr = PlateOCR()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: Could not open video file")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0
print(f"🎬 Video FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration:.1f}s")

frame_id = 0
processed_frame_id = 0
results_data = []
start_time = time.time()
last_progress_time = start_time

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    
    # Skip frames for faster processing
    if frame_id % FRAME_SKIP != 0:
        continue
    
    processed_frame_id += 1
    boxes = detector.detect(frame)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        plate_img = frame[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue

        plate_text = ocr.read_text(plate_img)

        # Save plate image
        plate_filename = f"plate_{frame_id}_{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, plate_filename), plate_img)

        if not FAST_MODE:
            # Draw box and text on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, plate_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        if plate_text:
            results_data.append({
                "frame": frame_id,
                "plate_text": plate_text,
                "image": plate_filename
            })

    # Display frame only every few processed frames for performance
    if not FAST_MODE and processed_frame_id % DISPLAY_SKIP == 0:
        cv2.imshow("License Plate Detection (Fast Mode)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Show progress every 5 seconds
    current_time = time.time()
    if SHOW_PROGRESS and current_time - last_progress_time >= 5:
        elapsed = current_time - start_time
        progress_pct = (frame_id / total_frames) * 100
        fps_current = processed_frame_id / elapsed if elapsed > 0 else 0
        
        print(f"⏱️  Progress: {progress_pct:.1f}% | "
              f"Processed: {processed_frame_id} frames | "
              f"FPS: {fps_current:.1f} | "
              f"Elapsed: {elapsed:.1f}s")
        
        last_progress_time = current_time

cap.release()

if not FAST_MODE:
    cv2.destroyAllWindows()

# Save CSV
df = pd.DataFrame(results_data)
df.to_csv(CSV_PATH, index=False)

end_time = time.time()
total_time = end_time - start_time
plates_found = len(results_data)

print("\n✅ Processing complete!")
print(f"📊 Statistics:")
print(f"   - Total processing time: {total_time:.1f}s")
print(f"   - Frames processed: {processed_frame_id}")
print(f"   - Average FPS: {processed_frame_id/total_time:.1f}")
print(f"   - License plates found: {plates_found}")
print(f"📄 Results saved to: {CSV_PATH}")

# Show some sample results
if len(results_data) > 0:
    print(f"\n🔍 Sample detections:")
    for i, result in enumerate(results_data[:5]):
        print(f"   Frame {result['frame']}: '{result['plate_text']}'")
    if len(results_data) > 5:
        print(f"   ... and {len(results_data) - 5} more")
