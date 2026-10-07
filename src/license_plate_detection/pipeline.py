import argparse
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Detect and read license plates in a video.")
    parser.add_argument("--video", type=Path, default=Path("data/videos/car.mp4"))
    parser.add_argument("--model", type=Path, default=Path("models/yolov8n.pt"))
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--class-id", type=int, help="Plate class ID for models with generic class names")
    parser.add_argument("--output-dir", type=Path, default=Path("output/plates"))
    parser.add_argument("--csv", type=Path, default=Path("output/results.csv"))
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--display-every", type=int, default=3)
    parser.add_argument("--no-display", action="store_true", help="Run without opening a video window")
    args = parser.parse_args()

    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1")
    if args.frame_skip < 1 or args.display_every < 1:
        parser.error("--frame-skip and --display-every must be at least 1")
    return args


def box_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0


def main():
    args = parse_args()

    import cv2
    import pandas as pd

    from .detector import PlateDetector
    from .ocr import PlateOCR

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.model.is_file():
        raise FileNotFoundError(f"YOLO model not found: {args.model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    detector = PlateDetector(args.model, args.confidence, args.class_id)
    ocr = PlateOCR()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"Video FPS: {fps:.2f}, frames: {total_frames}, duration: {duration:.1f}s")

    frame_id = 0
    processed_frames = 0
    tracks = []
    max_track_gap = max(args.frame_skip * 5, 10)
    start_time = time.time()
    last_progress_time = start_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % args.frame_skip != 0:
                continue

            processed_frames += 1
            boxes = detector.detect(frame)
            current_plates = 0
            matched_tracks = set()

            for index, (x1, y1, x2, y2) in enumerate(boxes):
                box = (x1, y1, x2, y2)
                best_track_index = None
                best_iou = 0.25
                for track_index, track in enumerate(tracks):
                    if track_index in matched_tracks or frame_id - track["last_frame"] > max_track_gap:
                        continue
                    overlap = box_iou(box, track["box"])
                    if overlap > best_iou:
                        best_iou = overlap
                        best_track_index = track_index

                if best_track_index is None:
                    track = {
                        "first_frame": frame_id,
                        "last_frame": frame_id,
                        "box": box,
                        "texts": {},
                        "image": None,
                    }
                    tracks.append(track)
                    best_track_index = len(tracks) - 1
                else:
                    track = tracks[best_track_index]
                    track["last_frame"] = frame_id
                    track["box"] = box
                matched_tracks.add(best_track_index)

                plate_img = frame[y1:y2, x1:x2]
                if plate_img.size == 0:
                    continue

                plate_text = ocr.read_text(plate_img)
                plate_filename = f"plate_{frame_id}_{index}.jpg"
                if not cv2.imwrite(str(args.output_dir / plate_filename), plate_img):
                    raise OSError(f"Could not save plate image: {args.output_dir / plate_filename}")
                if track["image"] is None:
                    track["image"] = plate_filename

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if plate_text:
                    normalized_text = "".join(character for character in plate_text.upper() if character.isalnum())
                    if normalized_text:
                        current_plates += 1
                        track["texts"][normalized_text] = track["texts"].get(normalized_text, 0) + 1

            if not args.no_display and processed_frames % args.display_every == 0:
                elapsed = max(time.time() - start_time, 0.001)
                info_text = (
                    f"Frame: {frame_id}/{total_frames} | Plates: {current_plates} | "
                    f"FPS: {processed_frames / elapsed:.1f}"
                )
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
                cv2.imshow("License Plate Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    cv2.waitKey(0)

            current_time = time.time()
            if current_time - last_progress_time >= 5:
                elapsed = current_time - start_time
                progress = frame_id / total_frames * 100 if total_frames else 0
                print(
                    f"Progress: {progress:.1f}% | processed {processed_frames} frames | "
                    f"{processed_frames / elapsed:.1f} FPS | "
                    f"{sum(bool(track['texts']) for track in tracks)} tracked plates"
                )
                last_progress_time = current_time
    finally:
        cap.release()
        cv2.destroyAllWindows()

    results_data = []
    for track in tracks:
        if not track["texts"]:
            continue
        plate_text = max(track["texts"], key=track["texts"].get)
        results_data.append({
            "first_frame": track["first_frame"],
            "last_frame": track["last_frame"],
            "plate_text": plate_text,
            "image": track["image"],
        })

    pd.DataFrame(
        results_data,
        columns=["first_frame", "last_frame", "plate_text", "image"],
    ).to_csv(args.csv, index=False)
    elapsed = max(time.time() - start_time, 0.001)
    print("Processing complete")
    print(f"Processed frames: {processed_frames}")
    print(f"Average FPS: {processed_frames / elapsed:.1f}")
    print(f"Tracked plates: {len(results_data)}")
    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()