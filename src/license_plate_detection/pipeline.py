import argparse
import time
from concurrent.futures import ThreadPoolExecutor
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
    parser.add_argument("--display-every", type=int, default=1,
                        help="Display every N source video frames")
    parser.add_argument("--no-display", action="store_true", help="Run without opening a video window")
    parser.add_argument(
        "--precise-boxes",
        action="store_true",
        help="Detect every frame synchronously so boxes align with the displayed frame",
    )
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


def _detect_frame(frame, frame_id, detector, ocr, output_dir, cv2):
    detections = []
    for index, (x1, y1, x2, y2) in enumerate(detector.detect(frame)):
        plate_img = frame[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue

        plate_text = ocr.read_text(plate_img)
        plate_filename = f"plate_{frame_id}_{index}.jpg"
        plate_path = output_dir / plate_filename
        if not cv2.imwrite(str(plate_path), plate_img):
            raise OSError(f"Could not save plate image: {plate_path}")
        detections.append((index, x1, y1, x2, y2, plate_text, plate_filename))

    return frame_id, detections


def _update_tracks(frame_id, detections, tracks, max_track_gap):
    annotations = []
    current_plates = 0
    matched_tracks = set()

    for index, x1, y1, x2, y2, plate_text, plate_filename in detections:
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

        if track["image"] is None:
            track["image"] = plate_filename
        annotations.append((x1, y1, x2, y2, plate_text))

        normalized_text = "".join(character for character in plate_text.upper() if character.isalnum())
        if normalized_text:
            current_plates += 1
            track["texts"][normalized_text] = track["texts"].get(normalized_text, 0) + 1

    return annotations, current_plates


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
    max_track_gap = max(args.frame_skip * 5, int(fps * 2), 10)
    start_time = time.time()
    last_progress_time = start_time
    playback_start = time.monotonic()
    last_annotations = []
    last_plates = 0
    pending_detection = None
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        while True:
            if not args.no_display and not args.precise_boxes and fps > 0:
                target_frame = int((time.monotonic() - playback_start) * fps) + 1
                while frame_id + 1 < target_frame:
                    if not cap.grab():
                        break
                    frame_id += 1

            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            should_detect = args.precise_boxes or frame_id % args.frame_skip == 0
            if pending_detection is not None and pending_detection.done():
                detected_frame_id, detections = pending_detection.result()
                last_annotations, last_plates = _update_tracks(
                    detected_frame_id, detections, tracks, max_track_gap
                )
                pending_detection = None

            if should_detect:
                if args.no_display or args.precise_boxes:
                    detected_frame_id, detections = _detect_frame(
                        frame, frame_id, detector, ocr, args.output_dir, cv2
                    )
                    last_annotations, last_plates = _update_tracks(
                        detected_frame_id, detections, tracks, max_track_gap
                    )
                    processed_frames += 1
                elif pending_detection is None:
                    pending_detection = executor.submit(
                        _detect_frame, frame.copy(), frame_id, detector, ocr,
                        args.output_dir, cv2
                    )
                    processed_frames += 1

            if not args.no_display and frame_id % args.display_every == 0:
                elapsed = max(time.time() - start_time, 0.001)
                info_text = (
                    f"Frame: {frame_id}/{total_frames} | Plates: {last_plates} | "
                    f"FPS: {processed_frames / elapsed:.1f}"
                )
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
                for x1, y1, x2, y2, plate_text in last_annotations:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, plate_text, (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("License Plate Detection", frame)
                wait_ms = 1
                if fps > 0:
                    target_time = playback_start + (frame_id - 1) / fps
                    wait_ms = max(1, round((target_time - time.monotonic()) * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    pause_start = time.monotonic()
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key in (ord(" "), ord("q")):
                            playback_start += time.monotonic() - pause_start
                            break
                    if key == ord("q"):
                        break

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
        executor.shutdown(wait=True)

    if pending_detection is not None:
        detected_frame_id, detections = pending_detection.result()
        last_annotations, last_plates = _update_tracks(
            detected_frame_id, detections, tracks, max_track_gap
        )

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