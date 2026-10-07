# License Plate Detection from Video

Detect license plates in video with Ultralytics YOLO and read detected plate crops with EasyOCR.

## Requirements

Install the dependencies with `pip install -r requirements.txt`.

The detector needs YOLO weights trained to detect license plates. The included `models/yolov8n.pt` is only a default model and may not contain a license-plate class. Put your trained weights in `models/` and pass their path with `--model`.

## Project layout

- `src/license_plate_detection/` contains the detector, OCR, and video pipeline.
- `data/` contains input videos and any image/label datasets.
- `models/` contains YOLO model weights.
- `output/` contains generated CSV results and plate crops.
- `legacy/` keeps the older, fixed-configuration pipeline for reference.
- `scripts/` contains local utility scripts.

## Run

From the project root, run the video pipeline:

```bash
python run_visual.py --video data/videos/car.mp4 --model models/license-plate.pt
```

For a model whose plate class is named generically, pass its class index:

```bash
python run_visual.py --video data/videos/car.mp4 --model models/license-plate.pt --class-id 0
```

Use `--precise-boxes` to detect every frame and align boxes exactly with the displayed frame; playback may be slower while inference runs. Other options include `--confidence`, `--frame-skip`, `--display-every`, `--output-dir`, `--csv`, and `--no-display`. Run `python run_visual.py --help` for the full list. The default CSV and cropped plate images are written under `output/`.
