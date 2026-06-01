# RoboflowAutomations

Automated image annotation pipeline for Roboflow projects. Runs a local YOLO model on unannotated images and uploads the results back to Roboflow.

Built for the [Técnico Fuel Cell](https://github.com/TecnicoFuelCell) in order to reduce annotation time for the AS department.

## How it works

1. Fetches unannotated images from a Roboflow project (or reads from a local folder)
2. Runs a local YOLO model on each image
3. Uploads the annotations back to Roboflow

## Setup

Copy `.env.example` to `.env` and fill in your values.

```bash
docker build -t roboflow-automations .
docker run --env-file .env -v /path/to/best.pt:/tmp/weights.pt roboflow-automations
```

## Configuration

| Variable | Description | Default |
|---|---|---|
| `ROBOFLOW_API_KEY` | Your Roboflow API key | required |
| `ROBOFLOW_WORKSPACE` | Workspace name | required |
| `ROBOFLOW_PROJECT` | Target project name | required |
| `MODEL_WEIGHTS_PATH` | Path to your YOLO `.pt` file | `/tmp/weights.pt` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence (0–100) | `50` |
| `CLASS_MAP` | Explicit class ID remapping (JSON) | none |
| `LOCAL_IMAGE_DIR` | Use local images instead of fetching | none |
| `MAX_IMAGES` | Cap number of images processed | `0` (no limit) |
| `DRY_RUN` | Run without writing or uploading | `false` |

> **Note on CLASS_MAP:** if your YOLO model and Roboflow project have classes in a different order, set this explicitly — e.g. `CLASS_MAP={"0":"2","1":"0","2":"1"}`. The pipeline will warn you if a mismatch is detected.

## Requirements
Needs Docker and Docker BuildX to run and build the container. <br>
Although, it can be ran directly from terminal, it is not recommended.