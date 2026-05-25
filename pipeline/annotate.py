import os
from roboflow import Roboflow
from tqdm import tqdm
import config
from utils import dump_to_json
from consts import ANNOTATION_FORMAT_JSON, ANNOTATION_FORMAT_YOLO, ANNOTATION_FORMAT_BOTH
from converter import convert_to_yolo


def load_model():
    """
    Loads the Roboflow hosted model.
    Uses MODEL_WORKSPACE / MODEL_PROJECT / MODEL_VERSION from config —
    these can point to a different project than the one images are fetched from.
    """
    rf = Roboflow(api_key=config.API_KEY)
    project = rf.workspace(config.MODEL_WORKSPACE).project(config.MODEL_PROJECT)
    model = project.version(config.MODEL_VERSION).model
    print(f"[annotate] model loaded: {config.MODEL_PROJECT} v{config.MODEL_VERSION}")
    return model


def annotate(model, images: list) -> list:
    """
    Runs Roboflow hosted inference on each image.
    Writes JSON and/or YOLO .txt files depending on ANNOTATION_FORMAT.

    Returns list of dicts: {image metadata + annotation paths}
    """
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    annotated = []

    for img in tqdm(images, desc="annotating"):
        try:
            raw = model.predict(
                image_path=img["path"],
                confidence=config.CONFIDENCE,
            ).json()
        except Exception as e:
            print(f"[annotate] inference failed on {img['name']}: {e}")
            continue

        predictions = raw.get("predictions", [])
        if not predictions:
            print(f"[annotate] no detections in {img['name']}, skipping")
            continue

        base_name = os.path.splitext(img["name"])[0]
        result    = {**img, "predictions": predictions}

        # ── JSON output (original repo behaviour) ──────────────────────────
        if config.ANNOTATION_FORMAT in (ANNOTATION_FORMAT_JSON, ANNOTATION_FORMAT_BOTH):
            json_path = os.path.join(config.WORKING_DIR, f"{base_name}.json")
            if not config.DRY_RUN:
                dump_to_json(json_path, raw)
            result["json_path"] = json_path

        # ── YOLO output (new) ──────────────────────────────────────────────
        if config.ANNOTATION_FORMAT in (ANNOTATION_FORMAT_YOLO, ANNOTATION_FORMAT_BOTH):
            img_w = raw.get("image", {}).get("width")
            img_h = raw.get("image", {}).get("height")
            txt_path = os.path.join(config.WORKING_DIR, f"{base_name}.txt")
            if not config.DRY_RUN:
                convert_to_yolo(predictions, float(img_w), float(img_h), txt_path)
            result["txt_path"] = txt_path

        annotated.append(result)
        print(f"[annotate] {img['name']}: {len(predictions)} detections")

    return annotated