import os
from roboflow import Roboflow
from ultralytics import YOLO
from tqdm import tqdm
import config
from utils import dump_to_json
from consts import ANNOTATION_FORMAT_JSON, ANNOTATION_FORMAT_YOLO, ANNOTATION_FORMAT_BOTH
from converter import convert_to_yolo
from visualizer import save_annotated_image


def load_model():
    if not os.path.exists(config.MODEL_WEIGHTS):
        raise FileNotFoundError(
            f"No weights found at {config.MODEL_WEIGHTS}\n"
            f"Mount with: -v /your/local/best.pt:{config.MODEL_WEIGHTS}"
        )
    model = YOLO(config.MODEL_WEIGHTS)
    print(f"[annotate] loaded local model from {config.MODEL_WEIGHTS}")
    return model


def build_class_map(model):
    """
    Builds class ID remapping using the YOLO model's own class names
    (model.names) mapped to the target project's class IDs.

    model.names is the ground truth: {0: 'obstacle', 1: 'parking', ...}
    target_project.classes is {name: count} — we enumerate keys for IDs.

    Returns a dict: {model_class_id (int) -> target_class_id (int)}
    """
    rf = Roboflow(api_key=config.API_KEY)
    target_project = rf.workspace(config.WORKSPACE).project(config.PROJECT)

    # model.names: {int -> class_name} — this is the true mapping the model uses
    model_names = model.names  # e.g. {0: 'obstacle', 1: 'parking', ...}

    # target project class list: {name: count}, enumerate keys for positional IDs
    target_classes = {name: i for i, name in enumerate(target_project.classes)}

    print(f"[annotate] model names  : {model_names}")
    print(f"[annotate] target classes: {target_classes}")

    class_map = {}
    for model_id, name in model_names.items():
        if name in target_classes:
            class_map[model_id] = target_classes[name]
        else:
            print(f"[annotate] warning: class '{name}' not in target project — will be skipped")

    print(f"[annotate] class map: {class_map}")
    return class_map


def annotate(model, images: list, class_map: dict) -> list:
    """
    Runs local YOLO inference on each image.
    Remaps class IDs using class_map before writing annotation files.
    Writes JSON and/or YOLO .txt files depending on ANNOTATION_FORMAT.
    """
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    annotated = []
    skipped = []

    for img in tqdm(images, desc="annotating"):
        try:
            preds  = model(img["path"], conf=config.CONFIDENCE / 100)
            result = preds[0]
            img_h, img_w = result.orig_shape

            predictions = []
            for i, box in enumerate(result.boxes):
                model_class_id = int(box.cls[0])

                # remap to target project class ID — skip if class not in target
                target_class_id = class_map.get(model_class_id)
                if target_class_id is None:
                    continue

                pred = {
                    "class_id": target_class_id,
                    "model_class_id": model_class_id,
                    "class_name": model.names.get(model_class_id, str(model_class_id)),
                    "confidence": float(box.conf[0]),
                }

                if result.masks is not None:
                    # segmentation model — use polygon points
                    polygon = result.masks.xy[i]
                    pred["points"] = [{"x": float(p[0]), "y": float(p[1])} for p in polygon]
                else:
                    # detection model — pass bbox, converter makes a rectangle
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    pred["x"]      = (x1 + x2) / 2
                    pred["y"]      = (y1 + y2) / 2
                    pred["width"]  = x2 - x1
                    pred["height"] = y2 - y1

                predictions.append(pred)

        except Exception as e:
            print(f"[annotate] inference failed on {img['name']}: {e}")
            skipped.append({**img, "reason": f"inference failed: {e}"})
            continue

        if not predictions:
            print(f"[annotate] no detections in {img['name']}, skipping")
            skipped.append({**img, "reason": "no detections after confidence/class filtering"})
            continue

        base_name   = os.path.splitext(img["name"])[0]
        result_dict = {**img, "predictions": predictions}

        if config.ANNOTATION_FORMAT in (ANNOTATION_FORMAT_JSON, ANNOTATION_FORMAT_BOTH):
            json_path = os.path.join(config.WORKING_DIR, f"{base_name}.json")
            if not config.DRY_RUN:
                dump_to_json(json_path, {"predictions": predictions})
            result_dict["json_path"] = json_path

        if config.ANNOTATION_FORMAT in (ANNOTATION_FORMAT_YOLO, ANNOTATION_FORMAT_BOTH):
            txt_path = os.path.join(config.WORKING_DIR, f"{base_name}.txt")
            if not config.DRY_RUN:
                convert_to_yolo(predictions, img_w, img_h, txt_path)
            result_dict["txt_path"] = txt_path

        if config.RENDER_ANNOTATED_IMAGES:
            preview_path = os.path.join(config.ANNOTATED_IMAGE_DIR, img["name"])
            if not config.DRY_RUN:
                save_annotated_image(img["path"], predictions, model.names, preview_path)
            result_dict["preview_path"] = preview_path

        annotated.append(result_dict)
        print(f"[annotate] {img['name']}: {len(predictions)} detections")

    if config.WRITE_RUN_MANIFEST and not config.DRY_RUN:
        _write_run_manifest(images, annotated, skipped)

    return annotated


def _write_run_manifest(images: list, annotated: list, skipped: list) -> None:
    manifest = {
        "total": len(images),
        "annotated_count": len(annotated),
        "skipped_count": len(skipped),
        "annotated": annotated,
        "skipped": skipped,
    }
    dump_to_json(config.RUN_MANIFEST_PATH, manifest)

    skipped_txt_path = os.path.splitext(config.RUN_MANIFEST_PATH)[0] + "_skipped.txt"
    with open(skipped_txt_path, "w") as f:
        for img in skipped:
            f.write(f"{img['path']}\n")

    annotated_txt_path = os.path.splitext(config.RUN_MANIFEST_PATH)[0] + "_annotated.txt"
    with open(annotated_txt_path, "w") as f:
        for img in annotated:
            f.write(f"{img['path']}\n")

    print(f"[annotate] wrote manifest: {config.RUN_MANIFEST_PATH}")
    print(f"[annotate] wrote skipped list: {skipped_txt_path}")
