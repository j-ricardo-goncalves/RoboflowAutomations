import os
import json

from roboflow import Roboflow
import config


def get_project():
    rf = Roboflow(api_key=config.API_KEY)
    return rf.workspace(config.WORKSPACE).project(config.PROJECT)


def upload_annotations(project, annotated: list) -> dict:
    """
    Attaches annotation files to images that already exist in Roboflow
    using project.save_annotation() — does NOT re-upload the image.

    Prefers YOLO .txt, falls back to JSON if that's all that was generated.
    """
    results = {"uploaded": 0, "skipped": 0, "failed": 0}

    for img in annotated:
        if config.DRY_RUN:
            print(f"[uploader] DRY RUN — would annotate {img['name']} (id={img['id']})")
            results["skipped"] += 1
            continue

        if not img.get("id"):
            # local mode — no Roboflow image ID, fall back to full upload
            _upload_new(project, img, results)
            continue

        annotation_path = img.get("txt_path") or img.get("json_path")
        if not annotation_path:
            print(f"[uploader] no annotation file for {img['name']}, skipping")
            results["skipped"] += 1
            continue
        annotation_labelmap = _annotation_labelmap(img)

        try:
            project.save_annotation(
                image_id=img["id"],
                annotation_path=annotation_path,
                annotation_labelmap=annotation_labelmap,
                is_prediction=True,       # marks as model-generated, pending human review
                annotation_overwrite=False,  # don't overwrite if someone already labeled it
            )
            print(f"[uploader] saved annotation for {img['name']}")
            results["uploaded"] += 1
        except Exception as e:
            print(f"[uploader] failed on {img['name']}: {e}")
            results["failed"] += 1

    return results


def upload_skipped_images(project) -> dict:
    """
    Uploads no-detection images listed in the run manifest as unannotated images.
    """
    results = {"uploaded": 0, "skipped": 0, "failed": 0}
    skipped_txt_path = _skipped_txt_path()

    if not skipped_txt_path:
        print("[uploader] no skipped-image manifest found, skipping unannotated upload")
        return results

    with open(skipped_txt_path) as f:
        image_paths = [line.strip() for line in f if line.strip()]

    for image_path in image_paths:
        if config.DRY_RUN:
            print(f"[uploader] DRY RUN — would upload unannotated {image_path}")
            results["skipped"] += 1
            continue

        try:
            project.upload(
                image_path=image_path,
                batch_name=config.UPLOAD_BATCH_NAME,
                tag_names=config.SKIPPED_UPLOAD_TAGS,
                num_retry_uploads=3,
            )
            print(f"[uploader] uploaded unannotated image {image_path}")
            results["uploaded"] += 1
        except Exception as e:
            print(f"[uploader] failed unannotated upload {image_path}: {e}")
            results["failed"] += 1

    return results


def load_existing_annotations(images: list) -> list:
    """
    Creates upload records by matching LOCAL_IMAGE_DIR images with existing
    annotation files in WORKING_DIR.
    """
    annotated = []
    for img in images:
        base_name = os.path.splitext(img["name"])[0]
        txt_path = os.path.join(config.WORKING_DIR, f"{base_name}.txt")
        json_path = os.path.join(config.WORKING_DIR, f"{base_name}.json")

        record = {**img}
        if os.path.exists(txt_path):
            record["txt_path"] = txt_path
            if os.path.exists(json_path):
                record["json_path"] = json_path
        elif os.path.exists(json_path):
            record["json_path"] = json_path
        else:
            print(f"[uploader] no existing annotation for {img['name']}, skipping")
            continue

        annotated.append(record)

    print(f"[uploader] found {len(annotated)} existing annotated images")
    return annotated


def _upload_new(project, img: dict, results: dict) -> None:
    """Fallback for local mode — image doesn't exist in Roboflow yet."""
    annotation_path = img.get("txt_path") or img.get("json_path")
    annotation_labelmap = _annotation_labelmap(img)
    try:
        project.upload(
            image_path=img["path"],
            annotation_path=annotation_path,
            annotation_labelmap=annotation_labelmap,
            batch_name=config.UPLOAD_BATCH_NAME,
            tag_names=config.UPLOAD_TAGS,
            num_retry_uploads=3,
        )
        print(f"[uploader] uploaded new image {img['name']}")
        results["uploaded"] += 1
    except Exception as e:
        print(f"[uploader] failed on {img['name']}: {e}")
        results["failed"] += 1


def _annotation_labelmap(img: dict) -> dict | None:
    json_path = img.get("json_path")
    if not json_path or not os.path.exists(json_path):
        return None

    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"[uploader] could not read label map from {json_path}: {e}")
        return None

    labelmap = {}
    for pred in data.get("predictions", []):
        class_name = pred.get("class_name")
        class_id = pred.get("class_id")
        if class_name is None or class_id is None:
            continue
        labelmap[int(class_id)] = class_name

    return labelmap or None


def _skipped_txt_path() -> str | None:
    path = config.RUN_MANIFEST_PATH.rsplit(".", 1)[0] + "_skipped.txt"
    try:
        with open(path):
            return path
    except FileNotFoundError:
        return None
