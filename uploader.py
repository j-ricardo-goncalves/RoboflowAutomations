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

        try:
            project.save_annotation(
                image_id=img["id"],
                annotation_path=annotation_path,
                is_prediction=True,       # marks as model-generated, pending human review
                annotation_overwrite=False,  # don't overwrite if someone already labeled it
            )
            print(f"[uploader] saved annotation for {img['name']}")
            results["uploaded"] += 1
        except Exception as e:
            print(f"[uploader] failed on {img['name']}: {e}")
            results["failed"] += 1

    return results


def _upload_new(project, img: dict, results: dict) -> None:
    """Fallback for local mode — image doesn't exist in Roboflow yet."""
    annotation_path = img.get("txt_path") or img.get("json_path")
    try:
        project.upload(
            image_path=img["path"],
            annotation_path=annotation_path,
            num_retry_uploads=3,
        )
        print(f"[uploader] uploaded new image {img['name']}")
        results["uploaded"] += 1
    except Exception as e:
        print(f"[uploader] failed on {img['name']}: {e}")
        results["failed"] += 1