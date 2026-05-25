import config
from fetcher   import fetch_images
from annotate  import load_model, annotate
from uploader  import get_project, upload_annotations


def main():
    print("=" * 50)
    print("  roboflow auto-annotate (extended)")
    print("=" * 50)

    if config.DRY_RUN:
        print("[main] DRY RUN — no files will be written, no uploads will happen")

    print(f"[main] inference project : {config.MODEL_PROJECT} v{config.MODEL_VERSION}")
    print(f"[main] target project    : {config.PROJECT}")
    print(f"[main] confidence        : {config.CONFIDENCE}")
    print(f"[main] iou threshold     : {config.IOU}")
    print(f"[main] annotation format : {config.ANNOTATION_FORMAT}")
    print(f"[main] upload back       : {config.UPLOAD_ANNOTATIONS}")
    print(f"[main] local image dir   : {config.LOCAL_IMAGE_DIR or 'none (fetching from Roboflow)'}")
    print()

    # authenticate early — fail fast before any work starts
    project = get_project()

    # load hosted model from Roboflow
    model = load_model()

    # stage 1: get images (local folder or fetch from Roboflow)
    images = fetch_images()
    if not images:
        print("[main] no images to process, exiting")
        return

    # stage 2: run inference + write annotation files
    annotated = annotate(model, images)

    # stage 3: upload back to Roboflow (optional)
    results = {"uploaded": 0, "skipped": 0, "failed": 0}
    if config.UPLOAD_ANNOTATIONS:
        results = upload_annotations(project, annotated)
    else:
        print("[main] UPLOAD_ANNOTATIONS=false — skipping upload")

    print()
    print("─" * 30)
    print("  run summary")
    print("─" * 30)
    print(f"  fetched      : {len(images)}")
    print(f"  annotated    : {len(annotated)}")
    print(f"  uploaded     : {results['uploaded']}")
    print(f"  skipped      : {results['skipped']}")
    print(f"  failed       : {results['failed']}")
    print("─" * 30)


if __name__ == "__main__":
    main()
