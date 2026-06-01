import config
from fetcher   import fetch_images
from annotate  import load_model, build_class_map, annotate
from uploader  import get_project, load_existing_annotations, upload_annotations, upload_skipped_images


def _local_class_map(model):
    class_map = config.CLASS_MAP or {int(class_id): int(class_id) for class_id in model.names}
    print(f"[main] local class map   : {class_map}")
    return class_map


def main():
    config.validate_config()

    print("=" * 50)
    print("  roboflow auto-annotate (extended)")
    print("=" * 50)

    if config.DRY_RUN:
        print("[main] DRY RUN — no files will be written, no uploads will happen")

    print(f"[main] inference model   : {config.MODEL_WEIGHTS}")
    print(f"[main] model project     : {config.MODEL_PROJECT}")
    print(f"[main] target project    : {config.PROJECT}")
    print(f"[main] confidence        : {config.CONFIDENCE}")
    print(f"[main] annotation format : {config.ANNOTATION_FORMAT}")
    print(f"[main] run inference     : {config.RUN_INFERENCE}")
    print(f"[main] upload back       : {config.UPLOAD_ANNOTATIONS}")
    print(f"[main] upload skipped    : {config.UPLOAD_SKIPPED_IMAGES}")
    print(f"[main] upload tags       : {config.UPLOAD_TAGS or 'none'}")
    print(f"[main] upload batch      : {config.UPLOAD_BATCH_NAME or 'none'}")
    print(f"[main] render previews   : {config.RENDER_ANNOTATED_IMAGES}")
    print(f"[main] local image dir   : {config.LOCAL_IMAGE_DIR or 'none (fetching from Roboflow)'}")
    print()

    project = None
    if config.UPLOAD_ANNOTATIONS:
        project = get_project()

    images = fetch_images()
    if not images:
        print("[main] no images to process, exiting")
        return

    if config.RUN_INFERENCE:
        model = load_model()
        if config.UPLOAD_ANNOTATIONS:
            class_map = build_class_map(model)
        else:
            class_map = _local_class_map(model)
        annotated = annotate(model, images, class_map)
    else:
        print("[main] RUN_INFERENCE=false — using existing annotation files")
        annotated = load_existing_annotations(images)

    results = {"uploaded": 0, "skipped": 0, "failed": 0}
    if config.UPLOAD_ANNOTATIONS:
        results = upload_annotations(project, annotated)
        if config.UPLOAD_SKIPPED_IMAGES:
            skipped_results = upload_skipped_images(project)
            results = {
                key: results[key] + skipped_results[key]
                for key in results
            }
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
