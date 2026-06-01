import os
import requests
from roboflow import Roboflow
import config
from consts import SUPPORTED_IMAGE_FORMATS
from utils import get_image_paths
import time

IMAGE_DETAIL_API = "https://api.roboflow.com/{workspace}/{project}/images/{image_id}"

def _get_with_retry(url: str, retries: int = 3, backoff: float = 2.0) -> requests.Response:
    """
    GET a URL with exponential backoff retries.
    Waits 2s, 4s, 8s between attempts before giving up.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_error = e
            if attempt < retries:
                wait = backoff ** attempt
                print(f"[fetcher] attempt {attempt}/{retries} failed, retrying in {wait}s — {e}")
                time.sleep(wait)

    raise requests.RequestException(
        f"[fetcher] all {retries} attempts failed: {last_error}"
    )

def fetch_images() -> list:
    if config.LOCAL_IMAGE_DIR:
        return _load_local_images(config.LOCAL_IMAGE_DIR)
    return _fetch_from_roboflow()


def _load_local_images(directory: str) -> list:
    paths = get_image_paths(directory, SUPPORTED_IMAGE_FORMATS)
    images = [
        {"id": None, "name": os.path.basename(p), "path": p}
        for p in paths
    ]
    print(f"[fetcher] local mode — {len(images)} images in {directory}")
    return images


def _get_image_url(image_id: str) -> str | None:
    """Fetch the original download URL for a single image via the detail endpoint."""
    url = IMAGE_DETAIL_API.format(
        workspace=config.WORKSPACE,
        project=config.PROJECT,
        image_id=image_id
    )
    try:
        r = _get_with_retry(f"{url}?api_key={config.API_KEY}")
        return r.json().get("image", {}).get("urls", {}).get("original")
    except requests.RequestException:
        return None

def _fetch_from_roboflow() -> list:
    """
    Uses project.search_all() from the SDK to find unannotated images,
    then downloads each via its original URL for local inference.
    """
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    rf      = Roboflow(api_key=config.API_KEY)
    project = rf.workspace(config.WORKSPACE).project(config.PROJECT)

    # collect all images via SDK — handles pagination automatically
    all_results = []
    for page in project.search_all(
        fields=["id", "name", "annotations"],
        limit=250,
    ):
        all_results.extend(page)

    # unannotated = no annotations field, or annotations.count == 0
    unannotated = [
        img for img in all_results
        if not img.get("annotations") or img["annotations"].get("count", 0) == 0
    ]
    print(f"[fetcher] {len(all_results)} total, {len(unannotated)} unannotated")

    if config.MAX_IMAGES > 0:
        unannotated = unannotated[: config.MAX_IMAGES]
        print(f"[fetcher] capped to {len(unannotated)} (MAX_IMAGES={config.MAX_IMAGES})")

    downloaded = []
    for img in unannotated:
        image_id   = img["id"]
        image_name = img.get("name", f"{image_id}.jpg")

        image_url = _get_image_url(image_id)
        if not image_url:
            print(f"[fetcher] no URL for {image_name}, skipping")
            continue

        local_path = os.path.join(config.WORKING_DIR, image_name)
        try:
            r = _get_with_retry(image_url)
            with open(local_path, "wb") as f:
                f.write(r.content)
            downloaded.append({"id": image_id, "name": image_name, "path": local_path})
            print(f"[fetcher] downloaded {image_name}")
        except requests.RequestException as e:
            print(f"[fetcher] giving up on {image_name} after retries: {e}")
            continue

    return downloaded