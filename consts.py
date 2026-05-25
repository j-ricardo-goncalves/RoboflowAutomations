SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png"]

ANNOTATION_FORMAT_JSON = "json"
ANNOTATION_FORMAT_YOLO = "yolo"
ANNOTATION_FORMAT_BOTH = "both"
VALID_ANNOTATION_FORMATS = [ANNOTATION_FORMAT_JSON, ANNOTATION_FORMAT_YOLO, ANNOTATION_FORMAT_BOTH]

# Roboflow images API — list all images in a project
# ⚠️  VERIFY: if the response shape doesn't match, add print(response.json())
# in fetcher.py right after the request to inspect the real field names
ROBOFLOW_IMAGES_API = "https://api.roboflow.com/{workspace}/{project}/images"
