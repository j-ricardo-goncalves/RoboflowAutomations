from dotenv import load_dotenv
from consts import VALID_ANNOTATION_FORMATS, ANNOTATION_FORMAT_BOTH
import os
import json

load_dotenv()  # only called here — every other file imports this module


def _get(key: str, default: str = None) -> str:
    """Get env var and strip whitespace — prevents inline comment bleed."""
    val = os.environ.get(key, default) if default is not None else os.environ[key]
    return val.strip() if val else val


def _get_list(key: str) -> list[str]:
    val = _get(key, "")
    if not val:
        return []
    return [item.strip() for item in val.split(",") if item.strip()]


# ── Roboflow settings ─────────────────────────────────────────────────────────
# Only required when fetching from Roboflow or uploading annotations.
API_KEY            = _get("ROBOFLOW_API_KEY", "")
WORKSPACE          = _get("ROBOFLOW_WORKSPACE", "")
PROJECT            = _get("ROBOFLOW_PROJECT", "")

# model can live in a different project — falls back to same project if not set
MODEL_WORKSPACE    = _get("ROBOFLOW_MODEL_WORKSPACE", WORKSPACE) or WORKSPACE
MODEL_PROJECT      = _get("ROBOFLOW_MODEL_PROJECT",   PROJECT)   or PROJECT
MODEL_VERSION      = int(_get("ROBOFLOW_MODEL_VERSION", "1"))

# ── inference ─────────────────────────────────────────────────────────────────
CONFIDENCE         = int(_get("CONFIDENCE_THRESHOLD", "40"))   # 0–100
IOU                = int(_get("IOU_THRESHOLD", "30"))           # 0–100

# ── fetch ─────────────────────────────────────────────────────────────────────
LOCAL_IMAGE_DIR    = _get("LOCAL_IMAGE_DIR", "") or None
MAX_IMAGES         = int(_get("MAX_IMAGES", "0"))               # 0 = no limit

# ── output ────────────────────────────────────────────────────────────────────
ANNOTATION_FORMAT  = _get("ANNOTATION_FORMAT", ANNOTATION_FORMAT_BOTH).lower()
WORKING_DIR        = _get("WORKING_DIR", "/tmp/a2")

# ── class mapping ─────────────────────────────────────────────────────────────
# maps model class IDs to target project class IDs when projects have
# different class orderings. e.g. CLASS_MAP={"0":"4","1":"1","2":"0","3":"3","4":"2"}
# leave empty ({}) if both projects have the same class order
_class_map_raw     = _get("CLASS_MAP", "{}")
CLASS_MAP          = {int(k): int(v) for k, v in json.loads(_class_map_raw).items()}

# ── pipeline behaviour ────────────────────────────────────────────────────────
RUN_INFERENCE      = _get("RUN_INFERENCE", "true").lower() == "true"
UPLOAD_ANNOTATIONS = _get("UPLOAD_ANNOTATIONS", "true").lower() == "true"
DRY_RUN            = _get("DRY_RUN", "false").lower() == "true"
RENDER_ANNOTATED_IMAGES = _get("RENDER_ANNOTATED_IMAGES", "false").lower() == "true"
ANNOTATED_IMAGE_DIR = _get(
    "ANNOTATED_IMAGE_DIR",
    os.path.join(WORKING_DIR, "annotated_images"),
)
WRITE_RUN_MANIFEST = _get("WRITE_RUN_MANIFEST", "true").lower() == "true"
RUN_MANIFEST_PATH = _get("RUN_MANIFEST_PATH", os.path.join(WORKING_DIR, "run_manifest.json"))
UPLOAD_TAGS = _get_list("UPLOAD_TAGS")
SKIPPED_UPLOAD_TAGS = _get_list("SKIPPED_UPLOAD_TAGS") or UPLOAD_TAGS
UPLOAD_BATCH_NAME = _get("UPLOAD_BATCH_NAME", "") or None
UPLOAD_SKIPPED_IMAGES = _get("UPLOAD_SKIPPED_IMAGES", "false").lower() == "true"

# ── validation ────────────────────────────────────────────────────────────────
if ANNOTATION_FORMAT not in VALID_ANNOTATION_FORMATS:
    raise ValueError(
        f"ANNOTATION_FORMAT must be one of {VALID_ANNOTATION_FORMATS}, got '{ANNOTATION_FORMAT}'"
    )
if not (0 <= CONFIDENCE <= 100):
    raise ValueError(f"CONFIDENCE_THRESHOLD must be 0–100, got {CONFIDENCE}")
if not (0 <= IOU <= 100):
    raise ValueError(f"IOU_THRESHOLD must be 0–100, got {IOU}")

# Local Inference
MODEL_WEIGHTS = _get("MODEL_WEIGHTS_PATH", "/tmp/weights/best.pt")
