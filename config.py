from dotenv import load_dotenv
from consts import VALID_ANNOTATION_FORMATS, ANNOTATION_FORMAT_BOTH
import os

load_dotenv()  # only called here — every other file imports this module

# ── required ──────────────────────────────────────────────────────────────────
API_KEY            = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE          = os.environ["ROBOFLOW_WORKSPACE"]
PROJECT            = os.environ["ROBOFLOW_PROJECT"]

# model can live in a different project — falls back to same project if not set
MODEL_WORKSPACE    = os.getenv("ROBOFLOW_MODEL_WORKSPACE") or WORKSPACE
MODEL_PROJECT      = os.getenv("ROBOFLOW_MODEL_PROJECT")   or PROJECT
MODEL_VERSION      = int(os.getenv("ROBOFLOW_MODEL_VERSION", "1"))

# ── inference ─────────────────────────────────────────────────────────────────
CONFIDENCE         = int(os.getenv("CONFIDENCE_THRESHOLD", "40"))   # 0–100
IOU                = int(os.getenv("IOU_THRESHOLD", "30"))           # 0–100

# ── fetch ─────────────────────────────────────────────────────────────────────
LOCAL_IMAGE_DIR    = os.getenv("LOCAL_IMAGE_DIR", "").strip() or None
MAX_IMAGES         = int(os.getenv("MAX_IMAGES", "0"))               # 0 = no limit

# ── output ────────────────────────────────────────────────────────────────────
ANNOTATION_FORMAT  = os.getenv("ANNOTATION_FORMAT", ANNOTATION_FORMAT_BOTH).lower()
WORKING_DIR        = os.getenv("WORKING_DIR", "/tmp/a2")

# ── pipeline behaviour ────────────────────────────────────────────────────────
UPLOAD_ANNOTATIONS = os.getenv("UPLOAD_ANNOTATIONS", "true").lower()  == "true"
DRY_RUN            = os.getenv("DRY_RUN", "false").lower() == "true"

# ── validation ────────────────────────────────────────────────────────────────
if ANNOTATION_FORMAT not in VALID_ANNOTATION_FORMATS:
    raise ValueError(
        f"ANNOTATION_FORMAT must be one of {VALID_ANNOTATION_FORMATS}, got '{ANNOTATION_FORMAT}'"
    )
if not (0 <= CONFIDENCE <= 100):
    raise ValueError(f"CONFIDENCE_THRESHOLD must be 0–100, got {CONFIDENCE}")
if not (0 <= IOU <= 100):
    raise ValueError(f"IOU_THRESHOLD must be 0–100, got {IOU}")
