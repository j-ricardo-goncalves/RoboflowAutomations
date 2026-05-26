from utils import clamp
import config


def _bbox_to_polygon(x: float, y: float, w: float, h: float) -> list:
    """
    Fallback for detection models that return center-x, center-y, w, h
    instead of polygon points. Produces a 4-point rectangle.
    """
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y - h / 2
    x3, y3 = x + w / 2, y + h / 2
    x4, y4 = x - w / 2, y + h / 2
    return [{"x": x1, "y": y1}, {"x": x2, "y": y2},
            {"x": x3, "y": y3}, {"x": x4, "y": y4}]


def convert_to_yolo(predictions: list, img_w: float, img_h: float, output_path: str) -> None:
    """
    Converts raw Roboflow inference predictions to YOLO segmentation format.

    YOLO segmentation format (one line per detection):
      <class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
    All coordinates normalized to [0.0, 1.0].

    Writes the .txt file to output_path.
    Does nothing if no valid lines are produced.
    """
    if not img_w or not img_h:
        print(f"[converter] missing image dimensions, cannot normalize — skipping {output_path}")
        return

    lines = []
    for pred in predictions:
        raw_class_id = pred.get("class_id", 0)
        class_id = config.CLASS_MAP.get(raw_class_id, raw_class_id)

        # segmentation model: Roboflow returns "points" list
        # detection model: Roboflow returns center x/y + width/height
        points = pred.get("points") or _bbox_to_polygon(
            pred["x"], pred["y"], pred["width"], pred["height"]
        )

        if len(points) < 3:
            continue  # YOLO requires at least 3 polygon points

        normalized = []
        for pt in points:
            nx = clamp(round(pt["x"] / img_w, 6))
            ny = clamp(round(pt["y"] / img_h, 6))
            normalized.extend([nx, ny])

        lines.append(f"{class_id} {' '.join(map(str, normalized))}")

    if lines:
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
