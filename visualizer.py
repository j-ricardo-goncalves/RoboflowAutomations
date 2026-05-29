import os

import cv2
import numpy as np


COLORS = [
    (0, 180, 255),
    (90, 220, 90),
    (255, 140, 70),
    (210, 120, 255),
    (255, 220, 70),
    (80, 190, 220),
]


def _prediction_points(prediction: dict) -> np.ndarray:
    points = prediction.get("points")
    if not points:
        x = prediction["x"]
        y = prediction["y"]
        w = prediction["width"]
        h = prediction["height"]
        points = [
            {"x": x - w / 2, "y": y - h / 2},
            {"x": x + w / 2, "y": y - h / 2},
            {"x": x + w / 2, "y": y + h / 2},
            {"x": x - w / 2, "y": y + h / 2},
        ]

    return np.array(
        [[round(point["x"]), round(point["y"])] for point in points],
        dtype=np.int32,
    )


def save_annotated_image(
    image_path: str,
    predictions: list,
    class_names: dict,
    output_path: str,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print(f"[visualizer] could not read {image_path}, skipping preview")
        return

    overlay = image.copy()
    height, width = image.shape[:2]
    thickness = max(2, round(min(width, height) / 400))
    font_scale = max(0.45, min(width, height) / 1200)

    for prediction in predictions:
        class_id = int(prediction.get("model_class_id", prediction["class_id"]))
        color = COLORS[class_id % len(COLORS)]
        polygon = _prediction_points(prediction)

        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=thickness)

        x, y = polygon.min(axis=0)
        label = str(prediction.get("class_name") or class_names.get(class_id, class_id))
        confidence = prediction.get("confidence")
        if confidence is not None:
            label = f"{label} {confidence:.2f}"

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        label_x = max(0, int(x))
        label_y = max(text_size[1] + 6, int(y) - 6)
        cv2.rectangle(
            image,
            (label_x, label_y - text_size[1] - baseline - 4),
            (label_x + text_size[0] + 6, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (label_x + 3, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )

    image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
