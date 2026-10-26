"""
YOLOv8 Object Detection Module for VisionTalk AI
Loads pretrained YOLOv8 model, runs real inference, extracts bounding boxes and confidence,
draws annotated visualization overlays using PIL, and provides graceful fallbacks.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

# Deterministic distinct colors for detection classes
CLASS_COLORS = [
    (239, 68, 68),    # Red
    (59, 130, 246),   # Blue
    (16, 185, 129),   # Green
    (245, 158, 11),   # Amber
    (139, 92, 246),   # Purple
    (236, 72, 153),   # Pink
    (14, 165, 233),   # Sky
    (20, 184, 166),   # Teal
    (249, 115, 22),   # Orange
    (99, 102, 241),   # Indigo
]


class YOLOv8Detector:
    """Pretrained YOLOv8 object detector with annotation and graceful fallback."""

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.25):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None
        self.is_available = False
        self.error_message: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        """Attempts to import ultralytics and load the pretrained YOLOv8 weights."""
        try:
            from ultralytics import YOLO
            # Load pretrained model (downloads yolov8n.pt once if not cached)
            self.model = YOLO(self.model_name)
            self.is_available = True
            self.error_message = None
        except ImportError as e:
            self.is_available = False
            self.error_message = f"Object detection unavailable: Ultralytics or PyTorch not installed ({str(e)})."
        except Exception as e:
            self.is_available = False
            self.error_message = f"Object detection unavailable: Failed to load YOLOv8 model ({str(e)})."

    def detect(
        self,
        image: Image.Image,
        conf_threshold: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Runs real YOLOv8 object detection on the PIL image.
        Returns:
            (detections: list of dicts, class_counts: dict of label->count)
        """
        if not self.is_available or self.model is None:
            return [], {}

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        try:
            results = self.model(image, conf=threshold, verbose=False)
            if not results or len(results) == 0:
                return [], {}

            first_result = results[0]
            boxes = first_result.boxes
            if boxes is None or len(boxes) == 0:
                return [], {}

            detections: List[Dict[str, Any]] = []
            class_counts: Dict[str, int] = {}

            names = self.model.names if hasattr(self.model, "names") else {}

            for box in boxes:
                # Extract coordinates [x1, y1, x2, y2]
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [round(float(v), 2) for v in xyxy]

                conf = round(float(box.conf[0].item()), 4)
                cls_id = int(box.cls[0].item())
                label = names.get(cls_id, f"class_{cls_id}")

                det = {
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "label": label
                }
                detections.append(det)
                class_counts[label] = class_counts.get(label, 0) + 1

            return detections, class_counts

        except Exception as e:
            # On runtime inference error, do not crash
            print(f"Inference error: {e}")
            return [], {}

    def annotate_image(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
        show_labels: bool = True,
        show_conf: bool = True,
        line_width: int = 3
    ) -> Image.Image:
        """
        Draws bounding boxes and labels onto a copy of the image.
        Uses PIL ImageDraw with responsive font sizing and colored label tags.
        """
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size

        # Dynamic font sizing based on image dimensions
        font_size = max(12, int(min(width, height) * 0.025))
        try:
            # Try system or default truetype font if available
            font = ImageFont.load_default()
        except Exception:
            font = None

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_id = det.get("class_id", 0)
            label = det.get("label", "object")
            conf = det.get("confidence", 0.0)

            # Assign consistent color from palette
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

            # Draw outer rectangle
            for w in range(line_width):
                draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=color)

            # Draw label tag
            if show_labels:
                tag_text = label
                if show_conf:
                    tag_text += f" {conf:.1%}"

                # Compute text bounding box
                if hasattr(draw, "textbbox"):
                    bbox = draw.textbbox((x1, y1), tag_text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                else:
                    text_w, text_h = (len(tag_text) * 7, 12)

                # Draw filled label background
                tag_y0 = max(0, y1 - text_h - 4)
                tag_y1 = y1 if y1 - text_h - 4 >= 0 else y1 + text_h + 4
                tag_x1 = min(width, x1 + text_w + 6)

                draw.rectangle([x1, tag_y0, tag_x1, tag_y1], fill=color)
                draw.text((x1 + 3, tag_y0 + 1), tag_text, fill=(255, 255, 255), font=font)

        return annotated
