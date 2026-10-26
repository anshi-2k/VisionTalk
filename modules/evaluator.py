"""
Evaluation Module for VisionTalk AI
Computes real object detection evaluation metrics:
- Intersection over Union (IoU)
- True Positives, False Positives, False Negatives
- Precision, Recall, F1-Score
- Average Precision (AP) and mean Average Precision (mAP@0.5)
- Precision-Recall curves and metric summaries
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes formatted as [x1, y1, x2, y2].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_width = max(0.0, x2 - x1)
    inter_height = max(0.0, y2 - y1)
    inter_area = inter_width * inter_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using standard 11-point or AUC interpolation.
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    # Append sentinel values at edges
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute maximum precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum area under envelope
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


class DetectionEvaluator:
    """Evaluates object detector performance against ground-truth annotations."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        ground_truths: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates predictions against ground truth boxes.
        
        Args:
            ground_truths: list of dicts: {"image_id": str, "label": str, "box": [x1, y1, x2, y2]}
            predictions: list of dicts: {"image_id": str, "label": str, "box": [x1, y1, x2, y2], "confidence": float}

        Returns:
            Dictionary with per-class and overall metrics (Precision, Recall, F1, AP, mAP).
        """
        # Collect unique classes
        classes = sorted(list(set(
            [gt["label"] for gt in ground_truths] + [pred["label"] for pred in predictions]
        )))

        per_class_results: Dict[str, Dict[str, Any]] = {}
        all_aps = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for cls in classes:
            # Filter for current class
            cls_gts = [gt for gt in ground_truths if gt["label"] == cls]
            cls_preds = [p for p in predictions if p["label"] == cls]

            # Sort predictions by confidence descending
            cls_preds = sorted(cls_preds, key=lambda x: x.get("confidence", 0.0), reverse=True)

            num_gts = len(cls_gts)
            num_preds = len(cls_preds)

            if num_gts == 0:
                # Class has no ground truths but has predictions -> all FP
                per_class_results[cls] = {
                    "tp": 0,
                    "fp": num_preds,
                    "fn": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "ap": 0.0,
                    "gt_count": 0,
                    "pred_count": num_preds,
                    "pr_curve": {"recalls": [0.0], "precisions": [0.0]}
                }
                total_fp += num_preds
                continue

            if num_preds == 0:
                # Class has ground truths but zero predictions -> all FN
                per_class_results[cls] = {
                    "tp": 0,
                    "fp": 0,
                    "fn": num_gts,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "ap": 0.0,
                    "gt_count": num_gts,
                    "pred_count": 0,
                    "pr_curve": {"recalls": [0.0], "precisions": [0.0]}
                }
                total_fn += num_gts
                all_aps.append(0.0)
                continue

            # Group ground truths by image_id
            gt_by_img: Dict[str, List[Dict[str, Any]]] = {}
            for gt in cls_gts:
                img_id = str(gt.get("image_id", "default"))
                gt_by_img.setdefault(img_id, []).append({
                    "box": gt["box"],
                    "matched": False
                })

            tp_flags = np.zeros(num_preds)
            fp_flags = np.zeros(num_preds)

            for p_idx, pred in enumerate(cls_preds):
                img_id = str(pred.get("image_id", "default"))
                pred_box = pred["box"]
                best_iou = 0.0
                best_gt_idx = -1

                img_gts = gt_by_img.get(img_id, [])
                for g_idx, candidate_gt in enumerate(img_gts):
                    iou = compute_iou(pred_box, candidate_gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    if not img_gts[best_gt_idx]["matched"]:
                        tp_flags[p_idx] = 1.0
                        img_gts[best_gt_idx]["matched"] = True
                    else:
                        fp_flags[p_idx] = 1.0  # Duplicate detection
                else:
                    fp_flags[p_idx] = 1.0  # Low IoU or wrong location

            cum_tp = np.cumsum(tp_flags)
            cum_fp = np.cumsum(fp_flags)

            recalls = cum_tp / num_gts
            precisions = cum_tp / (cum_tp + cum_fp)

            ap = compute_ap(recalls, precisions)
            all_aps.append(ap)

            cls_tp = int(cum_tp[-1]) if len(cum_tp) > 0 else 0
            cls_fp = int(cum_fp[-1]) if len(cum_fp) > 0 else 0
            cls_fn = num_gts - cls_tp

            total_tp += cls_tp
            total_fp += cls_fp
            total_fn += cls_fn

            final_prec = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
            final_rec = cls_tp / num_gts if num_gts > 0 else 0.0
            final_f1 = (2 * final_prec * final_rec / (final_prec + final_rec)) if (final_prec + final_rec) > 0 else 0.0

            per_class_results[cls] = {
                "tp": cls_tp,
                "fp": cls_fp,
                "fn": cls_fn,
                "precision": round(final_prec, 4),
                "recall": round(final_rec, 4),
                "f1": round(final_f1, 4),
                "ap": round(ap, 4),
                "gt_count": num_gts,
                "pred_count": num_preds,
                "pr_curve": {
                    "recalls": [round(float(r), 4) for r in recalls.tolist()],
                    "precisions": [round(float(p), 4) for p in precisions.tolist()]
                }
            }

        overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0
        map_score = float(np.mean(all_aps)) if all_aps else 0.0

        return {
            "iou_threshold": self.iou_threshold,
            "overall": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": round(overall_prec, 4),
                "recall": round(overall_rec, 4),
                "f1": round(overall_f1, 4),
                "mAP_50": round(map_score, 4),
                "total_classes": len(classes),
                "total_gts": len(ground_truths),
                "total_preds": len(predictions)
            },
            "per_class": per_class_results
        }
