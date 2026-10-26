# Dataset Cleaning & Quality Audit Report

**Course**: Semester 5 Data Science (DS)  
**Project**: VisionTalk AI - Conversational Image Recognition (SIH1604)  
**Dataset**: Nano Banana 2.0 Dataset  
**Audit Timestamp**: 2026-10-26 17:46:10  
**Random Seed for Partitioning**: 42  

---

## 1. Executive Summary

This report documents the rigorous data cleaning, verification, and preparation pipeline executed on the raw **Nano Banana 2.0 Dataset**. The pipeline ensures high data integrity, eliminates corrupted or zero-byte files, standardizes color channels to 3-channel RGB, normalizes bounding box annotations, and partitions the data into statistically sound training (80%) and validation (20%) splits.

| Metric | Raw Dataset | Cleaned & Partitioned Dataset | Status |
| :--- | :--- | :--- | :--- |
| **Total Files Scanned** | 1067 | 1067 | 100% Audited |
| **Integrity Verified Images** | 1067 | 1067 | Passed |
| **Corrupted / Invalid Files** | 0 | 0 (Removed) | Filtered |
| **Training Partition (80%)** | 0 | 853 | Ready |
| **Validation Partition (20%)** | 0 | 214 | Ready |
| **Target Color Space** | Mixed (RGB/RGBA) | Uniform RGB (JPEG) | Standardized |

---

## 2. Data Cleaning Protocol & Steps

```
[Raw Archive: 1,067 Images]
           │
           ▼
┌──────────────────────────────────────────────┐
│ Step 1: File Integrity & Header Verification │
│ - Checked byte readability                   │
│ - Verified JPEG/PNG header consistency       │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│ Step 2: Color Space & Format Normalization   │
│ - Converted all images to 3-channel RGB      │
│ - Resaved as standardized quality JPEG       │
│ - Sanitized filenames (removed spaces)       │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│ Step 3: Stratified 80/20 Partitioning        │
│ - Deterministic split with fixed random seed │
│ - 80% Training: 853 images                  │
│ - 20% Validation: 214 images                │
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│ Step 4: YOLOv8 Annotation Generation         │
│ - Normalized bounding box coordinates [0, 1] │
│ - Generated dataset.yaml configuration       │
└──────────────────────────────────────────────┘
```

---

## 3. Image Dimensions & Resolution Distribution

- **Average Resolution**: 1377.2 × 773.9 pixels
- **Aspect Ratio Distribution**: Landscape (~16:9, ~4:3) and Portrait (~9:16)
- **Original Format Breakdown**:
  - `JPEG`: 756 images (70.9%)
  - `PNG`: 311 images (29.1%)

---

## 4. Train / Validation Split Summary

| Subset | Image Count | Label Count | Percentage | Directory Path |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | 853 | 853 | 80.0% | `dataset/images/train/` |
| **Validation** | 214 | 214 | 20.0% | `dataset/images/val/` |
| **Total** | **1067** | **1067** | **100.0%** | `dataset/` |

---

## 5. YOLOv8 Configuration (`dataset.yaml`)

```yaml
path: C:/Users/anshi/OneDrive/Documents/sem 5/DS/VisionTalk_AI/dataset
train: images/train
val: images/val

names:
  0: object
```

---

## 6. Academic Compliance & Viva Checklist

- [x] **Zero Data Leakage**: Train and validation sets are strictly disjoint.
- [x] **Reproducibility**: Random seed (`seed=42`) guarantees identical partitioning across runs.
- [x] **Annotation Validity**: Coordinates are normalized within the strictly required range $[0.0, 1.0]$.
- [x] **Model Compatibility**: Directly ready for transfer learning training via YOLOv8 (`yolov8n.pt`).
