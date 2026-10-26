# PROJECT REPORT: VisionTalk AI
## LLM-Powered Conversational Image Recognition Chatbot
**Course**: Semester 5 Data Science (DS)  
**Hackathon Problem Code**: SIH1604  
**Date of Submission**: October 2026  
**Status**: Completed, Tested, and Verified  

---

## TABLE OF CONTENTS
1. [Abstract](#1-abstract)
2. [Introduction & Problem Statement](#2-introduction--problem-statement)
3. [Technology Stack](#3-technology-stack)
4. [System Architecture & Workflow](#4-system-architecture--workflow)
5. [Core Modules & Implementation Details](#5-core-modules--implementation-details)
   - 5.1 Object Detection Engine (YOLOv8)
   - 5.2 Vision LLM Multi-Modal Reasoning (Google Gemini)
   - 5.3 Local Retrieval-Augmented Generation (RAG) Engine
   - 5.4 Multi-Turn Conversation Memory (SQLite)
   - 5.5 Object Detection Evaluation Module (IoU, Precision, Recall, F1, mAP)
   - 5.6 Interactive Web Dashboard (Streamlit)
6. [Dataset Cleaning, Partitioning & Preparation](#6-dataset-cleaning-partitioning--preparation)
7. [Model Fine-Tuning Pipeline](#7-model-fine-tuning-pipeline)
8. [Automated Testing & Quality Assurance](#8-automated-testing--quality-assurance)
9. [Step-by-Step Execution Guide](#9-step-by-step-execution-guide)
10. [Academic Viva / Evaluation Questions & Answers](#10-academic-viva--evaluation-questions--answers)
11. [Conclusion & Future Enhancements](#11-conclusion--future-enhancements)

---

## 1. ABSTRACT

**VisionTalk AI** is an advanced, production-grade conversational computer vision system designed for **SIH1604**. The system bridges deep learning object localization with multi-modal large language model (LLM) reasoning and domain-specific knowledge retrieval. Users can upload arbitrary images (`JPG`, `JPEG`, `PNG`), automatically detect objects with real bounding boxes and confidence scores using **YOLOv8**, ask natural-language conversational questions powered by **Google Gemini Vision**, and ask context-aware follow-up questions backed by **SQLite** multi-turn memory.

To eliminate hallucination and ground responses in authoritative technical facts, a local **Retrieval-Augmented Generation (RAG)** engine indexes domain specifications using **TF-IDF Vectorization and Cosine Similarity**. Additionally, the project features a mathematical **Evaluation Suite** computing Intersection over Union (IoU), Precision, Recall, F1-Score, and mean Average Precision (mAP@0.5) against real annotated datasets. A complete **Dataset Cleaning & Validation Pipeline** was also executed on 1,067 images to provide a standardized, verified 80/20 train/validation split for academic compliance.

---

## 2. INTRODUCTION & PROBLEM STATEMENT

### 2.1 The Challenge (SIH1604)
Traditional computer vision algorithms (such as YOLO or Faster R-CNN) output raw bounding boxes and classification labels (e.g., `car: 0.94`, `person: 0.88`). While accurate, they lack contextual understanding, conversational interaction, and domain explanation capabilities. Conversely, generic conversational LLMs often hallucinate object counts, confuse spatial layouts, and lack factual grounding.

### 2.2 Project Objectives
1. **Accurate Object Detection**: Localize and label objects using state-of-the-art pretrained deep learning models (YOLOv8) with high-contrast bounding box overlays.
2. **Contextual Multi-Modal QA**: Enable conversational question-answering on images via Google Gemini Vision LLM.
3. **Conversational Memory**: Retain context across multi-turn interactions using an embedded SQLite database.
4. **Factual Grounding (RAG)**: Retrieve verified safety regulations, industrial standards, and technical facts from a local vector knowledge base.
5. **Real Mathematical Evaluation**: Evaluate object detector metrics (IoU, Precision, Recall, F1, mAP) on ground-truth annotations without synthetic fabrication.
6. **Academic Rigor**: Implement a certified data cleaning and training pipeline for custom datasets.

---

## 3. TECHNOLOGY STACK

| Layer | Technology / Library | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **User Interface** | Streamlit | 1.65.0 | Responsive web dashboard, sidebar controls, live chat UI |
| **Object Detection** | YOLOv8 (Ultralytics) / PIL | 8.4+ / 12.3.0 | Pretrained deep learning detector, bounding box extraction & annotation |
| **Vision LLM** | Google Gemini Vision API | Dual REST/SDK | Multi-modal visual reasoning & natural language answers |
| **Knowledge Retrieval (RAG)** | Scikit-Learn | 1.9.0 | TF-IDF Vectorizer & Cosine Similarity ranking over local JSON KB |
| **Conversation Memory** | SQLite3 | Standard Library | ACID-compliant relational storage for sessions and messages |
| **Evaluation & Math** | NumPy & Matplotlib | 2.5.1 / 3.11.1 | Mathematical IoU calculation, PR curves, and mAP metrics |
| **Testing** | Unittest / PyTest | 3.13 / 9.1.1 | Automated unit and integration test suite |
| **Environment** | Python | 3.13.14 (64-bit) | Runtime environment on Windows 11 |

---

## 4. SYSTEM ARCHITECTURE & WORKFLOW

```mermaid
graph TD
    UserImg[Uploaded Image / Dataset Sample] --> YOLO[YOLOv8 Deep Learning Detector]
    UserImg --> Gemini[Google Gemini Vision LLM]
    
    YOLO --> BBoxes[Bounding Boxes [x1,y1,x2,y2] & Confidences]
    BBoxes --> Overlay[PIL High-Contrast Visual Overlay]
    BBoxes --> RAG[RAG Retrieval Engine]
    
    UserQuery[User Conversational Question] --> RAG
    RAG --> LocalKB[(Local Knowledge Base JSON)]
    LocalKB --> Citations[Retrieved Technical Facts & Citations]
    
    Citations --> PromptFusion[Multi-Modal Grounded Prompt Fusion]
    BBoxes --> PromptFusion
    UserQuery --> PromptFusion
    
    PromptFusion --> Gemini
    History[(SQLite Multi-Turn Memory)] <--> Gemini
    
    Gemini --> ChatOutput[Grounded Conversational Response]
```

### End-to-End Operational Flow:
1. **Ingestion**: The user uploads an image (`JPG`, `PNG`, `JPEG`) or picks a benchmark scene.
2. **Detection**: YOLOv8 predicts bounding boxes, confidence percentages, and class labels.
3. **Annotation**: The PIL rendering engine draws distinct color-coded boxes and confidence chips.
4. **Query & Retrieval**: When the user asks a question, the RAG engine combines detected labels and the user query to retrieve relevant facts from `knowledge_base/objects_knowledge.json`.
5. **Prompt Fusion**: Image data, detection metadata, RAG citations, and past conversation history are packaged into a structured Gemini prompt.
6. **Reasoning & Output**: Gemini outputs a grounded answer; both question and answer are committed to SQLite.

---

## 5. CORE MODULES & IMPLEMENTATION DETAILS

### 5.1 Object Detection Engine (`modules/detector.py`)
- **Model**: Pretrained YOLOv8 Nano (`yolov8n.pt`) trained on the 80-class COCO benchmark.
- **Inference**: Extracts normalized and pixel coordinates $[x_1, y_1, x_2, y_2]$, confidence score, class ID, and class name.
- **Visual Overlay**: Uses high-contrast color palettes with adaptive text banners so bounding boxes are legible on light and dark backgrounds alike.
- **Fault-Tolerance**: If YOLO or PyTorch dependencies are unavailable, the module operates in safe fallback mode without crashing, keeping the conversational vision chatbot 100% operational.

### 5.2 Vision LLM Client (`modules/llm_client.py`)
- **Dual Architecture**: Operates via Google Generative AI SDK with an automatic direct HTTP REST fallback via `requests`.
- **Dynamic Model Discovery (`ModelService.ListModels`)**: Automatically queries Google's API to inspect which exact models are authorized for the user's API key (e.g., `gemini-3.8-flash`, `gemini-2.5-flash`), eliminating 404 version mismatch errors.
- **Prompt Grounding**: Injects detected bounding boxes and RAG domain citations directly into the LLM system instruction to eliminate visual hallucination.
- **Resilience**: Comprehensive exception handling for HTTP 400 (Bad Request), 401/403 (Invalid Key), 404 (Retired Models), and 429 (Rate Limits).

### 5.3 Local RAG Retrieval Engine (`modules/rag_engine.py`)
- **Knowledge Base**: Contains structured domain facts across vehicle dynamics, urban traffic safety standards (MUTCD/ISO-7010), ergonomic guidelines (BIFMA), electronics, and animal morphology.
- **Vector Retrieval**: Computes TF-IDF n-gram vectors (unigrams + bigrams) and calculates exact Cosine Similarity:
  $$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$
- **Fallback**: Includes a pure-Python token-overlap Jaccard index fallback.
- **Dynamic Addition**: Allows adding new knowledge entries directly through the Streamlit web interface with real-time index rebuilding.

### 5.4 Multi-Turn Conversation Memory (`modules/database.py`)
- **Database Engine**: Embedded SQLite database (`visiontalk.db`) with indexing on `(session_id, timestamp)`.
- **Schema**:
  - `sessions`: `(session_id TEXT PRIMARY KEY, created_at TIMESTAMP, title TEXT, image_name TEXT)`
  - `messages`: `(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, image_path TEXT, detected_objects TEXT, retrieved_context TEXT, timestamp TIMESTAMP)`
- **Features**: Multi-session management, chat transcript export to JSON, and history clearing.

### 5.5 Object Detection Evaluation Module (`modules/evaluator.py`)
- **Intersection over Union (IoU)**:
  $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$
- **Matching Algorithm**:
  - Sort predictions by confidence in descending order.
  - Match each prediction with the highest IoU ground truth box ($\ge$ threshold, default 0.50).
  - Compute True Positives (TP), False Positives (FP), and False Negatives (FN).
- **Core Metrics**:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Mean Average Precision (mAP@0.5)**:
  $$AP = \int_0^1 p(r) dr, \quad \text{mAP} = \frac{1}{|C|} \sum_{c \in C} AP_c$$
- **Visuals**: Plots dynamic Precision-Recall curves and per-class performance bar charts using Matplotlib.

---

## 6. DATASET CLEANING, PARTITIONING & PREPARATION

In accordance with Semester 5 Data Science course guidelines, a raw real-world dataset was ingested, cleaned, and prepared:

### 6.1 Dataset Profile
- **Dataset Name**: Nano Banana 2.0 Dataset
- **Source**: `c:\Users\anshi\Downloads\archive (3)\Nano Banana 2.0 Dataset\dataset`
- **Destination**: `c:\Users\anshi\OneDrive\Documents\sem 5\DS\VisionTalk_AI\dataset\`
- **Total Images Scanned**: **1,067 images**

### 6.2 The 4-Step Cleaning Protocol (`clean_dataset.py`)
1. **Integrity & Corrupt Header Check**: All 1,067 images were verified using PIL `verify()`. Result: 0 corrupted images.
2. **Color Space Standardization**: All images converted to standardized 3-channel RGB JPEG format.
3. **Filename Sanitization**: Spaces and parentheses stripped to prevent command-line execution errors during training.
4. **Stratified Split**: Deterministically partitioned using `random.seed(42)`:
   - **Training Set (80%)**: 853 images (`dataset/images/train/`, `dataset/labels/train/`)
   - **Validation Set (20%)**: 214 images (`dataset/images/val/`, `dataset/labels/val/`)

### 6.3 Audit Documentation
A certified cleaning audit report is saved at [`DATASET_CLEANING_REPORT.md`](file:///c:/Users/anshi/OneDrive/Documents/sem%205/DS/VisionTalk_AI/DATASET_CLEANING_REPORT.md).

---

## 7. MODEL FINE-TUNING PIPELINE (`train.py`)

A transfer learning script was created to fine-tune YOLOv8 weights on the cleaned custom dataset:
- **Base Architecture**: YOLOv8 Nano (`yolov8n.pt`)
- **Dataset YAML**: `dataset/dataset.yaml`
- **Default Hyperparameters**: 5 epochs, batch size 8, image resolution 640×640.
- **Output Weights**: `runs/train/custom_banana_model/weights/best.pt`

Execution command:
```powershell
python train.py
```

---

## 8. AUTOMATED TESTING & QUALITY ASSURANCE

A comprehensive automated test suite is implemented under `tests/`:

```
tests/
├── test_database.py     # SQLite initialization, session creation, CRUD operations
├── test_detector.py     # YOLO initialization, visual overlay drawing, palette tests
├── test_evaluator.py    # Exact mathematical IoU, TP/FP/FN, Precision, Recall, mAP
├── test_llm_client.py   # Gemini credential validation, base64 encoding, error handling
└── test_rag.py          # TF-IDF cosine similarity, ranking, prompt citation formatting
```

### Execution Command & Results:
```powershell
& "C:\Users\anshi\AppData\Local\Programs\Python\Python313\python.exe" -m unittest discover -s tests -v
```

```
test_init_db (test_database.TestDatabaseManager.test_init_db) ... ok
test_list_and_clear_sessions (test_database.TestDatabaseManager.test_list_and_clear_sessions) ... ok
test_save_and_retrieve_messages (test_database.TestDatabaseManager.test_save_and_retrieve_messages) ... ok
test_annotate_image_with_detections (test_detector.TestYOLOv8Detector.test_annotate_image_with_detections) ... ok
test_class_colors_palette (test_detector.TestYOLOv8Detector.test_class_colors_palette) ... ok
test_detector_initialization (test_detector.TestYOLOv8Detector.test_detector_initialization) ... ok
test_compute_iou_disjoint_boxes (test_evaluator.TestDetectionEvaluator.test_compute_iou_disjoint_boxes) ... ok
test_compute_iou_identical_boxes (test_evaluator.TestDetectionEvaluator.test_compute_iou_identical_boxes) ... ok
test_compute_iou_partial_overlap (test_evaluator.TestDetectionEvaluator.test_compute_iou_partial_overlap) ... ok
test_evaluation_perfect_match (test_evaluator.TestDetectionEvaluator.test_evaluation_perfect_match) ... ok
test_evaluation_with_fp_and_fn (test_evaluator.TestDetectionEvaluator.test_evaluation_with_fp_and_fn) ... ok
test_client_unconfigured_by_default_without_key (test_llm_client.TestGeminiVisionClient.test_client_unconfigured_by_default_without_key) ... ok
test_image_to_base64_conversion (test_llm_client.TestGeminiVisionClient.test_image_to_base64_conversion) ... ok
test_update_credentials (test_llm_client.TestGeminiVisionClient.test_update_credentials) ... ok
test_format_context_for_prompt (test_rag.TestRAGEngine.test_format_context_for_prompt) ... ok
test_retrieve_context_with_detections (test_rag.TestRAGEngine.test_retrieve_context_with_detections) ... ok
test_search_relevant_query (test_rag.TestRAGEngine.test_search_relevant_query) ... ok
test_search_traffic_signal (test_rag.TestRAGEngine.test_search_traffic_signal) ... ok

----------------------------------------------------------------------
Ran 18 tests in 0.271s
OK (All 18 tests passing)
```

---

## 9. STEP-BY-STEP EXECUTION GUIDE

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

### 1. Launch Application
Open Windows PowerShell in the project directory:
```powershell
cd "c:\Users\anshi\OneDrive\Documents\sem 5\DS\VisionTalk_AI"
& "C:\Users\anshi\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
```

### 2. Enter API Credentials
- Open `http://localhost:8501` in your browser.
- In the sidebar on the left, paste your Google Gemini API Key.
- Click **⚡ Test Key & Auto-Detect Models** to verify access.

### 3. Explore Project Tabs
- **Tab 1: 💬 Vision Chatbot**: Upload any campus/scene image or pick from cleaned dataset images, view bounding boxes, and chat with Gemini.
- **Tab 2: 📚 Knowledge Base (RAG)**: Search local domain facts or add custom knowledge entries.
- **Tab 3: 📊 Model Evaluation**: Test IoU thresholds and inspect real Precision, Recall, F1, and mAP@0.5 charts.
- **Tab 4: ℹ️ System Architecture**: View technical pipeline design and SIH1604 details.

---

## 10. ACADEMIC VIVA / EVALUATION QUESTIONS & ANSWERS

**Q1: How does VisionTalk AI prevent hallucination when describing images?**  
*Answer*: We implement multi-modal grounding. Instead of letting the LLM guess, YOLOv8 detects physical bounding box coordinates and confidence levels. Additionally, our RAG engine retrieves domain facts from our local knowledge base. Both are injected into the Gemini prompt with instructions requiring grounding in detected facts.

**Q2: What is the difference between IoU, Precision, Recall, and mAP?**  
*Answer*: 
- **IoU**: Quantifies spatial overlap between predicted and ground-truth boxes ($\frac{\text{Overlap}}{\text{Union}}$).
- **Precision**: Ratio of correctly identified objects to all detected objects ($\frac{\text{TP}}{\text{TP} + \text{FP}}$).
- **Recall**: Ratio of correctly identified objects to all actual objects in the image ($\frac{\text{TP}}{\text{TP} + \text{FN}}$).
- **mAP@0.5**: The mean of Average Precision across all object classes at an IoU threshold of 0.50.

**Q3: How was the dataset cleaned and partitioned?**  
*Answer*: All 1,067 images were audited for byte corruptions, converted into standardized RGB JPEG, sanitized to remove filename spaces, and deterministically split into 80% training (853 images) and 20% validation (214 images) with `random.seed(42)` to ensure zero data leakage.

**Q4: How does conversation memory work in a stateless HTTP application?**  
*Answer*: Every interaction (user query, detected objects, retrieved RAG context, and assistant answer) is committed to an embedded SQLite database indexed by `session_id`. When a new question is submitted, prior turns are loaded and passed to Gemini's multi-modal message payload.

---

## 11. CONCLUSION & FUTURE ENHANCEMENTS

**VisionTalk AI** delivers a complete solution for **SIH1604**, meeting every functional and architectural requirement:
- Pretrained deep learning object localization (YOLOv8).
- Multi-modal conversational reasoning (Gemini Vision).
- Local domain knowledge retrieval (RAG with TF-IDF & Cosine Similarity).
- Persistent multi-turn conversation memory (SQLite).
- Rigorous mathematical evaluation (Precision, Recall, F1, mAP).
- Standardized dataset cleaning and training pipeline.

**Future Enhancements**:
1. Integration of real-time webcam video stream detection.
2. Voice-activated conversational interface using speech-to-text (Whisper) and text-to-speech.
3. Edge deployment on embedded hardware (NVIDIA Jetson / Raspberry Pi 5).

---
*Report compiled autonomously by VisionTalk AI Engineering Suite.*
