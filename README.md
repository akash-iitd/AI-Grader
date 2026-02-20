<div align="center">

# 📝 AI Grading Assistant

**An intelligent answer sheet evaluation system for Indian schools (CBSE & CISCE) powered by Google Gemini Vision**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

</div>

---

## 📖 Project Overview

AI Grading Assistant is a production-grade tool that automates the evaluation of handwritten answer sheets for **CBSE and CISCE school examinations**. Teachers upload scanned answer sheets (images or PDFs), provide a marking scheme, and the system returns a detailed, step-by-step evaluation with marks, transcription, and constructive feedback — all in seconds.

The system leverages **Google Gemini 2.5 Flash** as a multimodal LLM to simultaneously read handwriting, interpret mathematical/scientific notation, evaluate diagrams, and cross-reference answers against a structured rubric.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit (Tabbed UI with custom CSS) |
| **AI Engine** | Google Gemini 2.5 Flash (Multimodal Vision + JSON mode) |
| **Image Processing** | OpenCV (Bilateral filtering, CLAHE, adaptive thresholding) |
| **PDF Handling** | PyMuPDF / fitz (300 DPI rendering for scanned documents) |
| **Report Generation** | ReportLab (PDF reports) + Pandas (CSV export) |
| **Image Library** | Pillow (PIL) |

---

## ✨ Core Features

- **🔍 Multimodal Answer Evaluation** — Gemini Vision reads handwritten text, mathematical expressions (LaTeX), diagrams, and graphs directly from answer sheet images.
- **📊 Step-by-Step Marking** — Each question is evaluated against the rubric with per-step mark allocation, partial credit logic, and detailed reasoning.
- **🤖 Auto-Rubric Generation** — Upload a teacher's answer key and the AI automatically generates a structured marking scheme with mark allocations.
- **📄 PDF & Image Support** — Handles JPG, PNG, and scanned PDFs. PDFs are rendered at 300 DPI for optimal handwriting recognition.
- **🖼️ Advanced Image Preprocessing** — OpenCV pipeline with bilateral filtering (bleed-through suppression), CLAHE contrast enhancement, and adaptive thresholding to clean up scanned documents.
- **🔄 Batched Evaluation with Overlap** — Large answer sheets are split into overlapping batches (stride-based) to ensure questions spanning page boundaries are fully captured. Duplicate evaluations are resolved by keeping the higher-quality response.
- **✏️ Teacher Override** — Per-question mark adjustment with real-time total recalculation before final export.
- **📥 Export Reports** — Download detailed evaluation reports as PDF (styled with ReportLab) or CSV (via Pandas).
- **🛡️ Robust JSON Parsing** — Multi-layer JSON repair pipeline handles markdown fences, truncated responses, trailing commas, and unescaped characters from AI output.
- **📐 Diagram & Figure Evaluation** — Dedicated evaluation logic for physics circuits, biology diagrams, chemistry apparatus, and geometry constructions.

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9+
- A [Google AI Studio API Key](https://aistudio.google.com/app/apikey) (free tier available)

### 1. Clone the Repository
```bash
git clone https://github.com/akash-iitd/AI-Grader.git
cd AI-Grader
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Usage
1. Enter your Google API Key in the sidebar
2. Upload answer sheet image(s) or scanned PDF(s)
3. Provide a marking scheme (manual or auto-generated)
4. Click **"Evaluate Answer"**
5. Review results, adjust marks if needed, and download the report

---

## 📁 Project Structure

```
AI-Grader/
├── app.py              # Full application — UI, AI evaluation, image processing, report generation
└── requirements.txt    # Python dependencies
```

---

## 🏗️ Architecture

```
Answer Sheet (Image/PDF)
        ↓
  PyMuPDF (300 DPI render) → PIL Image
        ↓
  OpenCV Preprocessing (Denoise → CLAHE → Adaptive Threshold → Morphological Clean)
        ↓
  Batch Splitter (Overlapping stride-based batches)
        ↓
  Gemini 2.5 Flash Vision API (JSON mode, structured system prompt)
        ↓
  JSON Repair Pipeline → Merge & Deduplicate Batches
        ↓
  Streamlit Results UI (Per-question breakdown + Teacher overrides)
        ↓
  ReportLab PDF / Pandas CSV Export
```

---

<div align="center">
  <sub>Made with ❤️ for Indian Educators | CBSE & CISCE Compatible</sub>
</div>
