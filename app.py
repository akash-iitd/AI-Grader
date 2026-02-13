"""
AI Grading Assistant for Indian Schools (CBSE/CISCE)
A Streamlit-based MVP that uses Google Gemini 1.5 Flash to evaluate handwritten answer sheets.
"""

import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import io
import os
import re
import tempfile
import fitz  # PyMuPDF - for PDF to image conversion
import json
import re
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Grading Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR PREMIUM UI
# ============================================
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        color: #333333 !important;
    }
    
    .info-card * {
        color: #333333 !important;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        color: #333333 !important;
    }
    
    .result-card * {
        color: #333333 !important;
    }
    
    /* Warning card */
    .warning-card {
        background: linear-gradient(145deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f39c12;
        margin: 1rem 0;
    }
    
    /* Success card */
    .success-card {
        background: linear-gradient(145deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    /* Marks display */
    .marks-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 12px;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e9ecef;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Step indicator */
    .step-indicator {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        font-weight: 700;
        margin-right: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None
if 'final_marks' not in st.session_state:
    st.session_state.final_marks = None
if 'max_marks' not in st.session_state:
    st.session_state.max_marks = 10
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_images' not in st.session_state:
    st.session_state.original_images = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'question_overrides' not in st.session_state:
    st.session_state.question_overrides = {}
if 'generated_rubric' not in st.session_state:
    st.session_state.generated_rubric = None

# ============================================
# SYSTEM PROMPT FOR GEMINI
# ============================================
SYSTEM_PROMPT = """You are a Strict CBSE/CISCE Evaluator API with 20+ years of experience evaluating student answer sheets in Indian schools.

CRITICAL: You are an API endpoint. You MUST output ONLY valid JSON. Do NOT include any conversational text, greetings, explanations, or markdown formatting outside the JSON object. Your entire response must be a single valid JSON object.

## MULTI-QUESTION EVALUATION PROCESS:

1. **Question Identification**:
   - Scan the ENTIRE image(s) and list ALL distinct questions found
   - Match question identifiers flexibly ("1.", "Q1", "Ans 1", "1)" are all treated as Question 1)
   - If a question spans multiple pages, evaluate it as one complete answer

2. **Marking Scheme Cross-Reference**:
   - Map each identified question to the corresponding header (## Q1, ## Q2) in the marking scheme
   - If a marking scheme is missing for a detected question, provide transcription but mark as "N/A (Marking Scheme Missing)"
   - If a question exists in the marking scheme but NOT in the student's answer, mark as "Not Attempted" with 0 marks

3. **Evaluation Rules**:
   - Award marks for each correct step as defined in the marking scheme
   - If the final answer is wrong but intermediate steps are correct, award partial marks
   - If handwriting is unclear, mention it but try your best to interpret
   - Be fair but strict - only award marks for demonstrated understanding

4. **Formatting for Math & Science**:
   - You MUST use LaTeX for all mathematical expressions, chemical formulas, subscripts, and superscripts
   - Example: Use $H_{2}O$ instead of H2O. Use $x^{2}$ instead of x^2 or x2
   - For chemical equations, use the $\\rightarrow$ symbol
   - Check for units ($cm$, $m/s$, $\\Omega$). Deduct marks only if marking scheme specifies

5. **Step-wise Logic Check**:
   - For Math: Check if intermediate calculation leads to next step, even with calculation errors
   - For Physics/Chemistry: Verify dimensional consistency and unit usage
   - If student uses a valid alternative method, award full marks if logic is sound

6. **Indian Context Edge Cases**:
   - If you see "double-written" or overwritten characters, flag confidence as "low"
   - Ignore text clearly marked as "Rough Work" or in side-margin columns
   - For ambiguous characters (1/l, 0/O), use marking scheme context to determine intent

7. **Diagram & Figure Evaluation**:
   - CAREFULLY examine any diagrams, figures, graphs, or sketches in the answer
   - Set "has_diagram" to true and describe the diagram for any question with a visual element
   
   **Physics Diagrams:**
   - Circuit diagrams: Check correct symbols, connections (series/parallel), labeling ($R$, $V$, $I$)
   - Ray diagrams: Check arrow directions, labeling of focal point ($F$), centre of curvature ($C$)
   - Force/free-body diagrams: Check direction and labeling of forces ($mg$, $N$, $f$, $T$)
   
   **Biology Diagrams:**
   - Check correct structure, verify proper labeling of ALL parts per marking scheme
   
   **Chemistry Diagrams:**
   - Apparatus: Check correct setup, labeling of chemicals and equipment
   - Molecular structures: Verify correct bonds, atoms, structural formulas
   
   **Mathematics/Geometry:**
   - Constructions: Check arcs, bisectors, correct measurements
   - Graphs: Check axes labels, scale, plotted points, line/curve
   
   **General Diagram Rules:**
   - Award marks for each correctly labeled part per marking scheme
   - Unlabeled but correct diagram = partial marks for structure only
   - Missing required diagram = note in feedback, deduct accordingly
   - Rough but identifiable drawing = award marks (accuracy > neatness)

IMPORTANT: Follow the marking scheme EXACTLY. Provide constructive feedback.

You MUST respond with ONLY this JSON structure (no other text):
{
    "total_score": <number>,
    "max_total": <number>,
    "confidence": "high" | "medium" | "low",
    "overall_feedback": "Brief overall feedback",
    "warnings": ["list of warnings"],
    "evaluations": [
        {
            "q_no": "Q1",
            "question_text": "Brief description",
            "status": "Attempted" | "Not Attempted" | "Partially Visible" | "Missing Scheme",
            "transcription": "Full transcription with LaTeX",
            "has_diagram": false,
            "diagram_description": "",
            "marks": <number>,
            "max_marks": <number>,
            "step_breakdown": [
                {"step": "Step description", "marks": <number>, "reason": "Why awarded/deducted"}
            ],
            "feedback": "Specific feedback"
        }
    ]
}"""

# ============================================
# HELPER FUNCTIONS
# ============================================

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the uploaded image for better OCR results.
    Enhanced to suppress bleed-through from back of paper while
    preserving text (including small subscripts/superscripts).
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if image is colored
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Bilateral filter: smooths out faint bleed-through while preserving sharp text edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(denoised)
    
    # Adaptive thresholding with larger block size (21 vs 11) and higher
    # constant (5 vs 2) to aggressively suppress faint bleed-through marks
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 5
    )
    
    # Morphological opening to remove small noise spots (bleed-through remnants)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Weight heavily toward the clean binary result (70%) to minimize bleed-through
    enhanced = cv2.addWeighted(clahe_enhanced, 0.3, cleaned, 0.7, 0)
    
    # Convert back to PIL Image
    return Image.fromarray(enhanced)


def pdf_to_images(uploaded_file) -> list:
    """
    Convert an uploaded PDF file into a list of PIL Images (one per page).
    Renders at 300 DPI for high-quality handwriting recognition from scanned docs.
    """
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        # Render page at 300 DPI for best OCR quality
        zoom = 300 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def process_uploaded_files(uploaded_files: list) -> list:
    """
    Process a list of uploaded files (images and/or PDFs).
    PDFs are converted to images (one per page). Returns a flat list of PIL Images.
    """
    all_images = []
    for f in uploaded_files:
        if f.name.lower().endswith('.pdf'):
            all_images.extend(pdf_to_images(f))
        else:
            all_images.append(Image.open(f))
    return all_images


def extract_and_repair_json(text: str) -> dict:
    """
    Robustly extract and repair JSON from AI response text.
    Handles: markdown fences, extra text before/after JSON, trailing commas,
    truncated responses, and other common AI output issues.
    """
    # Step 1: Strip markdown code fences
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        parts = text.split('```')
        if len(parts) >= 3:
            text = parts[1].strip()
    
    # Step 2: Find the outermost { ... } pair
    first_brace = text.find('{')
    if first_brace == -1:
        raise json.JSONDecodeError("No JSON object found in response", text, 0)
    
    # Find matching closing brace by counting depth
    depth = 0
    last_brace = -1
    in_string = False
    escape_next = False
    for i in range(first_brace, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\':
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                last_brace = i
                break
    
    if last_brace == -1:
        # Truncated response - try to close open structures
        json_text = text[first_brace:]
        # Close any open arrays and objects
        open_brackets = json_text.count('[') - json_text.count(']')
        open_braces = json_text.count('{') - json_text.count('}')
        json_text += ']' * max(0, open_brackets)
        json_text += '}' * max(0, open_braces)
    else:
        json_text = text[first_brace:last_brace + 1]
    
    # Step 3: Fix common issues
    # Remove trailing commas before ] or }
    json_text = re.sub(r',\s*([\]\}])', r'\1', json_text)
    
    # Try parsing
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Last resort: try fixing unescaped newlines in strings
        json_text = json_text.replace('\n', '\\n')
        return json.loads(json_text)


def evaluate_batch(api_key: str, images: list, marking_scheme: str, is_partial: bool = False, max_retries: int = 3) -> dict:
    """
    Evaluate a batch of answer sheet images against the marking scheme.
    Uses response_mime_type to force valid JSON output.
    
    Args:
        images: List of PIL Images for this batch
        marking_scheme: The full marking scheme text
        is_partial: True if this is one of multiple batches (adjusts prompt)
    """
    import time
    
    for attempt in range(max_retries):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Build prompt - adjust for partial vs full evaluation
            if is_partial:
                user_prompt = f"""MARKING SCHEME (Markdown Format):
{marking_scheme}

INSTRUCTIONS:
1. These are SOME pages from a larger answer sheet. Evaluate ONLY the questions visible in these specific images.
2. For each question found, match it to the corresponding ## QX section in the marking scheme.
3. Evaluate each question with step-by-step breakdown.
4. Use LaTeX for all math/science notation ($x^2$, $H_2O$, etc.).
5. If a question appears to continue from a previous page and you cannot see the beginning, evaluate only what is visible and note it.
6. Do NOT include questions that are not visible in these images.
7. Respond with ONLY the JSON object. No other text."""
            else:
                user_prompt = f"""MARKING SCHEME (Markdown Format):
{marking_scheme}

INSTRUCTIONS:
1. Scan ALL image(s) and identify EVERY question present (look for Q1, 1., Ans 1, etc.)
2. For EACH question found, match it to the corresponding ## QX section in the marking scheme.
3. Evaluate each question separately with step-by-step breakdown.
4. Use LaTeX for all math/science notation ($x^2$, $H_2O$, etc.).
5. If a question spans multiple pages, combine the answer.
6. If marking scheme is missing for a question, transcribe but mark as "Missing Scheme".
7. If a question in the scheme is not attempted, mark as "Not Attempted" with 0 marks.
8. Respond with ONLY the JSON object. No other text."""
            
            # Build content parts: system prompt + user prompt + images
            content_parts = [SYSTEM_PROMPT, user_prompt]
            content_parts.extend(images)
            
            # Use response_mime_type to force JSON output
            response = model.generate_content(
                content_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                )
            )
            
            # Parse JSON response with robust repair
            response_text = response.text.strip()
            result = extract_and_repair_json(response_text)
            
            # Validate required fields
            if 'evaluations' not in result:
                result['evaluations'] = []
            if 'total_score' not in result:
                result['total_score'] = sum(e.get('marks', 0) for e in result['evaluations'])
            if 'max_total' not in result:
                result['max_total'] = sum(e.get('max_marks', 0) for e in result['evaluations'])
            
            return result
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue  # Retry on JSON parse failure
            return {
                "error": True,
                "message": f"Failed to parse AI response after {max_retries} attempts. The AI returned invalid JSON.",
                "raw_response": response_text if 'response_text' in dir() else str(e)
            }
        except Exception as e:
            error_msg = str(e).lower()
            
            if "quota" in error_msg or "rate" in error_msg or "resource" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "error": True,
                        "message": f"API rate limit exceeded after {max_retries} attempts. Please wait 1-2 minutes and try again."
                    }
            elif "invalid" in error_msg and "key" in error_msg:
                return {
                    "error": True,
                    "message": "Invalid API key. Please check your Google API key in the sidebar."
                }
            else:
                return {
                    "error": True,
                    "message": f"An error occurred: {str(e)}"
                }
    
    return {"error": True, "message": "Maximum retries exceeded. Please try again later."}


def _eval_quality_score(eval_item: dict) -> int:
    """
    Compute a quality score for an evaluation to decide which duplicate to keep.
    Higher = more complete/detailed evaluation.
    """
    score = 0
    score += len(eval_item.get("transcription", ""))           # longer transcription = more complete
    score += len(eval_item.get("step_breakdown", [])) * 100   # more steps = more detailed
    score += len(eval_item.get("feedback", ""))                # longer feedback = richer
    if eval_item.get("status") == "Attempted":
        score += 200  # prefer "Attempted" over "Partially Visible"
    return score


def merge_evaluation_results(batch_results: list) -> dict:
    """
    Merge multiple batch evaluation results into one combined result.
    When overlapping batches produce the same question, keeps the
    more complete evaluation (longer transcription, more steps).
    """
    merged = {
        "total_score": 0,
        "max_total": 0,
        "evaluations": [],
        "confidence": "high",
        "overall_feedback": "",
        "warnings": []
    }
    
    # q_no -> (index in merged["evaluations"], quality score)
    question_index = {}
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    min_confidence = 3
    feedbacks = []
    
    for result in batch_results:
        if result.get("error"):
            continue
        
        for eval_item in result.get("evaluations", []):
            q_no = eval_item.get("q_no", "Q?")
            new_quality = _eval_quality_score(eval_item)
            
            if q_no not in question_index:
                # First time seeing this question — add it
                idx = len(merged["evaluations"])
                merged["evaluations"].append(eval_item)
                question_index[q_no] = (idx, new_quality)
            else:
                # Duplicate from overlapping batch — keep the better one
                existing_idx, existing_quality = question_index[q_no]
                if new_quality > existing_quality:
                    merged["evaluations"][existing_idx] = eval_item
                    question_index[q_no] = (existing_idx, new_quality)
        
        conf = result.get("confidence", "medium")
        min_confidence = min(min_confidence, confidence_rank.get(conf, 2))
        
        fb = result.get("overall_feedback", "")
        if fb:
            feedbacks.append(fb)
        
        merged["warnings"].extend(result.get("warnings", []))
    
    # Recalculate totals from merged evaluations
    merged["total_score"] = sum(e.get("marks", 0) for e in merged["evaluations"])
    merged["max_total"] = sum(e.get("max_marks", 0) for e in merged["evaluations"])
    
    # Set confidence to lowest across batches
    for level, val in confidence_rank.items():
        if val == min_confidence:
            merged["confidence"] = level
            break
    
    merged["overall_feedback"] = " ".join(feedbacks)
    
    # Sort evaluations by question number
    def sort_key(e):
        q = e.get("q_no", "Q0")
        nums = re.findall(r'\d+', q)
        return (int(nums[0]) if nums else 999, q)
    merged["evaluations"].sort(key=sort_key)
    
    return merged


def evaluate_with_gemini(api_key: str, images: list, marking_scheme: str, batch_size: int = 4, progress_callback=None) -> dict:
    """
    Evaluate answer sheet images against the marking scheme.
    Automatically splits large inputs into overlapping batches so that
    questions spanning a page boundary are seen by both adjacent batches.
    
    Overlap strategy: each batch shares 1 image with the next batch.
    Example with 8 images, batch_size=4:
        Batch 1: pages [1, 2, 3, 4]
        Batch 2: pages [4, 5, 6, 7]   ← page 4 overlaps
        Batch 3: pages [7, 8]          ← page 7 overlaps
    Duplicates from overlap are resolved by keeping the more complete evaluation.
    """
    if len(images) <= batch_size:
        # Small enough for a single call
        if progress_callback:
            progress_callback(1, 1)
        return evaluate_batch(api_key, images, marking_scheme, is_partial=False)
    
    # Build overlapping batches (stride = batch_size - 1 for 1-image overlap)
    stride = max(1, batch_size - 1)  # e.g. 4-1 = 3 new images per batch
    batches = []
    for start in range(0, len(images), stride):
        batch = images[start:start + batch_size]
        batches.append(batch)
        # Stop if this batch already includes the last image
        if start + batch_size >= len(images):
            break
    
    total_batches = len(batches)
    batch_results = []
    
    for batch_num, batch in enumerate(batches, 1):
        if progress_callback:
            progress_callback(batch_num, total_batches)
        
        result = evaluate_batch(api_key, batch, marking_scheme, is_partial=True)
        
        if result.get("error"):
            result["message"] = f"Batch {batch_num}/{total_batches} failed: {result.get('message', 'Unknown error')}"
            return result
        
        batch_results.append(result)
        
        # Brief pause between batches to avoid rate limits
        if batch_num < total_batches:
            import time
            time.sleep(2)
    
    return merge_evaluation_results(batch_results)


def generate_rubric_from_image(api_key: str, images: list, teacher_instructions: str = "", max_retries: int = 3) -> dict:
    """
    Auto-generate a marking rubric by scanning the uploaded answer sheet images.
    Uses Gemini to identify questions and create a structured marking scheme.
    """
    import time
    
    rubric_prompt = """You are an expert CBSE/CISCE examiner. Analyze the uploaded answer sheet image(s) and generate a structured marking scheme.

Your task:
1. Identify ALL questions present in the image(s) (look for Q1, 1., Ans 1, etc.)
2. For each question, determine the topic/subject area
3. Create a step-by-step marking rubric with marks for each step

Rules:
- Use Markdown format with ## Q1, ## Q2 headers
- Use LaTeX ($...$) for all math/science expressions
- Each step should have a clear mark allocation
- Be specific about what earns marks at each step
- Include partial marking criteria where applicable
"""
    
    user_msg = """Analyze the uploaded answer sheet image(s) and generate a marking rubric.

"""
    if teacher_instructions.strip():
        user_msg += f"""TEACHER'S INSTRUCTIONS:
{teacher_instructions}

"""
    
    user_msg += """Generate the marking scheme in this EXACT format:

# Subject: [Detected Subject]

## Q1. [Question description] ([Total] Marks)
- **Step 1:** [criterion]: [marks]
- **Step 2:** [criterion]: [marks]
...

## Q2. [Question description] ([Total] Marks)
- **Step 1:** [criterion]: [marks]
...

Respond with ONLY the marking scheme in Markdown, nothing else."""
    
    for attempt in range(max_retries):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            content_parts = [rubric_prompt, user_msg]
            if isinstance(images, list):
                content_parts.extend(images)
            else:
                content_parts.append(images)
            
            response = model.generate_content(
                content_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                )
            )
            
            rubric_text = response.text.strip()
            # Clean markdown fences if present
            if '```markdown' in rubric_text:
                rubric_text = rubric_text.split('```markdown')[1].split('```')[0].strip()
            elif '```' in rubric_text:
                parts = rubric_text.split('```')
                if len(parts) >= 3:
                    rubric_text = parts[1].strip()
            
            return {"success": True, "rubric": rubric_text}
            
        except Exception as e:
            error_msg = str(e).lower()
            if ("quota" in error_msg or "rate" in error_msg or "429" in error_msg) and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"success": False, "message": str(e)}
    
    return {"success": False, "message": "Maximum retries exceeded."}



def generate_csv_report(result: dict, final_marks: float, marking_scheme: str) -> bytes:
    """Generate a CSV report of the multi-question evaluation."""
    rows = []
    
    # Header row
    rows.append({
        "Question": "TOTAL",
        "Status": "",
        "AI Marks": result.get("total_score", 0),
        "Final Marks": final_marks,
        "Max Marks": result.get("max_total", 0),
        "Transcription": "",
        "Feedback": result.get("overall_feedback", ""),
        "Confidence": result.get("confidence", "N/A")
    })
    
    # Per-question rows
    for eval_item in result.get("evaluations", []):
        rows.append({
            "Question": eval_item.get("q_no", "Q?"),
            "Status": eval_item.get("status", "Unknown"),
            "AI Marks": eval_item.get("marks", 0),
            "Final Marks": eval_item.get("marks", 0),  # Can be overridden later
            "Max Marks": eval_item.get("max_marks", 0),
            "Transcription": eval_item.get("transcription", ""),
            "Feedback": eval_item.get("feedback", ""),
            "Confidence": ""
        })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode('utf-8')


def generate_pdf_report(result: dict, final_marks: float, marking_scheme: str) -> bytes:
    """Generate a PDF report of the multi-question evaluation."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#667eea')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#333333')
    )
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        textColor=colors.HexColor('#667eea')
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        leading=16
    )
    
    story = []
    
    # Title
    story.append(Paragraph("AI Grading Assistant - Evaluation Report", title_style))
    story.append(Spacer(1, 20))
    
    # Date
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 20))
    
    # Total Marks Summary
    story.append(Paragraph("Score Summary", heading_style))
    marks_data = [
        ["Metric", "Value"],
        ["AI Total Score", str(result.get("total_score", 0))],
        ["Final Score (Teacher)", str(final_marks)],
        ["Maximum Marks", str(result.get("max_total", 0))],
        ["Number of Questions", str(len(result.get("evaluations", [])))],
        ["AI Confidence", result.get("confidence", "N/A").upper()]
    ]
    marks_table = Table(marks_data, colWidths=[3*inch, 2*inch])
    marks_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e9ecef'))
    ]))
    story.append(marks_table)
    story.append(Spacer(1, 20))
    
    # Overall Feedback
    if result.get("overall_feedback"):
        story.append(Paragraph("Overall Feedback", heading_style))
        story.append(Paragraph(result.get("overall_feedback", ""), body_style))
        story.append(Spacer(1, 15))
    
    # Per-Question Breakdown
    evaluations = result.get("evaluations", [])
    if evaluations:
        story.append(Paragraph("Question-wise Evaluation", heading_style))
        story.append(Spacer(1, 10))
        
        for eval_item in evaluations:
            q_no = eval_item.get("q_no", "Q?")
            status = eval_item.get("status", "Unknown")
            marks = eval_item.get("marks", 0)
            max_marks = eval_item.get("max_marks", 0)
            
            # Question header
            story.append(Paragraph(f"{q_no} — {marks}/{max_marks} marks ({status})", subheading_style))
            
            # Transcription
            transcription = eval_item.get("transcription", "")
            if transcription:
                # Clean LaTeX for PDF (basic cleanup)
                clean_trans = transcription.replace('$', '').replace('\\', '')
                story.append(Paragraph(f"<b>Answer:</b> {clean_trans}", body_style))
            
            # Step breakdown
            step_breakdown = eval_item.get("step_breakdown", [])
            if step_breakdown:
                for i, step in enumerate(step_breakdown, 1):
                    step_text = f"Step {i}: {step.get('step', 'N/A')} — {step.get('marks', 0)} marks"
                    reason = step.get('reason', '')
                    if reason:
                        step_text += f" ({reason})"
                    story.append(Paragraph(step_text, body_style))
            
            # Feedback
            feedback = eval_item.get("feedback", "")
            if feedback:
                story.append(Paragraph(f"<i>Feedback: {feedback}</i>", body_style))
            
            story.append(Spacer(1, 15))
    
    # Marking Scheme
    story.append(Paragraph("Marking Scheme Used", heading_style))
    scheme_text = marking_scheme.replace('\n', '<br/>')
    story.append(Paragraph(scheme_text, body_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # API Key Input
    api_key = st.text_input(
        "🔑 Google API Key",
        type="password",
        help="Enter your Google Gemini API key",
        placeholder="Enter your API key..."
    )
    
    st.markdown("""
    <div style="background: #e8f4f8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <p style="margin: 0; font-size: 0.9rem;">
            🆓 <strong>Get a FREE API Key:</strong><br/>
            <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: #667eea;">
                Google AI Studio →
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📚 Quick Guide")
    st.markdown("""
    1. Enter your API key above
    2. Paste the marking scheme
    3. Upload the answer sheet image
    4. Click 'Evaluate Answer'
    5. Review and adjust marks if needed
    6. Download the report
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; font-size: 0.8rem;">
        Made with ❤️ for Indian Educators<br/>
        CBSE & CISCE Compatible
    </div>
    """, unsafe_allow_html=True)

# ============================================
# MAIN CONTENT
# ============================================

# Header
st.markdown("""
<div class="main-header">
    <h1>📝 AI Grading Assistant</h1>
    <p>Intelligent Answer Sheet Evaluation for CBSE & CISCE Schools</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2 = st.tabs(["📄 Evaluate Answer", "📊 Results & Export"])

with tab1:
    # Step 1: Upload Images FIRST (needed for auto-rubric)
    st.markdown("""
    <p><span class="step-indicator">1</span><strong>Upload Answer Sheet Image</strong></p>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload Answer Sheet",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        label_visibility="collapsed",
        accept_multiple_files=True,
        help="Upload image(s) or scanned PDF(s) of the handwritten answer. Supports JPG, PNG, and PDF. You can upload multiple files."
    )
    
    if uploaded_files:
        # Convert to images for preview (PDFs → images via PyMuPDF)
        st.session_state.original_images = process_uploaded_files(uploaded_files)
        st.session_state.processed_images = [preprocess_image(img) for img in st.session_state.original_images]
        
        num_pages = len(st.session_state.original_images)
        with st.expander(f"🔍 Preview Uploaded Image(s) ({num_pages} page(s))", expanded=True):
            for idx, (orig, proc) in enumerate(zip(st.session_state.original_images, st.session_state.processed_images)):
                if num_pages > 1:
                    st.markdown(f"**Page {idx + 1}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original**")
                    st.image(orig, use_container_width=True)
                with col2:
                    st.markdown("**Enhanced (Preview)**")
                    st.image(proc, use_container_width=True)
                if idx < num_pages - 1:
                    st.markdown("---")
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Step 2: Marking Scheme (Manual or Auto-Generated)
    st.markdown("""
    <p><span class="step-indicator">2</span><strong>Marking Scheme</strong></p>
    """, unsafe_allow_html=True)
    
    scheme_mode = st.radio(
        "How would you like to provide the marking scheme?",
        ["✍️ Write/Paste Manually", "🤖 Auto-Generate from Answer Sheet"],
        horizontal=True,
        help="You can paste your own rubric or let AI generate one from the uploaded answer sheet."
    )
    
    if scheme_mode == "🤖 Auto-Generate from Answer Sheet":
        st.markdown("""<div class="info-card">
            <strong>🤖 Auto-Generate Rubric</strong><br/>
            Upload your <strong>answer key / correct answers</strong> (teacher's handwritten or printed solutions).
            The AI will scan them, identify all questions, and create a structured marking scheme.
        </div>""", unsafe_allow_html=True)
        
        # Separate uploader for teacher's answer key
        st.markdown("**📄 Upload Answer Key (Teacher's Correct Answers)**")
        answer_key_files = st.file_uploader(
            "Upload Answer Key",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            label_visibility="collapsed",
            accept_multiple_files=True,
            key="answer_key_uploader",
            help="Upload image(s) or scanned PDF(s) of the correct answers / answer key. Supports JPG, PNG, and PDF."
        )
        
        if answer_key_files:
            answer_key_preview = process_uploaded_files(answer_key_files)
            num_key_pages = len(answer_key_preview)
            with st.expander(f"🔍 Preview Answer Key ({num_key_pages} page(s))", expanded=False):
                for idx, img in enumerate(answer_key_preview):
                    if num_key_pages > 1:
                        st.markdown(f"**Page {idx + 1}**")
                    st.image(img, use_container_width=True)
        
        teacher_instructions = st.text_area(
            "Optional: Specify marks or rubric instructions",
            height=100,
            placeholder="""e.g.:
- Q1 should be worth 5 marks total
- Q2: Give 2 marks for diagram, 3 for explanation
- Total paper is out of 20 marks
- Deduct marks for missing units""",
            help="Optional: Give the AI hints about mark distribution. Leave blank for AI to decide."
        )
        
        gen_col1, gen_col2 = st.columns([1, 1])
        with gen_col1:
            generate_rubric_btn = st.button("🪄 Generate Marking Scheme", use_container_width=True)
        
        if generate_rubric_btn:
            if not api_key:
                st.error("⚠️ Please enter your Google API key in the sidebar")
            elif not answer_key_files:
                st.error("⚠️ Please upload your answer key image(s) above")
            else:
                with st.spinner("🔄 Analyzing answer key and generating rubric..."):
                    # Use images (converted from PDFs if needed) for rubric generation
                    answer_key_images_for_api = process_uploaded_files(answer_key_files)
                    rubric_result = generate_rubric_from_image(
                        api_key, answer_key_images_for_api, teacher_instructions
                    )
                    if rubric_result.get("success"):
                        st.session_state.generated_rubric = rubric_result["rubric"]
                        st.success("✅ Marking scheme generated! Review and edit below.")
                    else:
                        st.error(f"❌ {rubric_result.get('message', 'Failed to generate rubric')}")
        
        # Show generated rubric in editable text area
        marking_scheme = st.text_area(
            "Generated Marking Scheme (editable)",
            height=300,
            value=st.session_state.generated_rubric or "",
            placeholder="Click 'Generate Marking Scheme' above to auto-generate, then edit as needed...",
            label_visibility="collapsed",
            help="Review and edit the auto-generated marking scheme. You can adjust marks, add/remove steps, etc."
        )
        
        # Preview rendered rubric
        if marking_scheme.strip():
            with st.expander("📋 Preview Marking Scheme (Rendered)", expanded=False):
                st.markdown(marking_scheme)
    
    else:
        marking_scheme = st.text_area(
            "Marking Scheme",
            height=250,
            placeholder="""# Subject: Mathematics - Term 1

## Q1. Solve for x: 2x + 5 = 15 (4 Marks)
- **Step 1:** Writing equation correctly: 1.0
- **Step 2:** Isolate 2x (2x = 10): 1.0
- **Step 3:** Divide by 2: 1.0
- **Step 4:** Final Answer (x=5): 1.0

## Q2. Find the area of circle with r=7 (3 Marks)
- **Step 1:** Formula ($\\pi r^2$): 1.0
- **Step 2:** Calculation ($22/7 \\times 7 \\times 7$): 1.0
- **Step 3:** Result (154 $cm^2$): 1.0

*Use ## Q1, ## Q2 headers for multi-question papers*""",
            label_visibility="collapsed",
            help="Use Markdown format with ## Q1, ## Q2 headers for multi-question papers."
        )
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Step 3: Evaluate Button
    st.markdown("""
    <p><span class="step-indicator">3</span><strong>Evaluate Answer</strong></p>
    """, unsafe_allow_html=True)
    
    evaluate_btn = st.button("🚀 Evaluate Answer", use_container_width=True)
    
    if evaluate_btn:
        # Validation
        errors = []
        if not api_key:
            errors.append("⚠️ Please enter your Google API key in the sidebar")
        if not marking_scheme.strip():
            errors.append("⚠️ Please enter the marking scheme")
        if not uploaded_files:
            errors.append("⚠️ Please upload an answer sheet image or PDF")
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            # Use images for evaluation (PDFs already converted in preview step)
            images_to_send = st.session_state.original_images
            num_images = len(images_to_send)
            stride = 3  # batch_size(4) - 1 overlap
            num_batches = max(1, (num_images - 1 + stride - 1) // stride) if num_images > 4 else 1
            
            # Show progress for multi-batch evaluations
            if num_images > 4:
                progress_bar = st.progress(0, text=f"🔄 Evaluating batch 1 of {num_batches}...")
                def update_progress(current, total):
                    progress_bar.progress(current / total, text=f"🔄 Evaluating batch {current} of {total}...")
            else:
                progress_bar = None
                def update_progress(current, total):
                    pass
            
            with st.spinner("🔄 Processing answer sheet..."):
                result = evaluate_with_gemini(
                    api_key, images_to_send, marking_scheme,
                    batch_size=4, progress_callback=update_progress
                )
            
            if progress_bar:
                progress_bar.empty()
            
            if result.get("error"):
                st.error(f"❌ {result.get('message', 'An error occurred')}")
            else:
                st.session_state.evaluation_result = result
                st.session_state.final_marks = result.get("total_score", 0)
                st.session_state.max_marks = result.get("max_total", 10)
                # Initialize question overrides
                st.session_state.question_overrides = {}
                for eval_item in result.get("evaluations", []):
                    q_no = eval_item.get("q_no", "Q?")
                    st.session_state.question_overrides[q_no] = eval_item.get("marks", 0)
                st.success("✅ Evaluation complete! Go to 'Results & Export' tab to view.")

with tab2:
    if st.session_state.evaluation_result and not st.session_state.evaluation_result.get("error"):
        result = st.session_state.evaluation_result
        
        # Step 4: Results Overview
        st.markdown("""
        <p><span class="step-indicator">4</span><strong>Evaluation Results</strong></p>
        """, unsafe_allow_html=True)
        
        # Total Score Display
        total_score = result.get('total_score', 0)
        max_total = result.get('max_total', 0)
        num_questions = len(result.get('evaluations', []))
        
        col_score, col_info = st.columns([1, 2])
        with col_score:
            st.markdown(f"""
            <div class="marks-display">
                {total_score} / {max_total}
                <div style="font-size: 0.8rem; font-weight: 400; opacity: 0.9;">Total AI Score ({num_questions} Questions)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_info:
            # Confidence Warning
            confidence = result.get("confidence", "medium")
            if confidence == "low":
                st.markdown("""
                <div class="warning-card">
                    ⚠️ <strong>Note:</strong> Handwriting is unclear in some areas, please verify carefully.
                </div>
                """, unsafe_allow_html=True)
            
            # Overall feedback
            overall_feedback = result.get("overall_feedback", "")
            if overall_feedback:
                st.markdown(f"""
                <div class="success-card">
                    💡 <strong>Overall:</strong> {overall_feedback}
                </div>
                """, unsafe_allow_html=True)
            
            # Warnings
            warnings = result.get("warnings", [])
            if warnings:
                for warning in warnings:
                    st.warning(f"⚠️ {warning}")
        
        st.markdown("---")
        
        # Two-column layout: Images on left, Questions on right
        col_img, col_questions = st.columns([1, 1])
        
        with col_img:
            st.markdown("### 📸 Uploaded Answer Sheet(s)")
            if st.session_state.original_images:
                for idx, img in enumerate(st.session_state.original_images):
                    if len(st.session_state.original_images) > 1:
                        st.markdown(f"**Page {idx + 1}**")
                    st.image(img, use_container_width=True)
            else:
                st.info("Image preview not available")
        
        with col_questions:
            st.markdown("### 📋 Question-wise Evaluation")
            
            evaluations = result.get("evaluations", [])
            if evaluations:
                for eval_item in evaluations:
                    q_no = eval_item.get("q_no", "Q?")
                    status = eval_item.get("status", "Unknown")
                    marks = eval_item.get("marks", 0)
                    max_marks = eval_item.get("max_marks", 0)
                    question_text = eval_item.get("question_text", "")
                    
                    # Status color coding
                    if status == "Attempted":
                        status_icon = "✅"
                        status_color = "#28a745"
                    elif status == "Not Attempted":
                        status_icon = "❌"
                        status_color = "#dc3545"
                    elif status == "Missing Scheme":
                        status_icon = "⚠️"
                        status_color = "#ffc107"
                    else:
                        status_icon = "❓"
                        status_color = "#6c757d"
                    
                    # Expandable question card
                    with st.expander(f"{status_icon} **{q_no}** — {marks}/{max_marks} marks | {status}", expanded=False):
                        if question_text:
                            st.markdown(f"**Question:** {question_text}")
                        
                        # Transcription (renders LaTeX automatically via st.markdown)
                        transcription = eval_item.get("transcription", "")
                        if transcription:
                            st.markdown("**📝 Student's Answer:**")
                            st.markdown(transcription)
                        
                        # Diagram description (if present)
                        if eval_item.get("has_diagram"):
                            diagram_desc = eval_item.get("diagram_description", "")
                            if diagram_desc:
                                st.markdown("**🖼️ Diagram Detected:**")
                                st.info(diagram_desc)
                        
                        # Step breakdown
                        step_breakdown = eval_item.get("step_breakdown", [])
                        if step_breakdown:
                            st.markdown("**📊 Step Breakdown:**")
                            for i, step in enumerate(step_breakdown, 1):
                                step_name = step.get("step", "N/A")
                                step_marks = step.get("marks", 0)
                                step_reason = step.get("reason", "")
                                st.markdown(f"- **Step {i}:** {step_name} — *{step_marks} marks*")
                                if step_reason:
                                    st.caption(f"  ↳ {step_reason}")
                        
                        # Feedback
                        feedback = eval_item.get("feedback", "")
                        if feedback:
                            st.markdown(f"**💡 Feedback:** {feedback}")
                        
                        # Per-question mark override
                        st.markdown("---")
                        new_marks = st.number_input(
                            f"Override marks for {q_no}",
                            min_value=0.0,
                            max_value=float(max_marks) if max_marks > 0 else 100.0,
                            value=float(st.session_state.question_overrides.get(q_no, marks)),
                            step=0.5,
                            key=f"override_{q_no}",
                            help=f"Adjust marks for {q_no}. Original AI suggestion: {marks}"
                        )
                        st.session_state.question_overrides[q_no] = new_marks
            else:
                st.info("No questions evaluated. Please check the image and marking scheme.")
        
        st.markdown("---")
        
        # Step 5: Final Totals and Export
        st.markdown("""
        <p><span class="step-indicator">5</span><strong>Final Marks & Export</strong></p>
        """, unsafe_allow_html=True)
        
        # Calculate adjusted total from overrides
        adjusted_total = sum(st.session_state.question_overrides.values())
        original_total = result.get("total_score", 0)
        
        col_total, col_download = st.columns([1, 1])
        
        with col_total:
            st.markdown("#### 📊 Final Score Summary")
            
            if adjusted_total != original_total:
                st.markdown(f"""
                <div class="marks-display" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                    {adjusted_total} / {max_total}
                    <div style="font-size: 0.8rem; font-weight: 400; opacity: 0.9;">Teacher Adjusted Total</div>
                </div>
                """, unsafe_allow_html=True)
                st.info(f"📝 Total adjusted from AI's {original_total} to {adjusted_total}")
            else:
                st.markdown(f"""
                <div class="marks-display">
                    {adjusted_total} / {max_total}
                    <div style="font-size: 0.8rem; font-weight: 400; opacity: 0.9;">Final Total</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Per-question summary table
            if result.get("evaluations"):
                st.markdown("**Question Summary:**")
                summary_data = []
                for eval_item in result.get("evaluations", []):
                    q_no = eval_item.get("q_no", "Q?")
                    ai_marks = eval_item.get("marks", 0)
                    final_marks = st.session_state.question_overrides.get(q_no, ai_marks)
                    max_m = eval_item.get("max_marks", 0)
                    summary_data.append({
                        "Question": q_no,
                        "AI Marks": ai_marks,
                        "Final Marks": final_marks,
                        "Max": max_m
                    })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            st.session_state.final_marks = adjusted_total
        
        with col_download:
            st.markdown("#### 📥 Download Report")
            
            # CSV Download
            csv_data = generate_csv_report(result, adjusted_total, marking_scheme)
            st.download_button(
                label="📄 Download CSV Report",
                data=csv_data,
                file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # PDF Download
            pdf_data = generate_pdf_report(result, adjusted_total, marking_scheme)
            st.download_button(
                label="📑 Download PDF Report",
                data=pdf_data,
                file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    else:
        st.markdown("""
        <div class="info-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📋 No Evaluation Yet</h3>
            <p style="color: #666;">
                Complete the evaluation in the "Evaluate Answer" tab to see results here.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🎓 AI Grading Assistant v1.0 | Built for CBSE & CISCE Educators</p>
    <p style="font-size: 0.8rem;">Powered by Google Gemini 1.5 Flash</p>
</div>
""", unsafe_allow_html=True)
