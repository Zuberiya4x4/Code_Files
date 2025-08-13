import streamlit as st
import fitz  # PyMuPDF
import json
import re
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from datetime import datetime
import time
import warnings
import os

# Suppress warnings at the very beginning
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PADDLEOCR_LOG_LEVEL'] = 'ERROR'

# Import List type for type hints
from typing import Dict, List, Tuple, Any

# Configure page
st.set_page_config(
    page_title="AI-Powered PDF Field Extractor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .field-card {
        border: 2px solid #e3e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    .accepted-field {
        border-color: #00b894 !important;
        background: #d1f2eb !important;
    }
    .rejected-field {
        border-color: #e17055 !important;
        background: #fadbd8 !important;
    }
    .pending-field {
        border-color: #fdcb6e !important;
        background: #fef9e7 !important;
    }
    .extraction-stats {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedPDFHeaderExtractor:
    def __init__(self):
        # Enhanced header patterns with more variations and better coverage
        self.header_patterns = {
            "document_title": [
                r"payment\s*advice", r"remittance\s*advice", r"transfer\s*advice", 
                r"transaction\s*advice", r"advice\s*note"
            ],
            "advice_date": [
                r"advice\s*sending\s*date", r"advice\s*date", r"date\s*of\s*advice", 
                r"sending\s*date", r"transaction\s*date", r"process\s*date"
            ],
            "advice_ref": [
                r"advice\s*reference\s*no\.?", r"advice\s*ref\s*no\.?", 
                r"advice\s*reference", r"advice\s*ref", r"reference\s*number"
            ],
            "recipient_name": [
                r"recipient'?s?\s*name\s*and\s*contact\s*information", 
                r"recipient'?s?\s*name", r"company\s*name", r"customer\s*name",
                r"to\s*name", r"recipient", r"client\s*name"
            ],
            "receipt_email": [
                r"receipt\s*email", r"email\s*address", r"e-?mail", 
                r"contact\s*email", r"notification\s*email", r"email\s*id"
            ],
            "transaction_type": [
                r"transaction\s*type", r"payment\s*type", r"transfer\s*type"
            ],
            "sub_payment_type": [
                r"sub\s*payment\s*type", r"payment\s*sub\s*type", 
                r"payment\s*method", r"transfer\s*method"
            ],
            "beneficiary_name": [
                r"beneficiary'?s?\s*name", r"beneficiary\s*name", 
                r"payee\s*name", r"receiver\s*name", r"credit\s*to",
                r"beneficiary", r"payee"
            ],
            "beneficiary_bank": [
                r"beneficiary'?s?\s*bank", r"beneficiary\s*bank", 
                r"receiving\s*bank", r"destination\s*bank", r"bank\s*name",
                r"credit\s*bank", r"payee\s*bank"
            ],
            "account_number": [
                r"beneficiary'?s?\s*account\s*number", r"beneficiary'?s?\s*account\s*no\.?",
                r"account\s*number", r"account\s*no\.?", r"a/?c\s*no\.?",
                r"credit\s*account", r"destination\s*account", r"payee\s*account"
            ],
            "customer_reference": [
                r"customer\s*reference", r"cust\s*ref", r"customer\s*ref",
                r"client\s*reference", r"internal\s*reference"
            ],
            "debit_amount": [
                r"debit\s*amount", r"amount\s*debited", r"charged\s*amount",
                r"total\s*debit"
            ],
            "remittance_amount": [
                r"remittance\s*amount", r"remit\s*amount", r"transfer\s*amount",
                r"credit\s*amount", r"net\s*amount"
            ],
            "handling_fee": [
                r"handling\s*fee", r"charges", r"fees", r"service\s*charge",
                r"transaction\s*fee", r"processing\s*fee"
            ],
            "value_date": [
                r"value\s*date", r"settlement\s*date", r"effective\s*date",
                r"posting\s*date"
            ],
            "remitter_name": [
                r"remitter'?s?\s*name", r"sender\s*name", r"from\s*name",
                r"originator\s*name", r"remitter", r"sender"
            ],
            "remitting_bank": [
                r"remitting\s*bank", r"sender\s*bank", r"originating\s*bank",
                r"source\s*bank", r"of\s*remitting\s*bank", r"from\s*bank"
            ],
            "instruction_reference": [
                r"instruction\s*reference", r"instr\s*ref", r"instruction\s*no",
                r"mandate\s*reference"
            ],
            "other_reference": [
                r"other\s*reference", r"additional\s*ref", r"misc\s*reference",
                r"extra\s*reference"
            ],
            "remitter_to_beneficiary_info": [
                r"remitter\s*to\s*beneficiary\s*information", 
                r"additional\s*info", r"payment\s*info", r"transaction\s*details", r"remarks"
            ]
        }
        
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (0, 128, 0), (128, 128, 0), (0, 0, 128),
            (128, 0, 0), (0, 128, 128), (192, 192, 192), (255, 20, 147),
            (32, 178, 170), (255, 69, 0), (154, 205, 50), (75, 0, 130),
            (220, 20, 60), (255, 140, 0), (50, 205, 50), (70, 130, 180)
        ]
    
    @st.cache_resource
    def setup_ocr_engines(_self):
        """Initialize multiple OCR engines for better accuracy with caching"""
        engines = {}
        
        try:
            # EasyOCR - suppress GPU warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                engines['easyocr'] = easyocr.Reader(['en'], verbose=False)
        except Exception as e:
            st.warning(f"EasyOCR initialization failed: {e}")
            engines['easyocr'] = None
            
        try:
            # RapidOCR
            engines['rapidocr'] = RapidOCR()
        except Exception as e:
            st.warning(f"RapidOCR initialization failed: {e}")
            engines['rapidocr'] = None
            
        try:
            # PaddleOCR - Fixed deprecation warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                engines['paddleocr'] = PaddleOCR(
                    use_textline_orientation=True,  # Updated parameter
                    lang='en', 
                    show_log=False
                )
        except Exception as e:
            st.warning(f"PaddleOCR initialization failed: {e}")
            engines['paddleocr'] = None
        
        return engines
    
    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with coordinates from PDF with enhanced filtering"""
        doc = fitz.open(pdf_path)
        text_blocks = []
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            # Enhanced filtering - keep more text including single meaningful characters
                            if text and (len(text) > 1 or text.isdigit() or text in [':', '-', '/']):
                                text_blocks.append({
                                    "text": text,
                                    "bbox": span["bbox"],
                                    "page": page_num,
                                    "font_size": span["size"],
                                    "flags": span.get("flags", 0),
                                    "font": span.get("font", "")
                                })
        
        doc.close()
        return text_blocks
    
    def find_header_field_matches(self, text_blocks: List[Dict]) -> Dict[str, List[Dict]]:
        """Find potential header field matches with enhanced pattern matching"""
        matches = {}
        
        for field_name, patterns in self.header_patterns.items():
            matches[field_name] = []
            
            for block in text_blocks:
                text_lower = block["text"].lower()
                text_clean = re.sub(r'[^\w\s]', ' ', text_lower).strip()
                
                for pattern in patterns:
                    # Enhanced pattern matching with fuzzy matching
                    if re.search(f"\\b{pattern}\\b", text_clean):
                        matches[field_name].append(block)
                        break
                    # Also check for partial matches for important fields
                    elif field_name in ['beneficiary_name', 'recipient_name', 'receipt_email']:
                        # More flexible matching for these critical fields
                        pattern_words = pattern.split(r'\s+')
                        if len(pattern_words) > 1 and any(word in text_clean for word in pattern_words):
                            matches[field_name].append(block)
                            break
        
        return matches
    
    def is_valid_email(self, text: str) -> bool:
        """Check if text contains a valid email address"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return bool(re.search(email_pattern, text))
    
    def is_valid_account_number(self, text: str) -> bool:
        """Check if text looks like an account number"""
        # Account numbers are typically numeric, may have spaces or dashes
        cleaned = re.sub(r'[^\w]', '', text)
        return (len(cleaned) >= 6 and 
                (cleaned.isdigit() or 
                 (cleaned.isalnum() and sum(c.isdigit() for c in cleaned) >= len(cleaned) * 0.7)))
    
    def clean_extracted_value(self, text: str, field_name: str) -> str:
        """Clean extracted values based on field type"""
        if not text:
            return ""
        
        # Remove common prefixes that might be extracted along with values
        common_prefixes = [
            r'^[:\-\s]+', r'^\w+:', r'^name[:\s]*', r'^number[:\s]*', 
            r'^amount[:\s]*', r'^date[:\s]*', r'^reference[:\s]*'
        ]
        
        cleaned = text.strip()
        for prefix in common_prefixes:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Field-specific cleaning
        if field_name == 'receipt_email':
            # Extract email from text
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', cleaned)
            if email_match:
                cleaned = email_match.group()
        elif field_name == 'account_number':
            # Clean account number - keep only alphanumeric and common separators
            cleaned = re.sub(r'[^\w\-\s]', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def find_values_for_headers(self, text_blocks: List[Dict], header_matches: Dict) -> Dict[str, Dict]:
        """Enhanced value finding with better logic for specific fields"""
        field_data = {}
        
        for field_name, headers in header_matches.items():
            if not headers:
                continue
                
            best_match = None
            best_values = []
            best_score = float('inf')
            
            for header in headers:
                header_bbox = header["bbox"]
                header_y = header_bbox[1]
                header_x_end = header_bbox[2]
                
                # Enhanced adjacent value finding
                potential_values = []
                
                for block in text_blocks:
                    # Skip if it's the header itself
                    if block["text"].lower() == header["text"].lower():
                        continue
                    
                    # Enhanced header text filtering
                    block_text_lower = block["text"].lower()
                    header_text_lower = header["text"].lower()
                    
                    # Skip blocks that are clearly part of header labels
                    skip_patterns = [
                        r'\b(name|number|amount|date|reference|bank|account)\b',
                        r'^[:\-\s]+$',  # Only punctuation
                    ]
                    
                    should_skip = False
                    for pattern in skip_patterns:
                        if re.search(pattern, block_text_lower) and len(block["text"].strip()) < 15:
                            should_skip = True
                            break
                    
                    if should_skip:
                        continue
                        
                    # Skip if block contains too much of the header text
                    if (len(header_text_lower) > 5 and 
                        header_text_lower in block_text_lower and 
                        len(block_text_lower.replace(header_text_lower, '').strip()) < 3):
                        continue
                    
                    block_bbox = block["bbox"]
                    block_x = block_bbox[0]
                    block_y = block_bbox[1]
                    
                    # Calculate distances
                    horizontal_distance = block_x - header_x_end
                    vertical_distance = abs(block_y - header_y)
                    
                    # Field-specific validation
                    is_valid_for_field = True
                    if field_name == 'receipt_email':
                        is_valid_for_field = self.is_valid_email(block["text"])
                    elif field_name == 'account_number':
                        is_valid_for_field = self.is_valid_account_number(block["text"])
                    elif field_name in ['beneficiary_name', 'recipient_name', 'remitter_name']:
                        # Names shouldn't be just numbers or single characters
                        text_clean = re.sub(r'[^\w\s]', '', block["text"]).strip()
                        is_valid_for_field = (len(text_clean) > 1 and 
                                            not text_clean.isdigit() and
                                            not re.match(r'^[:\-\s]+$', block["text"]))
                    
                    if not is_valid_for_field:
                        continue
                        
                    # Scoring logic (same line preferred, then below)
                    if vertical_distance < 15 and horizontal_distance > -10 and horizontal_distance < 300:
                        score = abs(horizontal_distance) + vertical_distance * 2
                        potential_values.append({
                            "block": block,
                            "score": score,
                            "type": "right"
                        })
                    elif block_y > header_y and vertical_distance < 80:
                        horizontal_alignment_penalty = abs(block_x - header_bbox[0]) * 0.3
                        score = vertical_distance * 3 + horizontal_alignment_penalty
                        potential_values.append({
                            "block": block,
                            "score": score,
                            "type": "below"
                        })
                    
                    # Enhanced value selection
                    if potential_values:
                        potential_values.sort(key=lambda x: x["score"])
                        
                        # Take the best value(s)
                        closest_values = [potential_values[0]]
                        best_score_val = potential_values[0]["score"]
                        
                        # Look for continuation values
                        for val in potential_values[1:]:
                            if val["score"] - best_score_val < 25:
                                prev_bbox = closest_values[-1]["block"]["bbox"]
                                curr_bbox = val["block"]["bbox"]
                                
                                # Same line continuation
                                if (abs(curr_bbox[1] - prev_bbox[1]) < 10 and 
                                    curr_bbox[0] - prev_bbox[2] < 60):
                                    closest_values.append(val)
                                # Multi-line continuation
                                elif (curr_bbox[1] - prev_bbox[3] < 30 and
                                      abs(curr_bbox[0] - prev_bbox[0]) < 40):
                                    closest_values.append(val)
                        
                        if best_score_val < best_score:
                            best_match = header
                            best_values = [v["block"] for v in closest_values]
                            best_score = best_score_val
            
            # Process the best match found
            if best_match and best_values:
                # Create combined bounding box
                all_bboxes = [best_match["bbox"]] + [v["bbox"] for v in best_values]
                combined_bbox = [
                    min(bbox[0] for bbox in all_bboxes),
                    min(bbox[1] for bbox in all_bboxes),
                    max(bbox[2] for bbox in all_bboxes),
                    max(bbox[3] for bbox in all_bboxes)
                ]
                
                # Combine and clean values
                value_texts = []
                for value in best_values:
                    clean_text = self.clean_extracted_value(value["text"], field_name)
                    if clean_text:
                        value_texts.append(clean_text)
                
                combined_value_text = " ".join(value_texts)
                final_clean_value = self.clean_extracted_value(combined_value_text, field_name)
                
                field_data[field_name] = {
                    "header": best_match,
                    "values": best_values,
                    "combined_bbox": combined_bbox,
                    "extracted_text": f"{best_match['text']}: {final_clean_value}",
                    "clean_value": final_clean_value,
                    "confidence": max(0, min(100, int((100 - best_score) * 2)))  # Simple confidence score
                }
        
        return field_data
    
    def draw_bounding_boxes(self, pdf_path: str, field_data: Dict) -> Image.Image:
        """Enhanced bounding box drawing with better visual appeal"""
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        
        # Convert PDF to high-quality image
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))  # Higher resolution
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        
        draw = ImageDraw.Draw(img)
        
        # Try to load fonts
        try:
            label_font = ImageFont.truetype("arial.ttf", 18)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except:
            label_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        color_index = 0
        for field_name, data in field_data.items():
            if "combined_bbox" in data:
                bbox = data["combined_bbox"]
                scaled_bbox = [coord * 2.5 for coord in bbox]  # Match scaling factor
                
                color = self.colors[color_index % len(self.colors)]
                
                # Draw enhanced bounding box with rounded corners effect
                x1, y1, x2, y2 = scaled_bbox
                
                # Main bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
                
                # Add corner markers for better visibility
                corner_size = 10
                # Top-left corner
                draw.rectangle([x1-2, y1-2, x1+corner_size, y1+2], fill=color)
                draw.rectangle([x1-2, y1-2, x1+2, y1+corner_size], fill=color)
                # Top-right corner  
                draw.rectangle([x2-corner_size, y1-2, x2+2, y1+2], fill=color)
                draw.rectangle([x2-2, y1-2, x2+2, y1+corner_size], fill=color)
                # Bottom corners
                draw.rectangle([x1-2, y2-2, x1+corner_size, y2+2], fill=color)
                draw.rectangle([x1-2, y2-corner_size, x1+2, y2+2], fill=color)
                draw.rectangle([x2-corner_size, y2-2, x2+2, y2+2], fill=color)
                draw.rectangle([x2-2, y2-corner_size, x2+2, y2+2], fill=color)
                
                # Enhanced label
                label = field_name.replace("_", " ").title()
                confidence = data.get("confidence", 0)
                label_with_confidence = f"{label} ({confidence}%)"
                
                # Calculate text dimensions
                text_bbox = draw.textbbox((0, 0), label_with_confidence, font=label_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Position label
                label_x = max(5, x1)
                label_y = max(5, y1 - text_height - 12)
                
                # Draw label background with shadow effect
                padding = 8
                shadow_offset = 2
                
                # Shadow
                draw.rectangle(
                    [label_x + shadow_offset, label_y + shadow_offset, 
                     label_x + text_width + padding + shadow_offset, 
                     label_y + text_height + padding + shadow_offset],
                    fill=(0, 0, 0, 50)  # Semi-transparent shadow
                )
                
                # Background
                draw.rectangle(
                    [label_x, label_y, 
                     label_x + text_width + padding, label_y + text_height + padding],
                    fill=color
                )
                
                # Text
                draw.text((label_x + padding//2, label_y + padding//2), 
                         label_with_confidence, fill="white", font=label_font)
                
                color_index += 1
        
        doc.close()
        return img
    
    def extract_json_data(self, field_data: Dict) -> Dict[str, Any]:
        """Enhanced JSON extraction with metadata"""
        json_data = {}
        metadata = {
            "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_fields_detected": len(field_data),
            "field_confidence_scores": {}
        }
        
        for field_name, data in field_data.items():
            clean_value = data.get("clean_value", "")
            confidence = data.get("confidence", 0)
            
            json_data[field_name] = clean_value
            metadata["field_confidence_scores"][field_name] = confidence
        
        return {"extracted_data": json_data, "metadata": metadata}

def main():
    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI-Powered PDF Field Extractor</h1>
        <p>Advanced document processing with intelligent field detection and extraction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Processing options
        st.subheader("🎛️ Processing Options")
        confidence_threshold = st.slider("Confidence Threshold", 0, 100, 50, 5)
        show_debug = st.checkbox("Show Debug Information", False)
        auto_accept_high_confidence = st.checkbox("Auto-accept High Confidence Fields (>80%)", True)
        
        # Statistics placeholder
        st.subheader("📊 Session Statistics")
        stats_placeholder = st.empty()
    
    # Initialize session state
    if 'field_data' not in st.session_state:
        st.session_state.field_data = {}
    if 'json_data' not in st.session_state:
        st.session_state.json_data = {}
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'field_status' not in st.session_state:
        st.session_state.field_status = {}
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'final_json' not in st.session_state:
        st.session_state.final_json = {}
    
    extractor = EnhancedPDFHeaderExtractor()
    
    # File upload with enhanced UI
    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload a PDF document containing structured header fields for automatic extraction"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with open("temp_pdf.pdf", "wb") as f:
            f.write(uploaded_file.read())
        
        # Process PDF with progress tracking
        if not st.session_state.processing_complete:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Extracting text blocks...")
            progress_bar.progress(20)
            text_blocks = extractor.extract_text_blocks("temp_pdf.pdf")
            
            status_text.text("🎯 Finding header field matches...")
            progress_bar.progress(40)
            header_matches = extractor.find_header_field_matches(text_blocks)
            
            status_text.text("🔗 Linking values to headers...")
            progress_bar.progress(60)
            field_data = extractor.find_values_for_headers(text_blocks, header_matches)
            
            status_text.text("🎨 Generating visualization...")
            progress_bar.progress(80)
            processed_image = extractor.draw_bounding_boxes("temp_pdf.pdf", field_data)
            
            status_text.text("📊 Preparing data...")
            progress_bar.progress(90)
            json_data = extractor.extract_json_data(field_data)
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Store in session state
            st.session_state.field_data = field_data
            st.session_state.json_data = json_data
            st.session_state.processed_image = processed_image
            st.session_state.processing_complete = True
            
            # Initialize field status with auto-accept logic
            for field_name, data in field_data.items():
                confidence = data.get("confidence", 0)
                if auto_accept_high_confidence and confidence > 80:
                    st.session_state.field_status[field_name] = "accepted"
                else:
                    st.session_state.field_status[field_name] = "pending"
            
            time.sleep(0.5)  # Brief pause to show completion
            # Trigger a re-render by changing a session state variable
            st.session_state._processing_complete = True
    
        # Display results
        if st.session_state.processing_complete:
            # Success message with stats
            detected_fields = len(st.session_state.field_data)
            high_confidence_fields = sum(1 for data in st.session_state.field_data.values() 
                                       if data.get("confidence", 0) > 80)
            
            st.markdown(f"""
            <div class="extraction-stats">
                <h3>✅ Extraction Complete!</h3>
                <p>Successfully detected <strong>{detected_fields}</strong> fields with <strong>{high_confidence_fields}</strong> high-confidence matches</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("📋 Document with Detected Fields")
                if st.session_state.processed_image:
                    st.image(st.session_state.processed_image, use_column_width=True)
                    
                    # Download annotated image
                    img_buffer = io.BytesIO()
                    st.session_state.processed_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=img_buffer.getvalue(),
                        file_name=f"annotated_{uploaded_file.name}.png",
                        mime="image/png"
                    )
            
            with col2:
                st.subheader("🎯 Field Selection & Validation")
                
                # Initialize selections if not exists
                if not st.session_state.field_status:
                    st.session_state.field_status = {
                        field: "accepted" if auto_accept_high_confidence and data.get("confidence", 0) > 80 
                        else "pending" 
                        for field, data in st.session_state.field_data.items()
                    }
                
                # Display fields with accept/reject options
                for field_name, data in st.session_state.field_data.items():
                    status = st.session_state.field_status.get(field_name, "pending")
                    
                    with st.expander(f"📌 {field_name.replace('_', ' ').title()}", expanded=True):
                        col_info, col_actions = st.columns([2, 1])
                        
                        with col_info:
                            st.write(f"**Value:** {data['clean_value'][:100]}...")
                            st.write(f"**Confidence:** {data['confidence']:.1f}%")
                            st.write(f"**Method:** {data.get('pattern', 'Pattern matching')}")
                            st.write(f"**Status:** {status.title()}")
                        
                        with col_actions:
                            if st.button("✅ Accept" if status != "accepted" else "✅ Accepted", 
                                        key=f"accept_{field_name}", 
                                        disabled=(status == "accepted"),
                                        type="primary" if status != "accepted" else "secondary"):
                                st.session_state.field_status[field_name] = "accepted"
                                st.experimental_rerun()
                            
                            if st.button("❌ Reject" if status != "rejected" else "❌ Rejected", 
                                        key=f"reject_{field_name}",
                                        disabled=(status == "rejected"),
                                        type="primary" if status != "rejected" else "secondary"):
                                st.session_state.field_status[field_name] = "rejected"
                                st.experimental_rerun()
            
            # Submit and JSON generation
            st.divider()
            col_submit, col_download, col_stats = st.columns([2, 2, 1])
            
            with col_submit:
                if st.button("🚀 Submit & Generate JSON", type="primary", use_container_width=True):
                    # Generate final JSON
                    final_json = {}
                    for field_name, status in st.session_state.field_status.items():
                        if status == "accepted" and field_name in st.session_state.field_data:
                            final_json[field_name] = st.session_state.field_data[field_name]['clean_value']
                    
                    st.session_state.final_json = final_json
                    st.success("✅ JSON generated successfully!")
            
            with col_download:
                if st.session_state.final_json:
                    json_str = json.dumps(st.session_state.final_json, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            with col_stats:
                total_fields = len(st.session_state.field_data)
                accepted_fields = sum(1 for status in st.session_state.field_status.values() if status == "accepted")
                rejected_fields = total_fields - accepted_fields
                
                st.metric("Total Fields", total_fields)
                st.metric("Accepted", accepted_fields)
                st.metric("Rejected", rejected_fields)
            
            # Display extracted JSON
            st.divider()
            st.subheader("📄 Extracted JSON Data")
            if st.session_state.final_json:
                st.json(st.session_state.final_json)
            else:
                st.info("Submit your selections to generate JSON data")
    
    # Instructions
    with st.expander("ℹ️ How to use this tool", expanded=False):
        st.markdown("""
        1. **Upload** your PDF document using the uploader
        2. **Review** the detected fields and their confidence scores
        3. **Accept/Reject** fields using the buttons
        4. **Submit** your selections to generate JSON
        5. **Download** the JSON or annotated image
        
        The system uses multiple extraction methods:
        - PyMuPDF (direct text extraction)
        - pdfplumber (detailed text analysis)
        - Multiple OCR engines (Tesseract, EasyOCR, RapidOCR, PaddleOCR)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | Advanced PDF Field Extractor v1.0")

if __name__ == "__main__":
    main()