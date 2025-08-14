#!/usr/bin/env python3
"""
Template-Agnostic PDF and Image Processing with Streamlit UI
Combines document processing functionality with user-friendly interface
Supports PDF, image, and JSON files with two-column layout
"""
import json
import os
import glob
import pandas as pd
from datetime import datetime
import pdfplumber
import logging
from typing import Dict, List, Tuple, Any, Optional
import traceback
import fitz  # PyMuPDF for PDF to image conversion
import pytesseract  # OCR for image-based PDFs and images
from PIL import Image
import io
import re
import streamlit as st
import tempfile
import base64
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_image_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Helper functions
def is_image_file(file_path):
    """Check if the file is an image based on its extension"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)
def is_json_file(file_path):
    """Check if the file is a JSON based on its extension"""
    return file_path.lower().endswith('.json')
def clean_text(text):
    """Clean up extracted text by removing extra spaces and newlines"""
    if not text:
        return ""
    # Replace multiple spaces with a single space and clean up
    text = re.sub(r'\s+', ' ', text.strip())
    return text
def extract_text_from_pdf_region(pdf_path, page_num, crop_coords=None):
    """Extract text from a specific region of a PDF page"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                logger.warning(f"Page {page_num + 1} not found in PDF")
                return ""
            
            page = pdf.pages[page_num]
            if crop_coords:
                x1, y1, x2, y2 = crop_coords
                # Ensure coordinates are within page bounds
                page_width = page.width
                page_height = page.height
                x1 = max(0, min(x1, page_width))
                y1 = max(0, min(y1, page_height))
                x2 = max(x1, min(x2, page_width))
                y2 = max(y1, min(y2, page_height))
                
                bbox = (x1, y1, x2, y2)
                page = page.crop(bbox)
            
            text = page.extract_text()
            return clean_text(text) if text else ""
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""
def extract_text_from_image_region(pdf_path, page_num, crop_coords=None, dpi=300):
    """Extract text from a specific region of a PDF page using OCR"""
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            logger.warning(f"Page {page_num + 1} not found in PDF")
            doc.close()
            return ""
        
        page = doc[page_num]
        
        # Convert page to image with higher DPI for better OCR
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Crop image if coordinates provided
        if crop_coords:
            x1, y1, x2, y2 = crop_coords
            # Convert PDF coordinates to image coordinates
            img_width, img_height = img.size
            pdf_width = page.rect.width
            pdf_height = page.rect.height
            
            # Scale coordinates
            x1_img = int(x1 * img_width / pdf_width)
            y1_img = int(y1 * img_height / pdf_height)
            x2_img = int(x2 * img_width / pdf_width)
            y2_img = int(y2 * img_height / pdf_height)
            
            # Ensure crop coordinates are within image bounds
            x1_img = max(0, min(x1_img, img_width))
            y1_img = max(0, min(y1_img, img_height))
            x2_img = max(x1_img, min(x2_img, img_width))
            y2_img = max(y1_img, min(y2_img, img_height))
            
            img = img.crop((x1_img, y1_img, x2_img, y2_img))
        
        # Extract text using OCR with custom config
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@.-/:() '
        text = pytesseract.image_to_string(img, config=custom_config)
        doc.close()
        return clean_text(text) if text else ""
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return ""
def extract_text_from_image_region_direct(image_path, crop_coords=None):
    """Extract text from a specific region of an image using OCR"""
    try:
        # Open image
        img = Image.open(image_path)
        
        # Crop image if coordinates provided
        if crop_coords:
            x1, y1, x2, y2 = crop_coords
            img_width, img_height = img.size
            
            # Ensure crop coordinates are within image bounds
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(x1, min(x2, img_width))
            y2 = max(y1, min(y2, img_height))
            
            img = img.crop((x1, y1, x2, y2))
        
        # Extract text using OCR with custom config
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@.-/:() '
        text = pytesseract.image_to_string(img, config=custom_config)
        return clean_text(text) if text else ""
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return ""
def process_json_file(file_path: str, original_filename: str) -> Dict:
    """Process a JSON file and extract relevant information"""
    logger.info(f"Processing JSON file: {file_path}")
    
    result = {
        "source_file": original_filename,  # Use original filename
        "file_type": "JSON",
        "processing_status": "in_progress",
        "extraction_timestamp": datetime.now().isoformat(),
        "json_data": None,
        "extracted_data": {},
        "error": None
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        result["json_data"] = json_data
        
        # Try to extract common fields from JSON
        if isinstance(json_data, dict):
            # Look for common field patterns
            common_fields = [
                'name', 'title', 'subject', 'date', 'amount', 'reference', 
                'id', 'number', 'email', 'address', 'phone'
            ]
            
            for field in common_fields:
                # Look for exact match
                if field in json_data:
                    result["extracted_data"][field] = str(json_data[field])
                
                # Look for case-insensitive match
                for key, value in json_data.items():
                    if field.lower() == key.lower() and field not in result["extracted_data"]:
                        result["extracted_data"][field] = str(value)
            
            # Look for nested fields
            for key, value in json_data.items():
                if isinstance(value, dict):
                    for sub_field in common_fields:
                        if sub_field in value:
                            full_key = f"{key}_{sub_field}"
                            result["extracted_data"][full_key] = str(value[sub_field])
        
        result["processing_status"] = "completed"
        logger.info(f"Successfully processed JSON file: {file_path}")
        
    except Exception as e:
        logger.error(f"Error processing JSON file {file_path}: {e}")
        result["processing_status"] = "failed"
        result["error"] = str(e)
    
    return result
def get_pdf_page_as_image(pdf_path, page_num=0, zoom=2):
    """Convert a PDF page to an image"""
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            doc.close()
            return None
        
        page = doc[page_num]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        doc.close()
        return img_data
    except Exception as e:
        logger.error(f"Error converting PDF page to image: {e}")
        return None
# Main processor class
class TemplateAgnosticProcessor:
    def __init__(self, field_info_path: str):
        """
        Initialize the template-agnostic PDF and image processor
        
        Args:
            field_info_path: Path to the JSON file containing field information and coordinates
        """
        self.field_info = self.load_field_info(field_info_path)
        self.template_name = self.field_info.get('template_name', 'Unknown')
        
    def load_field_info(self, field_info_path: str) -> Dict:
        """Load field information from JSON file"""
        try:
            with open(field_info_path, 'r', encoding='utf-8') as f:
                field_info = json.load(f)
            
            logger.info(f"Successfully loaded field info for template: {field_info.get('template_name', 'Unknown')}")
            logger.info(f"Template contains {len(field_info.get('fields', []))} fields")
            return field_info
        except Exception as e:
            logger.error(f"Error loading field info: {e}")
            raise
    
    def convert_coordinates(self, coords: Dict) -> Tuple[float, float, float, float]:
        """
        Convert coordinates from JSON format to (x1, y1, x2, y2) format
        
        Args:
            coords: Coordinates in JSON format (x, y, width, height)
            
        Returns:
            Coordinates in (x1, y1, x2, y2) format
        """
        x = coords.get('x', 0)
        y = coords.get('y', 0)
        width = coords.get('width', 0)
        height = coords.get('height', 0)
        
        # Convert to (x1, y1, x2, y2) format
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height
        
        return (x1, y1, x2, y2)
    
    def extract_specific_value(self, raw_text: str, trigger_word: str = None, field_name: str = "") -> str:
        """
        Extract specific value from raw text using trigger words and patterns
        
        Args:
            raw_text: Raw extracted text
            trigger_word: Trigger word to look for
            field_name: Field name for context-specific extraction
            
        Returns:
            Cleaned specific value
        """
        if not raw_text:
            return ""
        
        text = clean_text(raw_text)
        logger.debug(f"Processing field '{field_name}' with raw text: '{text[:100]}...'")
        
        # If we have a trigger word, extract text after it
        if trigger_word:
            trigger_clean = clean_text(trigger_word)
            
            # Look for trigger word in the text
            trigger_pos = text.lower().find(trigger_clean.lower())
            if trigger_pos != -1:
                # Get text after the trigger word
                after_trigger = text[trigger_pos + len(trigger_clean):].strip()
                # Remove common separators
                after_trigger = re.sub(r'^[:.\-\s]+', '', after_trigger).strip()
                if after_trigger:
                    text = after_trigger
                    logger.debug(f"After trigger word extraction: '{text[:50]}...'")
        
        # Field-specific value extraction patterns
        if field_name.lower() in ['advice_date', 'value_date']:
            # Extract date patterns
            date_pattern = r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
            match = re.search(date_pattern, text, re.IGNORECASE)
            if match:
                return match.group().strip()
        
        elif field_name.lower() == 'advice_ref':
            # Specific pattern for advice reference - look for the reference after "reference no:"
            ref_patterns = [
                r'(?:reference\s+no[:.]?\s*)?([A-Z0-9]+-IN)\b',  # Pattern like A2cg67fZPXxK-IN
                r'([A-Z0-9]{8,}-IN)\b',                          # Alphanumeric followed by -IN
                r'([A-Z0-9]{10,20})\s*Payment\s*Advice',        # Reference before "Payment Advice"
                r'([A-Z0-9]{8,20})\b'                           # General alphanumeric pattern
            ]
            for pattern in ref_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    ref = match.group(1) if match.groups() else match.group()
                    if ref and ref.lower() not in ['payment', 'advice', 'page', 'jpmorgan']:
                        return ref.strip()
        
        elif field_name.lower() in ['instruction_reference', 'other_reference']:
            # Different patterns for different reference types
            if field_name.lower() == 'instruction_reference':
                # Look for alphanumeric codes like 90012BZ6EFR5
                ref_patterns = [
                    r'\b([A-Z0-9]{10,15})\b',
                    r'\b(\d{5}[A-Z0-9]{5,10})\b'
                ]
            else:  # other_reference
                # Look for HSBC reference patterns like HSBCN24285996982
                ref_patterns = [
                    r'\b(HSBC[A-Z0-9]{10,15})\b',
                    r'\b([A-Z]{4,6}\d{8,15})\b'
                ]
            
            for pattern in ref_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    ref = match.group(1).strip()
                    # Avoid duplicates between instruction and other reference
                    if field_name.lower() == 'other_reference' and not ref.startswith(('HSBC', 'hsbc')):
                        continue
                    if field_name.lower() == 'instruction_reference' and ref.startswith(('HSBC', 'hsbc')):
                        continue
                    return ref
        
        elif field_name.lower() == 'customer_reference':
            # Look for numeric references
            ref_match = re.search(r'\b(\d{5,10})\b', text)
            if ref_match:
                return ref_match.group(1).strip()
        
        elif field_name.lower() in ['debit_amount', 'remittance_amount']:
            # Extract currency amounts
            amount_patterns = [
                r'(INR\s*[\d,]+\.?\d*)',
                r'(USD\s*[\d,]+\.?\d*)',
                r'(EUR\s*[\d,]+\.?\d*)',
                r'([\d,]+\.?\d*)\s*INR',
            ]
            for pattern in amount_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        elif field_name.lower() in ['receipt_name', 'beneficiary_name', 'remitter_name']:
            # Extract company/person names
            # Remove email addresses first
            text_no_email = re.sub(r'\S+@\S+\.\S+', '', text)
            
            # Look for company name patterns
            name_patterns = [
                r'\b([A-Z][a-zA-Z\s&]+(?:Ltd|Pvt|Corp|Inc|Company|Services))\b',
                r'\b([A-Z][A-Z\s]+[A-Z])\b',  # All caps names
                r'\b([A-Z][a-zA-Z\s]+(?:\s[A-Z][a-zA-Z]+){1,3})\b'  # Title case names
            ]
            
            for pattern in name_patterns:
                matches = re.findall(pattern, text_no_email)
                if matches:
                    # Return the longest match
                    longest_match = max(matches, key=len).strip()
                    if len(longest_match) > 5:  # Reasonable name length
                        return longest_match
        
        elif field_name.lower() == 'receipt_email':
            # Extract email addresses
            email_match = re.search(r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b', text)
            if email_match:
                return email_match.group(1).strip()
        
        elif field_name.lower() == 'account_number':
            # Extract account numbers - look for CIFS pattern specifically
            account_patterns = [
                r'\b(CIFS\d+[X\*]+)\b',     # Pattern like CIFS105X1*****
                r'\b([A-Z]{4}\d+[X\*]+)\b', # General pattern like above
                r'\b(\d{8,20})\b',          # Numeric account numbers
            ]
            for pattern in account_patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()
        
        elif field_name.lower() == 'beneficiary_bank':
            # Extract bank names - prioritize specific bank names
            bank_patterns = [
                r'\b(JPMORGAN\s+CHASE\s+BANK(?:\s+[A-Z]+)?)\b',  # JPMORGAN CHASE BANK
                r'\b([A-Z][A-Za-z\s]*BANK[A-Za-z\s]*)\b',       # Other bank patterns
                r'\b(HSBC(?:\s+[A-Za-z]+)*)\b',                 # HSBC variations
            ]
            for pattern in bank_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    bank_name = match.group(1).strip()
                    # Clean up bank name
                    if 'JPMORGAN' in bank_name.upper():
                        return 'JPMORGAN CHASE BANK'
                    return bank_name
        
        elif field_name.lower() == 'remitting_bank':
            # Extract remitting bank names
            bank_patterns = [
                r'\b(HK\s+and\s+Shanghai\s+Banking\s+Corp\s+Ltd)\b',  # Full HSBC name
                r'\b(HSBC(?:\s+[A-Za-z]+)*)\b',                       # HSBC variations
                r'\b([A-Z][A-Za-z\s]*Bank(?:ing)?\s+Corp(?:oration)?[A-Za-z\s]*)\b',
                r'\b([A-Z][A-Za-z\s]+BANK[A-Za-z\s]*)\b',
            ]
            for pattern in bank_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    bank_name = match.group(1).strip()
                    # Clean up common bank name variations
                    if 'HK and Shanghai' in bank_name or 'HSBC' in bank_name:
                        return 'HK and Shanghai Banking Corp Ltd'
                    return bank_name
        
        # Default: return the first meaningful part of the text
        # Split by common delimiters and take the first substantial part
        parts = re.split(r'[:\n\r]+', text)
        for part in parts:
            part = part.strip()
            if len(part) > 2 and part.lower() not in ['page', 'recipient', 'name', 'information', 'advice', 'payment']:
                # Take first line if multi-line
                first_line = part.split('\n')[0].strip()
                if first_line:
                    return first_line
        
        return text.split('\n')[0].strip()  # Return first line as fallback
    
    def extract_field_value(self, file_path: str, field_data: Dict) -> str:
        """
        Extract value from a specific field using coordinates
        
        Args:
            file_path: Path to PDF or image file
            field_data: Field information including coordinates
            
        Returns:
            Extracted field value as string
        """
        try:
            field_name = field_data.get('field_name', 'unknown')
            trigger_word = field_data.get('trigger_word', '')
            logger.debug(f"Extracting field: {field_name}")
            
            # Get value coordinates - prioritize value_coordinates, fallback to coordinates
            value_coords = field_data.get('value_coordinates') or field_data.get('coordinates')
            if not value_coords:
                logger.error(f"No coordinates found for field: {field_name}")
                return ""
            
            # Convert coordinates to proper format
            crop_coords = self.convert_coordinates(value_coords)
            
            # Check if the file is an image
            if is_image_file(file_path):
                # For images, ignore page number and use direct image extraction
                extracted_text = extract_text_from_image_region_direct(file_path, crop_coords)
            else:
                # For PDFs, get page number
                page_num = value_coords.get('page', 1) - 1  # Convert to 0-indexed
                # Try PDF text extraction first
                extracted_text = extract_text_from_pdf_region(file_path, page_num, crop_coords)
                
                # If no text or minimal text, try OCR
                if not extracted_text or len(extracted_text.strip()) < 2:
                    logger.debug(f"PDF text extraction minimal for {field_name}, trying OCR")
                    extracted_text = extract_text_from_image_region(file_path, page_num, crop_coords)
            
            # Extract specific value using intelligent parsing
            result = self.extract_specific_value(extracted_text, trigger_word, field_name)
            
            logger.debug(f"Extracted value for {field_name}: '{result}'")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting field '{field_data.get('field_name', 'unknown')}': {e}")
            return ""
    
    def process_single_file(self, file_path: str, original_filename: str) -> Dict:
        """
        Process a single PDF or image file using the loaded field information
        
        Args:
            file_path: Path to PDF or image file
            original_filename: Original name of the uploaded file
            
        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Processing file: {file_path}")
        
        # Initialize result structure with original filename
        result = {
            "source_file": original_filename,  # Use original filename
            "template_name": self.template_name,
            "extraction_timestamp": datetime.now().isoformat(),
            "extracted_data": {},
            "processing_status": "in_progress"
        }
        
        try:
            # Validate file
            if is_image_file(file_path):
                logger.info("Processing as image file")
            elif file_path.lower().endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    page_count = len(pdf.pages)
                    logger.info(f"PDF has {page_count} pages")
                if page_count == 0:
                    raise Exception("PDF has no pages")
            else:
                raise Exception("Unsupported file format")
            
            # Process each field in the template
            fields = self.field_info.get('fields', [])
            total_fields = len(fields)
            successful_extractions = 0
            
            logger.info(f"Processing {total_fields} fields from template")
            
            for i, field_data in enumerate(fields, 1):
                field_name = field_data.get('field_name', f'field_{i}')
                logger.info(f"Processing field {i}/{total_fields}: {field_name}")
                
                # Extract values for identified field names
                extracted_value = self.extract_field_value(file_path, field_data)
                
                # Create key-value pairs
                if extracted_value:
                    result["extracted_data"][field_name] = extracted_value
                    successful_extractions += 1
                    logger.info(f"✓ Successfully extracted '{field_name}': '{extracted_value[:50]}{'...' if len(extracted_value) > 50 else ''}'")
                else:
                    result["extracted_data"][field_name] = ""
                    logger.warning(f"✗ Failed to extract '{field_name}'")
            
            # Add default receipt_email if not extracted and receipt_name contains email
            if 'receipt_email' not in result["extracted_data"] or not result["extracted_data"]['receipt_email']:
                receipt_name = result["extracted_data"].get('receipt_name', '')
                email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', receipt_name)
                if email_match:
                    result["extracted_data"]['receipt_email'] = email_match.group()
                elif 'receipt_email' not in result["extracted_data"]:
                    result["extracted_data"]['receipt_email'] = "AR.FS@compass-group.co.in"  # Default fallback
            
            # Update processing status
            result["processing_status"] = "completed"
            
            logger.info(f"Completed processing {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            logger.error(traceback.format_exc())
            result["processing_status"] = "failed"
            result["error"] = str(e)
            return result
# Streamlit UI
def main():
    st.set_page_config(
        page_title="PDF & Image Processor",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📄 Template-Agnostic Document Processing")
    st.markdown("""
    Upload a document and a JSON configuration to extract data.
    The document will be displayed on the left, and the extracted JSON data will appear on the right.
    """)
    
    # Initialize session state variables
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'template_info' not in st.session_state:
        st.session_state.template_info = None
    if 'document_file' not in st.session_state:
        st.session_state.document_file = None
    if 'config_file' not in st.session_state:
        st.session_state.config_file = None
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Document Upload")
        
        # Upload document
        document_file = st.file_uploader(
            "Upload Document (PDF or Image)", 
            type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "gif"],
            key="document_uploader"
        )
        
        # Upload configuration JSON
        config_file = st.file_uploader(
            "Upload Configuration JSON", 
            type=["json"],
            key="config_uploader"
        )
        
        # Process button
        process_button = st.button("Process Document", type="primary")
        
        # Display document preview
        if document_file:
            st.subheader("Document Preview")
            
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{document_file.name.split('.')[-1]}") as tmp:
                tmp.write(document_file.getvalue())
                tmp_path = tmp.name
            
            try:
                if is_image_file(document_file.name):
                    # Display image
                    st.image(document_file, use_column_width=True)
                elif document_file.name.lower().endswith('.pdf'):
                    # Display first page of PDF as image
                    img_data = get_pdf_page_as_image(tmp_path)
                    if img_data:
                        st.image(img_data, use_column_width=True)
                    else:
                        st.info("PDF document (preview not available)")
                
                # Clean up temporary file
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error displaying document: {e}")
                # Clean up temporary file if it exists
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    with col2:
        st.header("Extracted Data")
        
        # Processing section
        if process_button and document_file and config_file:
            try:
                # Save the uploaded files to temporary locations
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{document_file.name.split('.')[-1]}") as doc_tmp:
                    doc_tmp.write(document_file.getvalue())
                    doc_path = doc_tmp.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as config_tmp:
                    config_tmp.write(config_file.getvalue())
                    config_path = config_tmp.name
                
                # Initialize processor
                with st.spinner("Processing document..."):
                    processor = TemplateAgnosticProcessor(config_path)
                    
                    # Process the document with the original filename
                    if is_json_file(document_file.name):
                        result = process_json_file(doc_path, document_file.name)
                    else:
                        result = processor.process_single_file(doc_path, document_file.name)
                    
                    st.session_state.results = result
                    st.session_state.processing_complete = True
                
                # Clean up temporary files
                os.unlink(doc_path)
                os.unlink(config_path)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error(traceback.format_exc())
                
                # Clean up temporary files if they exist
                if 'doc_path' in locals() and os.path.exists(doc_path):
                    os.unlink(doc_path)
                if 'config_path' in locals() and os.path.exists(config_path):
                    os.unlink(config_path)
        
        # Display results
        if st.session_state.processing_complete and st.session_state.results:
            result = st.session_state.results
            
            if result["processing_status"] == "completed":
                # Display extracted data as JSON
                st.subheader("Extracted JSON Data")
                
                # Format extracted data for display
                extracted_data = result.get('extracted_data', {})
                
                # Create a clean JSON structure
                json_data = {
                    "source_file": result.get('source_file', ''),
                    "template_name": result.get('template_name', ''),
                    "extraction_timestamp": result.get('extraction_timestamp', ''),
                    "extracted_fields": extracted_data
                }
                
                # Display JSON
                st.json(json_data)
                
                # Download JSON button
                json_str = json.dumps(json_data, indent=2)
                st.download_button(
                    label="Download Extracted Data",
                    data=json_str,
                    file_name=f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
        else:
            st.info("Upload a document and configuration JSON, then click 'Process Document' to extract data.")
if __name__ == "__main__":
    main()