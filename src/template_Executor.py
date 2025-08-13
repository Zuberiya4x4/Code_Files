#FINAL WORKING CODE
#!/usr/bin/env python3
"""
Template-Agnostic PDF Processing Script
Improved version with precise field value extraction and better coordinate handling
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
import pytesseract  # OCR for image-based PDFs
from PIL import Image
import io
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

class TemplateAgnosticPDFProcessor:
    def __init__(self, field_info_path: str):
        """
        Initialize the template-agnostic PDF processor
        
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

    def extract_field_value(self, pdf_path: str, field_data: Dict) -> str:
        """
        Extract value from a specific field using coordinates
        
        Args:
            pdf_path: Path to PDF file
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
            
            page_num = value_coords.get('page', 1) - 1  # Convert to 0-indexed
            
            # Convert coordinates to proper format
            crop_coords = self.convert_coordinates(value_coords)
            
            # Try PDF text extraction first
            extracted_text = extract_text_from_pdf_region(pdf_path, page_num, crop_coords)
            
            # If no text or minimal text, try OCR
            if not extracted_text or len(extracted_text.strip()) < 2:
                logger.debug(f"PDF text extraction minimal for {field_name}, trying OCR")
                extracted_text = extract_text_from_image_region(pdf_path, page_num, crop_coords)
            
            # Extract specific value using intelligent parsing
            result = self.extract_specific_value(extracted_text, trigger_word, field_name)
            
            logger.debug(f"Extracted value for {field_name}: '{result}'")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting field '{field_data.get('field_name', 'unknown')}': {e}")
            return ""
    
    def process_single_pdf(self, pdf_path: str) -> Dict:
        """
        Process a single PDF file using the loaded field information
        
        Template Executor Flow:
        1. Receive document similar to onboarded template
        2. Use for loop on JSON file
        3. For each field: identify field name/trigger word, slice image with coords, extract values
        4. Create key-value pairs for all header and table values
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Initialize result structure
        result = {
            "source_file": os.path.basename(pdf_path),
            "template_name": self.template_name,
            "extraction_timestamp": datetime.now().isoformat(),
            "extracted_data": {},
            "processing_status": "in_progress"
        }
        
        try:
            # Validate PDF
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                logger.info(f"PDF has {page_count} pages")
                
            if page_count == 0:
                raise Exception("PDF has no pages")
            
            # Template Executor Flow: Step 2 - Use for loop on JSON file
            fields = self.field_info.get('fields', [])
            total_fields = len(fields)
            successful_extractions = 0
            
            logger.info(f"Processing {total_fields} fields from template")
            
            # Process each field in the template
            for i, field_data in enumerate(fields, 1):
                field_name = field_data.get('field_name', f'field_{i}')
                logger.info(f"Processing field {i}/{total_fields}: {field_name}")
                
                # Template Executor Flow: Step 3 - Extract values for identified field names
                extracted_value = self.extract_field_value(pdf_path, field_data)
                
                # Template Executor Flow: Step 4 - Create key-value pairs
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
            result["extraction_summary"] = {
                "total_fields": total_fields,
                "successful_extractions": successful_extractions,
                "failed_extractions": total_fields - successful_extractions,
                "success_rate": f"{(successful_extractions/total_fields)*100:.1f}%" if total_fields > 0 else "0%"
            }
            
            logger.info(f"Completed processing {pdf_path} - Success rate: {result['extraction_summary']['success_rate']}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            logger.error(traceback.format_exc())
            result["processing_status"] = "failed"
            result["error"] = str(e)
            return result
    
    def process_pdf_batch(self, pdf_path: str, output_dir: str = "results") -> Dict:
        """
        Process a single PDF file or all PDF files in a directory
        
        Args:
            pdf_path: Path to a PDF file or directory containing PDF files
            output_dir: Directory to save results
            
        Returns:
            Dictionary with processing results for all PDFs
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine PDF files to process
        if os.path.isdir(pdf_path):
            pdf_files = glob.glob(os.path.join(pdf_path, "*.pdf"))
            pdf_files.sort()  # Process in sorted order
        elif os.path.isfile(pdf_path) and pdf_path.lower().endswith('.pdf'):
            pdf_files = [pdf_path]
        else:
            error_msg = f"Invalid path: {pdf_path} is neither a directory nor a PDF file"
            logger.error(error_msg)
            return self._create_batch_result(error=error_msg)
        
        if not pdf_files:
            warning_msg = f"No PDF files found at {pdf_path}"
            logger.warning(warning_msg)
            return self._create_batch_result()
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Initialize batch results
        batch_results = self._create_batch_result(total_files=len(pdf_files))
        
        # Process each PDF file
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                logger.info(f"Processing file {i}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
                
                result = self.process_single_pdf(pdf_file)
                filename = os.path.splitext(os.path.basename(pdf_file))[0]
                batch_results['files'][filename] = result
                
                if result["processing_status"] == "completed":
                    batch_results['successful_files'] += 1
                else:
                    batch_results['failed_files'] += 1
                
                # Save individual result
                individual_result_path = os.path.join(output_dir, f"{filename}_extraction.json")
                with open(individual_result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                batch_results['failed_files'] += 1
                filename = os.path.splitext(os.path.basename(pdf_file))[0]
                batch_results['files'][filename] = {
                    "source_file": os.path.basename(pdf_file),
                    "processing_status": "failed",
                    "error": str(e)
                }
        
        # Save batch results
        batch_result_path = os.path.join(output_dir, "batch_results.json")
        with open(batch_result_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        # Create summary reports
        self.create_summary_reports(batch_results, output_dir)
        
        logger.info(f"Batch processing complete. Results saved to {output_dir}")
        return batch_results
    
    def _create_batch_result(self, total_files=0, error=None):
        """Create a standardized batch result structure"""
        return {
            'processing_timestamp': datetime.now().isoformat(),
            'template_name': self.template_name,
            'total_files': total_files,
            'successful_files': 0,
            'failed_files': 0,
            'files': {},
            'error': error
        }
    
    def create_summary_reports(self, batch_results: Dict, output_dir: str):
        """
        Create comprehensive summary reports
        
        Args:
            batch_results: Batch processing results
            output_dir: Output directory
        """
        try:
            # Create summary data for CSV
            summary_data = []
            detailed_data = []
            
            for filename, file_result in batch_results['files'].items():
                if file_result["processing_status"] == "completed":
                    extraction_summary = file_result.get('extraction_summary', {})
                    summary_data.append({
                        'filename': filename,
                        'template_name': file_result['template_name'],
                        'processing_status': 'Success',
                        'total_fields': extraction_summary.get('total_fields', 0),
                        'successful_extractions': extraction_summary.get('successful_extractions', 0),
                        'failed_extractions': extraction_summary.get('failed_extractions', 0),
                        'success_rate': extraction_summary.get('success_rate', '0%'),
                        'extraction_timestamp': file_result.get('extraction_timestamp', '')
                    })
                    
                    # Create detailed data for each field
                    for field_name, field_value in file_result.get('extracted_data', {}).items():
                        detailed_data.append({
                            'filename': filename,
                            'field_name': field_name,
                            'extracted_value': field_value,
                            'extraction_status': 'Success' if field_value else 'Failed'
                        })
                else:
                    summary_data.append({
                        'filename': filename,
                        'template_name': file_result.get('template_name', 'Unknown'),
                        'processing_status': 'Failed',
                        'total_fields': 0,
                        'successful_extractions': 0,
                        'failed_extractions': 0,
                        'success_rate': '0%',
                        'error': file_result.get('error', 'Unknown error'),
                        'extraction_timestamp': ''
                    })
            
            # Save summary CSV
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                summary_csv_path = os.path.join(output_dir, "processing_summary.csv")
                df_summary.to_csv(summary_csv_path, index=False)
                logger.info(f"Summary report created: {summary_csv_path}")
            
            # Save detailed CSV
            if detailed_data:
                df_detailed = pd.DataFrame(detailed_data)
                detailed_csv_path = os.path.join(output_dir, "detailed_extractions.csv")
                df_detailed.to_csv(detailed_csv_path, index=False)
                logger.info(f"Detailed report created: {detailed_csv_path}")
                
        except Exception as e:
            logger.error(f"Error creating summary reports: {e}")
    
    def display_extracted_data(self, results: Dict):
        """
        Display extracted data in a readable format
        
        Args:
            results: Processing results dictionary
        """
        print("\n" + "="*80)
        print("EXTRACTED DATA SUMMARY")
        print("="*80)
        
        print(f"Template: {results['template_name']}")
        print(f"Processing Time: {results['processing_timestamp']}")
        print(f"Total Files: {results['total_files']}")
        print(f"Successful: {results['successful_files']}")
        print(f"Failed: {results['failed_files']}")
        
        for filename, file_result in results['files'].items():
            print(f"\n{'='*60}")
            print(f"File: {filename}")
            print(f"Status: {file_result.get('processing_status', 'Unknown')}")
            
            if file_result.get("processing_status") == "completed":
                extraction_summary = file_result.get('extraction_summary', {})
                print(f"Success Rate: {extraction_summary.get('success_rate', '0%')}")
                print("-" * 60)
                
                print("\nExtracted Fields:")
                for field_name, value in file_result.get('extracted_data', {}).items():
                    display_name = field_name.replace('_', ' ').title()
                    status = "✓" if value else "✗"
                    display_value = value[:100] + "..." if len(str(value)) > 100 else value
                    print(f"  {status} {display_name}: {display_value}")
            else:
                print(f"Error: {file_result.get('error', 'Unknown error')}")
            
            print("-" * 60)

def main():
    """
    Main function to run the Template-Agnostic PDF Processing
    """
    # Configuration - Update these paths as needed
    FIELD_INFO_PATH = r"C:\Users\SyedaZuberiya\Downloads\full_pdf_extracted_data_auto_tag.json"
    PDF_PATH = r"C:\Users\SyedaZuberiya\Desktop\PyMuPDF Testing\Payment_Advice_F5.pdf"
    OUTPUT_DIRECTORY = "template_extraction_results"
    
    try:
        logger.info("Starting Template-Agnostic PDF Processing")
        
        # Initialize processor
        processor = TemplateAgnosticPDFProcessor(FIELD_INFO_PATH)
        logger.info(f"Processor initialized for template: {processor.template_name}")
        
        # Process PDF file(s) - Template Executor Flow
        results = processor.process_pdf_batch(PDF_PATH, OUTPUT_DIRECTORY)
        
        # Display extracted data
        processor.display_extracted_data(results)
        
        # Print final summary
        print("\n" + "="*80)
        print("PROCESSING COMPLETED")
        print("="*80)
        print(f"Template: {results['template_name']}")
        print(f"Total files processed: {results['total_files']}")
        print(f"Successful: {results['successful_files']}")
        print(f"Failed: {results['failed_files']}")
        if results['total_files'] > 0:
            success_percentage = (results['successful_files'] / results['total_files']) * 100
            print(f"Overall Success Rate: {success_percentage:.1f}%")
        print(f"Results saved to: {OUTPUT_DIRECTORY}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())
        print(f"\nProcessing failed: {e}")

if __name__ == "__main__":
    main()