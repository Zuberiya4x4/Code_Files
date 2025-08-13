import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Dict, List, Tuple, Any
import pandas as pd
import os
# Configure page
st.set_page_config(
    page_title="PDF Header Field & Table Extractor",
    page_icon="📄",
    layout="wide"
)

# Initialize session state for selections
if 'field_selections' not in st.session_state:
    st.session_state.field_selections = {}
if 'table_selections' not in st.session_state:
    st.session_state.table_selections = {}
if 'json_output' not in st.session_state:
    st.session_state.json_output = {}

class PDFHeaderExtractor:
    def __init__(self):
        # Define expected header fields with more comprehensive patterns
        self.header_patterns = {
            "document_title": [
                r"payment\s*advice", r"remittance\s*advice", r"transfer\s*advice",
                r"payment\s*notification", r"wire\s*transfer", r"fund\s*transfer"
            ],
            "advice_date": [
                r"advice\s*sending\s*date", r"advice\s*date", r"date\s*of\s*advice", 
                r"sending\s*date", r"date\s*sent", r"transaction\s*date"
            ],
            "advice_ref": [
                r"advice\s*reference\s*no", r"advice\s*ref\s*no", r"advice\s*reference", 
                r"advice\s*ref", r"reference\s*number", r"ref\s*no"
            ],
            "recipient_name": [
                r"recipient'?s?\s*name\s*and\s*contact\s*information", r"recipient'?s?\s*name", 
                r"company\s*name", r"recipient", r"to", r"pay\s*to"
            ],
            "receipt_email": [
                r"receipt\s*email", r"email", r"e-mail", r"email\s*address",
                r"contact\s*email", r"notification\s*email"
            ],
            "transaction_type": [
                r"transaction\s*type", r"payment\s*type", r"transfer\s*type"
            ],
            "sub_payment_type": [
                r"sub\s*payment\s*type", r"payment\s*sub\s*type", r"payment\s*method"
            ],
            "beneficiary_name": [
                r"beneficiary'?s?\s*name", r"beneficiary", r"payee", r"receiver"
            ],
            "beneficiary_bank": [
                r"beneficiary'?s?\s*bank", r"bank\s*name", r"receiving\s*bank",
                r"destination\s*bank", r"payee\s*bank"
            ],
            "account_number": [
                r"beneficiary'?s?\s*account", r"account\s*number", r"account\s*no", 
                r"a/c\s*no", r"acct\s*no", r"account"
            ],
            "customer_reference": [
                r"customer\s*reference", r"cust\s*ref", r"customer\s*ref", 
                r"client\s*reference", r"your\s*reference"
            ],
            "debit_amount": [
                r"debit\s*amount", r"amount\s*debited", r"total\s*debit", r"charged\s*amount",
                r"debit", r"amount\s*to\s*be\s*debited"
            ],
            "remittance_amount": [
                r"remittance\s*amount", r"remit\s*amount", r"transfer\s*amount", 
                r"payment\s*amount", r"net\s*amount", r"amount\s*to\s*be\s*remitted"
            ],
            "handling_fee": [
                r"handling\s*fee", r"charges", r"fees", r"service\s*charge", 
                r"processing\s*fee", r"transaction\s*fee"
            ],
            "value_date": [
                r"value\s*date", r"settlement\s*date", r"effective\s*date", 
                r"processing\s*date", r"cleared\s*date"
            ],
            "remitter_name": [
                r"remitter'?s?\s*name", r"sender\s*name", r"from", r"remitter", 
                r"payer", r"originator"
            ],
            "remitting_bank": [
                r"remitting\s*bank", r"sender\s*bank", r"of\s*remitting\s*bank", 
                r"originating\s*bank", r"from\s*bank"
            ],
            "instruction_reference": [
                r"instruction\s*reference", r"instr\s*ref", r"instruction\s*no", 
                r"wire\s*reference"
            ],
            "other_reference": [
                r"other\s*reference", r"additional\s*ref", r"misc\s*reference", 
                r"secondary\s*ref"
            ],
            "remitter_to_beneficiary_info": [
                r"remitter\s*to\s*beneficiary\s*information", r"remitter\s*to\s*beneficiary", 
                r"additional\s*info", r"payment\s*info", r"payment\s*details", 
                r"memo", r"purpose", r"description"
            ]
        }
        
        # Fields that should have separate header and value bounding boxes
        self.separate_box_fields = {
            "advice_date", "advice_ref", "recipient_name", "receipt_email",
            "beneficiary_name", "beneficiary_bank", "debit_amount", "remittance_amount",
            "account_number", "customer_reference", "handling_fee", "value_date",
            "remitter_name", "remitting_bank", "instruction_reference", "other_reference",
            "transaction_type", "sub_payment_type", "remitter_to_beneficiary_info"
        }
        
        # Patterns to exclude from detection (table headers, notes, etc.)
        self.exclusion_patterns = [
            r"document\s*number", r"page\s*\d+", r"\d+\s*of\s*\d+",
            r"important\s*note", r"note", r"disclaimer", r"terms\s*and\s*conditions",
            r"header", r"footer", r"confidential", r"private", r"security"
        ]
        
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (0, 128, 0), (128, 128, 0), (0, 0, 128),
            (128, 0, 0), (0, 128, 128), (192, 192, 192), (255, 20, 147),
            (32, 178, 170), (255, 69, 0), (154, 205, 50), (75, 0, 130)
        ]
        
        # Table detection color (purple for tables)
        self.table_color = (128, 0, 128)
        
        # Initialize selection dictionaries
        self.field_coordinates = {}
        self.table_coordinates = {}
    
    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with coordinates from PDF"""
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
                            if text and len(text) > 0:
                                text_blocks.append({
                                    "text": text,
                                    "bbox": span["bbox"],
                                    "page": page_num,
                                    "font_size": span["size"]
                                })
        
        doc.close()
        return text_blocks
    
    def detect_tables(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """Detect tables using pdfplumber"""
        tables_data = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.find_tables()
                tables_data[page_num] = []
                
                for table_index, table in enumerate(tables):
                    # Extract table data
                    table_data = table.extract()
                    
                    # Get table bounding box
                    bbox = table.bbox  # (x0, y0, x1, y1)
                    
                    # Convert table data to DataFrame
                    if table_data:
                        # Clean data - remove None values and empty strings
                        cleaned_data = []
                        for row in table_data:
                            cleaned_row = [cell if cell is not None else "" for cell in row]
                            cleaned_data.append(cleaned_row)
                        
                        # Create DataFrame
                        if len(cleaned_data) > 1:
                            df = pd.DataFrame(cleaned_data[1:], columns=cleaned_data[0])
                        else:
                            df = pd.DataFrame(cleaned_data)
                        
                        # Remove completely empty rows and columns
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        
                        table_id = f"table_{page_num}_{table_index}"
                        tables_data[page_num].append({
                            "table_id": table_id,
                            "table_index": table_index,
                            "bbox": bbox,
                            "dataframe": df,
                            "raw_data": cleaned_data,
                            "table_json": df.to_dict('records') if not df.empty else []
                        })
        
        return tables_data
    
    def is_table_or_unwanted_content(self, text: str, bbox: List[float]) -> bool:
        """Check if the text is part of table headers, notes, or unwanted content"""
        text_lower = text.lower().strip()
        
        # Check exclusion patterns
        for pattern in self.exclusion_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check if it's likely a table header (short text in specific positions)
        if len(text) < 15 and any(word in text_lower for word in ['amount', 'date', 'number', 'type']):
            # But allow if it matches our specific patterns
            for field_patterns in self.header_patterns.values():
                for pattern in field_patterns:
                    if re.search(f"\\b{pattern}\\b", text_lower):
                        return False
            return True
            
        # Check if it's in a table-like structure (multiple similar items in a row)
        if len(text) < 5 and text_lower in ['inr', 'usd', 'eur', 'gbp']:
            return True
            
        return False
    
    def is_likely_header(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the header patterns"""
        text_lower = text.lower()
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower).strip()
        
        for pattern in patterns:
            # Try exact pattern match
            if re.search(f"\\b{pattern}\\b", text_clean):
                return True
            # Try partial match for longer patterns
            if len(pattern.split()) > 2:
                pattern_words = pattern.split()
                if all(word in text_clean for word in pattern_words):
                    return True
        return False
    
    def is_amount_value(self, text: str) -> bool:
        """Check if text looks like a monetary amount"""
        text = text.strip()
        # Look for currency symbols and numbers
        currency_pattern = r'(?:INR|USD|EUR|GBP|₹|\$|€|£)?\s*[\d,]+\.?\d*'
        return bool(re.search(currency_pattern, text))
    
    def is_date_value(self, text: str) -> bool:
        """Check if text looks like a date"""
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}',
            r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}'
        ]
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def is_reference_value(self, text: str) -> bool:
        """Check if text looks like a reference number"""
        # Reference numbers often contain alphanumeric combinations
        if len(text) > 3 and any(char.isdigit() for char in text) and any(char.isalpha() for char in text):
            return True
        if len(text) > 5 and text.replace(' ', '').isalnum():
            return True
        return False
    
    def is_email_value(self, text: str) -> bool:
        """Check if text looks like an email address"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return bool(re.search(email_pattern, text))
    
    def is_transaction_type_value(self, text: str) -> bool:
        """Check if text looks like a transaction type"""
        transaction_keywords = [
            'swift', 'wire', 'neft', 'rtgs', 'eft', 'ach', 'sepa', 'domestic', 'international',
            'credit', 'debit', 'transfer', 'payment', 'remittance', 'urgent', 'regular'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in transaction_keywords) or len(text.strip()) > 3
    
    def is_payment_method_value(self, text: str) -> bool:
        """Check if text looks like a payment method/sub payment type"""
        payment_keywords = [
            'online', 'branch', 'phone', 'mobile', 'internet', 'atm', 'pos', 'check', 'cheque',
            'cash', 'card', 'electronic', 'manual', 'auto', 'scheduled', 'immediate'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in payment_keywords) or len(text.strip()) > 3
    
    def find_header_field_matches(self, text_blocks: List[Dict]) -> Dict[str, List[Dict]]:
        """Find potential header field matches with improved accuracy"""
        matches = {}
        
        for field_name, patterns in self.header_patterns.items():
            matches[field_name] = []
            
            # Special handling for document_title - look for specific document types
            if field_name == "document_title":
                for block in text_blocks:
                    text_lower = block["text"].lower()
                    # Look for payment advice specifically
                    if any(pattern in text_lower for pattern in patterns):
                        # Exclude if it contains unwanted content or is in table
                        if not self.is_table_or_unwanted_content(block["text"], block["bbox"]):
                            matches[field_name].append(block)
                continue
            
            # For other fields, use standard pattern matching
            for block in text_blocks:
                # Skip if it's table or unwanted content
                if self.is_table_or_unwanted_content(block["text"], block["bbox"]):
                    continue
                    
                if self.is_likely_header(block["text"], patterns):
                    matches[field_name].append(block)
        
        return matches
    
    def find_field_value(self, text_blocks: List[Dict], header_block: Dict, field_name: str) -> Tuple[str, List[float]]:
        """Find the value associated with a header field with improved search for all field types"""
        header_bbox = header_block["bbox"]
        page_num = header_block["page"]
        
        # Check if header already contains the value (colon pattern)
        header_text = header_block["text"]
        if ':' in header_text:
            parts = header_text.split(':', 1)
            if len(parts) == 2:
                value_part = parts[1].strip()
                if value_part:
                    return value_part, header_bbox
        
        # Look for values near the header
        potential_values = []
        ref_x, ref_y, ref_x2, ref_y2 = header_bbox
        
        for block in text_blocks:
            if block["page"] != page_num:
                continue
                
            block_text = block["text"].strip()
            if not block_text or block_text.lower() == header_text.lower():
                continue
            
            # Skip if it's another header
            is_other_header = False
            for other_field, other_patterns in self.header_patterns.items():
                if other_field != field_name and self.is_likely_header(block_text, other_patterns):
                    is_other_header = True
                    break
            
            if is_other_header:
                continue
                
            block_bbox = block["bbox"]
            block_x, block_y, block_x2, block_y2 = block_bbox
            
            # Calculate distances
            horizontal_distance = block_x - ref_x2
            vertical_distance = abs(block_y - ref_y)
            vertical_distance_below = block_y - ref_y2
            
            # Score based on position and content type
            base_score = float('inf')
            
            # Same line (to the right) - most common for header:value pairs
            if vertical_distance < 15 and horizontal_distance > -10 and horizontal_distance < 300:
                base_score = horizontal_distance + vertical_distance * 2
                
            # Below the header (next line)
            elif vertical_distance_below > 0 and vertical_distance_below < 50:
                horizontal_alignment = abs(block_x - ref_x)
                if horizontal_alignment < 100:  # Reasonable alignment
                    base_score = vertical_distance_below * 3 + horizontal_alignment
            
            # Extended search area for text-heavy fields
            elif field_name == "remitter_to_beneficiary_info" and vertical_distance_below > 0 and vertical_distance_below < 100:
                horizontal_alignment = abs(block_x - ref_x)
                if horizontal_alignment < 150:
                    base_score = vertical_distance_below * 4 + horizontal_alignment
            
            # Apply content-specific scoring
            if base_score != float('inf'):
                content_bonus = 0
                
                # Field-specific content validation
                if field_name in ["debit_amount", "remittance_amount", "handling_fee"]:
                    if self.is_amount_value(block_text):
                        content_bonus = -50  # Strong preference for amounts
                elif field_name in ["advice_date", "value_date"]:
                    if self.is_date_value(block_text):
                        content_bonus = -50  # Strong preference for dates
                elif field_name in ["advice_ref", "customer_reference", "instruction_reference", "other_reference"]:
                    if self.is_reference_value(block_text):
                        content_bonus = -30  # Preference for reference-like text
                elif field_name == "receipt_email":
                    if self.is_email_value(block_text):
                        content_bonus = -100  # Very strong preference for emails
                elif field_name == "transaction_type":
                    if self.is_transaction_type_value(block_text):
                        content_bonus = -30  # Preference for transaction type keywords
                elif field_name == "sub_payment_type":
                    if self.is_payment_method_value(block_text):
                        content_bonus = -30  # Preference for payment method keywords
                elif field_name == "remitter_to_beneficiary_info":
                    # For this field, prefer longer text blocks that look like descriptions
                    if len(block_text) > 10:
                        content_bonus = -20
                    if len(block_text) > 30:
                        content_bonus = -40
                
                final_score = base_score + content_bonus
                
                potential_values.append({
                    "block": block,
                    "score": final_score,
                    "text": block_text,
                    "bbox": block_bbox
                })
        
        if not potential_values:
            return "", header_bbox
        
        # Sort by score and take the best value
        potential_values.sort(key=lambda x: x["score"])
        best_value = potential_values[0]
        
        return best_value["text"], best_value["bbox"]
    
    def extract_field_data(self, text_blocks: List[Dict], header_matches: Dict) -> Dict[str, Dict]:
        """Extract field data with separate header and value bounding boxes where needed"""
        field_data = {}
        
        for field_name, headers in header_matches.items():
            if not headers:
                continue
                
            best_header = None
            best_score = float('inf')
            
            # Choose the best header match
            for header in headers:
                # Score based on text quality and position
                score = 0
                header_text = header["text"].lower()
                
                if len(header_text) > 100:  # Penalize very long text
                    score += 30
                if header["bbox"][1] > 600:  # Penalize text too far down (likely table)
                    score += 20
                    
                if score < best_score:
                    best_score = score
                    best_header = header
            
            if best_header:
                # Initialize field selection state
                st.session_state.field_selections[field_name] = True  # Default to selected
                
                # Determine if this field should have separate boxes
                if field_name in self.separate_box_fields:
                    # Find separate value
                    value_text, value_bbox = self.find_field_value(text_blocks, best_header, field_name)
                    
                    field_data[field_name] = {
                        "header_text": best_header["text"],
                        "header_bbox": best_header["bbox"],
                        "value_text": value_text,
                        "value_bbox": value_bbox,
                        "clean_value": value_text,
                        "has_separate_boxes": True
                    }
                else:
                    # Single box for combined header+value
                    complete_text = best_header["text"]
                    if ':' in complete_text:
                        value_part = complete_text.split(':', 1)[1].strip()
                    else:
                        value_part = ""
                    
                    field_data[field_name] = {
                        "complete_text": complete_text,
                        "clean_value": value_part,
                        "bbox": best_header["bbox"],
                        "has_separate_boxes": False
                    }
        
        return field_data
    
    def draw_bounding_boxes(self, pdf_path: str, field_data: Dict, tables_data: Dict = None) -> Image.Image:
        """Draw bounding boxes with interactive selection buttons"""
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)  # First page
        
        # Convert PDF page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        
        draw = ImageDraw.Draw(img)
        
        # Try to load fonts
        try:
            label_font = ImageFont.truetype("arial.ttf", 12)
            symbol_font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                label_font = ImageFont.load_default()
                symbol_font = ImageFont.load_default()
            except:
                label_font = None
                symbol_font = None
        
        # Draw table bounding boxes first (behind other elements)
        if tables_data and 0 in tables_data:
            for table_info in tables_data[0]:
                table_id = table_info["table_id"]
                bbox = table_info["bbox"]
                scaled_bbox = [coord * 2 for coord in bbox]  # Scale for 2x image
                
                # Draw table bounding box
                draw.rectangle(scaled_bbox, outline=self.table_color, width=6)
                
                # Add table label and selection button
                if label_font:
                    label_text = f"Table {table_info['table_index'] + 1}"
                    text_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Position label above the table
                    label_x = scaled_bbox[0]
                    label_y = max(5, scaled_bbox[1] - text_height - 35)
                    
                    # Background for label
                    padding = 4
                    draw.rectangle(
                        [label_x - padding, label_y - padding, 
                         label_x + text_width + padding, label_y + text_height + padding],
                        fill=self.table_color
                    )
                    draw.text((label_x, label_y), label_text, fill="white", font=label_font)
                    
                    # Draw selection button for table
                    button_size = 25
                    button_x = scaled_bbox[0]
                    button_y = max(5, scaled_bbox[1] - button_size - 5)
                    
                    # Draw selection button
                    button_bg = [button_x, button_y, button_x + button_size, button_y + button_size]
                    button_color = (0, 255, 0) if st.session_state.table_selections.get(table_id, True) else (255, 0, 0)
                    draw.rectangle(button_bg, fill=button_color)
                    
                    # Add symbol
                    symbol = "✓" if st.session_state.table_selections.get(table_id, True) else "✗"
                    if symbol_font:
                        draw.text((button_x + 5, button_y + 2), symbol, fill="white", font=symbol_font)
                    
                    # Store coordinates for click detection
                    self.table_coordinates[table_id] = {
                        "button_coords": button_bg,
                        "selected": st.session_state.table_selections.get(table_id, True)
                    }
        
        # Draw field bounding boxes
        color_index = 0
        for field_name, data in field_data.items():
            color = self.colors[color_index % len(self.colors)]
            is_selected = st.session_state.field_selections.get(field_name, True)
            
            # Use green for selected, red for unselected
            selection_color = (0, 255, 0) if is_selected else (255, 0, 0)
            
            if data.get("has_separate_boxes", False):
                # Draw separate boxes for header and value
                header_bbox = data["header_bbox"]
                value_bbox = data["value_bbox"]
                
                # Scale bounding boxes for 2x image
                scaled_header_bbox = [coord * 2 for coord in header_bbox]
                scaled_value_bbox = [coord * 2 for coord in value_bbox]
                
                # Draw header box with solid line
                line_width = 3 if is_selected else 1
                draw.rectangle(scaled_header_bbox, outline=color, width=line_width)
                
                # Draw value box with dashed effect if different
                if value_bbox != header_bbox:  # Only if different
                    for i in range(3 if is_selected else 1):
                        offset_bbox = [
                            scaled_value_bbox[0] + i,
                            scaled_value_bbox[1] + i,
                            scaled_value_bbox[2] + i,
                            scaled_value_bbox[3] + i
                        ]
                        draw.rectangle(offset_bbox, outline=color, width=1)
                
                # Add field label for header
                if label_font:
                    label_text = f"{field_name.replace('_', ' ').title()} (H)"
                    text_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Position label above the header box
                    label_x = scaled_header_bbox[0]
                    label_y = max(5, scaled_header_bbox[1] - text_height - 35)
                    
                    # Background for label
                    padding = 2
                    draw.rectangle(
                        [label_x - padding, label_y - padding, 
                         label_x + text_width + padding, label_y + text_height + padding],
                        fill=color
                    )
                    draw.text((label_x, label_y), label_text, fill="white", font=label_font)
                
                # Add selection button beside the header box
                if symbol_font:
                    button_size = 25
                    button_x = scaled_header_bbox[0]
                    button_y = max(5, scaled_header_bbox[1] - button_size - 5)
                    
                    # Draw selection button
                    button_bg = [button_x, button_y, button_x + button_size, button_y + button_size]
                    button_color = (0, 200, 0) if is_selected else (200, 0, 0)
                    draw.rectangle(button_bg, fill=button_color)
                    
                    # Add symbol
                    symbol = "✓" if is_selected else "✗"
                    draw.text((button_x + 5, button_y + 2), symbol, fill="white", font=symbol_font)
                    
                    # Store coordinates for click detection
                    self.field_coordinates[field_name] = {
                        "button_coords": button_bg,
                        "selected": is_selected
                    }
                    
                # Add value label if different bbox and has value
                if value_bbox != header_bbox and label_font and data["value_text"]:
                    value_label = f"{field_name.replace('_', ' ').title()} (V)"
                    text_bbox = draw.textbbox((0, 0), value_label, font=label_font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Position label above the value box
                    label_x = scaled_value_bbox[0]
                    label_y = max(5, scaled_value_bbox[1] - text_height - 5)
                    
                    # Background for label
                    padding = 2
                    draw.rectangle(
                        [label_x - padding, label_y - padding, 
                         label_x + text_width + padding, label_y + text_height + padding],
                        fill=(color[0]//2, color[1]//2, color[2]//2)  # Darker shade
                    )
                    draw.text((label_x, label_y), value_label, fill="white", font=label_font)
                    
            else:
                # Draw single box for combined field
                bbox = data["bbox"]
                scaled_bbox = [coord * 2 for coord in bbox]  # Scale for 2x image
                
                # Draw main bounding box with thick border
                line_width = 4 if is_selected else 2
                draw.rectangle(scaled_bbox, outline=color, width=line_width)
                
                # Add field label
                if label_font:
                    label_text = field_name.replace('_', ' ').title()
                    text_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Position label above the bounding box
                    label_x = scaled_bbox[0]
                    label_y = max(5, scaled_bbox[1] - text_height - 8)
                    
                    # Background for label
                    padding = 4
                    draw.rectangle(
                        [label_x - padding, label_y - padding, 
                         label_x + text_width + padding, label_y + text_height + padding],
                        fill=color
                    )
                    draw.text((label_x, label_y), label_text, fill="white", font=label_font)
            
            color_index += 1
        
        doc.close()
        return img
    
    def extract_json_data(self, field_data: Dict) -> Dict[str, str]:
        """Extract data as JSON format with cleaned values"""
        json_data = {}
        
        for field_name, data in field_data.items():
            clean_value = data.get("clean_value", "").strip()
            json_data[field_name] = clean_value
            
        return json_data
    
    def display_field_with_selection(self, field_data: Dict) -> Dict:
        """Display fields with selection options and real-time JSON output"""
        selected_fields = {}
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            for field_name, data in field_data.items():
                st.subheader(f"Field: {field_name.replace('_', ' ').title()}")
                
                # Create interactive button
                button_key = f"btn_{field_name}"
                is_selected = st.session_state.field_selections.get(field_name, True)
                
                # Create button with callback
                if st.button(
                    f"{'✓' if is_selected else '✗'} {field_name.replace('_', ' ').title()}",
                    key=button_key,
                    type="primary" if is_selected else "secondary"
                ):
                    # Toggle selection
                    st.session_state.field_selections[field_name] = not is_selected
                    # Rerun to update UI
                    st.rerun()
                
                st.markdown(f"**Value**: {data.get('clean_value', '')}")
                st.markdown("---")
        
        with col2:
            st.subheader("Current Selection")
            
            # Generate JSON based on current selections
            selected_json = {}
            for field_name, data in field_data.items():
                if st.session_state.field_selections.get(field_name, False):
                    selected_json[field_name] = data.get("clean_value", "")
            
            # Display JSON
            st.json(selected_json)
            
            # Update session state JSON output
            st.session_state.json_output = selected_json
        
        return selected_fields
    
    def process_selected_fields(self, field_data: Dict, selected_fields: Dict) -> Dict:
        """Process fields based on user selections"""
        processed_fields = {}
        
        for field_name, data in field_data.items():
            if selected_fields.get(field_name, False):
                processed_fields[field_name] = {
                    "clean_value": data.get("clean_value", ""),
                    "bbox": data.get("bbox", []),
                    "has_separate_boxes": data.get("has_separate_boxes", False)
                }
                if data.get("has_separate_boxes", False):
                    processed_fields[field_name].update({
                        "header_text": data.get("header_text", ""),
                        "header_bbox": data.get("header_bbox", []),
                        "value_text": data.get("value_text", ""),
                        "value_bbox": data.get("value_bbox", [])
                    })
        
        return processed_fields
    
    def extract_all_data(self, pdf_path: str) -> Dict[str, Any]:
        """Main extraction function that coordinates all extraction steps"""
        # Extract text blocks
        text_blocks = self.extract_text_blocks(pdf_path)
        
        # Detect tables
        tables_data = self.detect_tables(pdf_path)
        
        # Find header field matches
        header_matches = self.find_header_field_matches(text_blocks)
        
        # Extract field data
        field_data = self.extract_field_data(text_blocks, header_matches)
        
        # Extract JSON data
        json_data = self.extract_json_data(field_data)
        
        # Draw bounding boxes
        annotated_image = self.draw_bounding_boxes(pdf_path, field_data, tables_data)
        
        return {
            "field_data": field_data,
            "json_data": json_data,
            "tables_data": tables_data,
            "annotated_image": annotated_image
        }
    
    def handle_click(self, x: int, y: int):
        """Handle click events on the image"""
        # Check if click is on a field button
        for field_name, coords in self.field_coordinates.items():
            button_coords = coords["button_coords"]
            if (button_coords[0] <= x <= button_coords[2] and 
                button_coords[1] <= y <= button_coords[3]):
                # Toggle selection
                st.session_state.field_selections[field_name] = not coords["selected"]
                return
        
        # Check if click is on a table button
        for table_id, coords in self.table_coordinates.items():
            button_coords = coords["button_coords"]
            if (button_coords[0] <= x <= button_coords[2] and 
                button_coords[1] <= y <= button_coords[3]):
                # Toggle selection
                st.session_state.table_selections[table_id] = not coords["selected"]
                return

def main():
    st.title("📄 PDF Header Field & Table Extractor")
    st.markdown("Upload a PDF to extract payment advice information and detect tables")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Save uploaded file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize extractor
        extractor = PDFHeaderExtractor()
        
        # Show processing message
        with st.spinner("Processing PDF..."):
            # Extract data
            result = extractor.extract_all_data("temp.pdf")
        
        # Display results
        st.success("✅ Extraction completed!")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Annotated PDF (First Page)")
            st.image(result["annotated_image"], caption="PDF with detected fields and tables", use_column_width=True)
            
            # Add download button for annotated image
            buffered = io.BytesIO()
            result["annotated_image"].save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:application/octet-stream;base64,{img_str}" download="annotated_pdf.png">📥 Download Annotated Image</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Interactive Selection")
            st.markdown("Click the buttons below to toggle field selections:")
            
            # Display fields with interactive buttons
            selected_fields = extractor.display_field_with_selection(result["field_data"])
            
            st.markdown("---")
            
            # Display tables with selection
            st.subheader("Tables")
            if result["tables_data"] and 0 in result["tables_data"]:
                for i, table_info in enumerate(result["tables_data"][0]):
                    table_id = table_info["table_id"]
                    is_selected = st.session_state.table_selections.get(table_id, True)
                    
                    if st.button(
                        f"{'✓' if is_selected else '✗'} Table {i+1}",
                        key=f"table_{table_id}",
                        type="primary" if is_selected else "secondary"
                    ):
                        st.session_state.table_selections[table_id] = not is_selected
                        st.rerun()
            else:
                st.info("No tables detected in the first page")
        
        # Display final output
        st.subheader("Final JSON Output")
        st.json(st.session_state.json_output)

if __name__ == "__main__":
    main()