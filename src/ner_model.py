import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import PyPDF2
from collections import defaultdict, OrderedDict
import string
from typing import Dict, List, Tuple, Optional, Any, Set
import os
import re
import streamlit as st
import pandas as pd
import io
import time
import fitz  # PyMuPDF for coordinate extraction
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from dataclasses import dataclass

# Page configuration
st.set_page_config(
    page_title="PDF Field Extraction System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class BoundingBox:
    """Class to represent bounding box coordinates"""
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int = 0
    
    def width(self) -> float:
        return abs(self.x1 - self.x0)
    
    def height(self) -> float:
        return abs(self.y1 - self.y0)
    
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    def area(self) -> float:
        return self.width() * self.height()
    
    def to_dict(self) -> Dict:
        """Convert BoundingBox to a dictionary for JSON serialization"""
        return {
            'x0': self.x0,
            'y0': self.y0,
            'x1': self.x1,
            'y1': self.y1,
            'page_num': self.page_num
        }

@dataclass
class WordInfo:
    """Enhanced word information with coordinates"""
    index: int
    original: str
    clean: str
    normalized: str
    bbox: BoundingBox
    page_num: int
    length: int
    has_alpha: bool
    has_numeric: bool
    is_capitalized: bool
    is_all_caps: bool
    is_all_lower: bool
    is_separator: bool
    is_month: bool
    is_organization_word: bool
    is_email_like: bool
    is_currency: bool
    is_amount_like: bool
    is_date_like: bool
    is_code_like: bool
    is_phone_like: bool
    is_pan_like: bool
    is_gst_like: bool
    entity_types: List[Dict]
    confidence: float = 0.0

class TemplateConfig:
    """Template configuration class to manage different document templates"""
    
    def __init__(self, config_file_path: str = None):
        self.templates = {}
        self.current_template = None
        self.field_types = {
            "email": ["receipt_email", "email"],
            "name": ["receipt_name", "beneficiary_name", "remitter_name", "seller_name", "buyer_name", "name"],
            "bank": ["beneficiary_bank", "remitting_bank", "bank"],
            "amount": ["debit_amount", "remittance_amount", "total_amount", "amount"],
            "date": ["advice_date", "value_date", "order_date", "invoice_date", "due_date", "date"],
            "reference": ["advice_ref", "customer_reference", "instruction_reference", "other_reference", "order_number", "invoice_number", "reference"],
            "account": ["account_number"],
            "type": ["transaction_type", "sub_payment_type", "invoice_type", "payment_method", "document_type"],
            "fee": ["handling_fee", "fee"],
            "info": ["remitter_to_beneficiary_info", "invoice_details", "details", "info"],
            "address": ["billing_address", "shipping_address", "seller_address", "buyer_address", "ship_from_address", "bill_to", "ship_to", "sold_by", "address"],
            "place": ["place_of_supply", "place_of_delivery", "place"],
            "code": ["state_ut_code", "pan_number", "gst_number", "seller_gst_number", "buyer_gst_number", "gstin", "state_code", "code"],
            "phone": ["phone_number", "phone"]
        }
        self.currency_indicators = ['INR', 'USD', 'EUR', 'GBP', '$', '₹', 'Rs', 'rs']
        
        # Enhanced regex patterns
        self.pan_pattern = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b')
        self.gst_pattern = re.compile(r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b')
        self.pan_loose_pattern = re.compile(r'[A-Z]{5}\d{4}[A-Z]')
        self.gst_loose_pattern = re.compile(r'\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]')
        
        if config_file_path and os.path.exists(config_file_path):
            self.load_from_file(config_file_path)
        else:
            self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default templates"""
        
        # Payment Advice Template
        self.templates["payment_advice"] = {
            "field_order": [
                "advice_date", "advice_ref", "receipt_name", "receipt_email", "transaction_type",
                "sub_payment_type", "beneficiary_name", "beneficiary_bank", "account_number",
                "customer_reference", "debit_amount", "remittance_amount", "handling_fee",
                "value_date", "remitter_name", "remitting_bank", "instruction_reference",
                "other_reference", "remitter_to_beneficiary_info"
            ],
            "head_keys": {
                "Advice sending date": "advice_date",
                "Advice reference no": "advice_ref",
                "Recipient's name": "receipt_name",
                "Recipient's email": "receipt_email",
                "Transaction type": "transaction_type",
                "Sub payment type": "sub_payment_type",
                "Beneficiary's name": "beneficiary_name",
                "Beneficiary's bank": "beneficiary_bank",
                "Beneficiary's account": "account_number",
                "Customer reference": "customer_reference",
                "Debit amount": "debit_amount",
                "Remittance amount": "remittance_amount",
                "Handling fee of remitting bank": "handling_fee",
                "Value date": "value_date",
                "Remitter's name": "remitter_name",
                "Remitting bank": "remitting_bank",
                "Instruction reference": "instruction_reference",
                "Other reference": "other_reference",
                "Remitter to beneficiary information": "remitter_to_beneficiary_info"
            },
            "trigger_words": {
                "advice_date": ["advice", "sending", "date"],
                "advice_ref": ["advice", "reference", "no"],
                "receipt_name": ["recipient", "name"],
                "receipt_email": ["recipient", "email"],
                "transaction_type": ["transaction", "type"],
                "sub_payment_type": ["sub", "payment", "type"],
                "beneficiary_name": ["beneficiary", "name"],
                "beneficiary_bank": ["beneficiary", "bank"],
                "account_number": ["beneficiary", "account"],
                "customer_reference": ["customer", "reference"],
                "debit_amount": ["debit", "amount"],
                "remittance_amount": ["remittance", "amount"],
                "handling_fee": ["handling", "fee", "remitting", "bank"],
                "value_date": ["value", "date"],
                "remitter_name": ["remitter", "name"],
                "remitting_bank": ["remitting", "bank"],
                "instruction_reference": ["instruction", "reference"],
                "other_reference": ["other", "reference"],
                "remitter_to_beneficiary_info": ["remitter", "to", "beneficiary", "information"]
            },
            "detection_keywords": [
                ('advice', 3), ('remittance', 3), ('beneficiary', 2), ('remitter', 2)
            ],
            "output_structure": "flat"
        }
        
        # Amazon Invoice Template
        self.templates["amazon_invoice"] = {
            "field_order": [
                "document_type", "sold_by", "pan_number", "gst_number", "billing_address", "shipping_address",
                "place_of_supply", "place_of_delivery", "order_number", "order_date",
                "invoice_number", "invoice_details", "invoice_date"
            ],
            "head_keys": {
                "Tax Invoice": "document_type",
                "Bill of Supply": "document_type",
                "Cash Memo": "document_type",
                "Seller": "sold_by",
                "Sold by": "sold_by",
                "PAN No": "pan_number",
                "PAN Number": "pan_number", 
                "PAN": "pan_number",
                "GST Registration No": "gst_number",
                "GSTIN": "gst_number",
                "GST No": "gst_number",
                "Order Number": "order_number",
                "Order Date": "order_date",
                "Invoice Number": "invoice_number",
                "Invoice Date": "invoice_date",
                "Invoice Details": "invoice_details",
                "Billing Address": "billing_address",
                "Shipping Address": "shipping_address",
                "Place of supply": "place_of_supply",
                "Place of delivery": "place_of_delivery",
                "State/UT Code": "state_ut_code",
                "Bill To": "billing_address",
                "Ship To": "shipping_address"
            },
            "trigger_words": {
                "document_type": ["tax", "invoice", "bill", "supply", "cash", "memo"],
                "sold_by": ["sold", "by", "seller"],
                "pan_number": ["pan", "no", "number"],
                "gst_number": ["gst", "registration", "no", "gstin"],
                "order_number": ["order", "number"],
                "order_date": ["order", "date"],
                "invoice_number": ["invoice", "number"],
                "invoice_date": ["invoice", "date"],
                "invoice_details": ["invoice", "details"],
                "billing_address": ["billing", "address", "bill", "to"],
                "shipping_address": ["shipping", "address", "ship", "to"],
                "place_of_supply": ["place", "of", "supply"],
                "place_of_delivery": ["place", "of", "delivery"],
                "state_ut_code": ["state", "ut", "code"]
            },
            "detection_keywords": [
                ('amazon', 4), ('sold by', 3), ('pan no', 2), ('order number', 2), ('tiger pug', 3)
            ],
            "output_structure": "nested",
            "output_formatter": "amazon_formatter"
        }
        
        # Meesho Invoice Template
        self.templates["meesho_invoice"] = {
            "field_order": [
                "invoice_type", "order_number", "order_date", "invoice_number",
                "invoice_date", "bill_to", "ship_to", "place_of_supply"
            ],
            "head_keys": {
                "Tax Invoice": "invoice_type",
                "Original For Recipient": "invoice_type",
                "Order Number": "order_number",
                "Order Date": "order_date",
                "Invoice Number": "invoice_number",
                "Invoice Date": "invoice_date",
                "BILL TO": "bill_to",
                "SHIP TO": "ship_to",
                "Place of Supply": "place_of_supply"
            },
            "trigger_words": {
                "invoice_type": ["tax", "invoice", "original", "for", "recipient"],
                "order_number": ["order", "number"],
                "order_date": ["order", "date"],
                "invoice_number": ["invoice", "number"],
                "invoice_date": ["invoice", "date"],
                "bill_to": ["bill", "to"],
                "ship_to": ["ship", "to"],
                "place_of_supply": ["place", "of", "supply"]
            },
            "detection_keywords": [
                ('meesho', 4), ('tax invoice', 3), ('original for recipient', 3), ('bill to', 2), ('ship to', 2)
            ],
            "output_structure": "flat"
        }
    
    def load_from_file(self, config_file_path: str):
        """Load template configuration from JSON file"""
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.templates = data.get('templates', {})
                self.field_types = data.get('field_types', self.field_types)
                print(f"✅ Loaded templates from {config_file_path}")
        except Exception as e:
            print(f"⚠️ Failed to load config file: {e}. Using default templates.")
            self._initialize_default_templates()
    
    def detect_template(self, text: str) -> str:
        """Auto-detect template from text content"""
        text_lower = text.lower()
        scores = {}
        
        for template_name, template_config in self.templates.items():
            score = 0
            detection_keywords = template_config.get('detection_keywords', [])
            
            for keyword, weight in detection_keywords:
                if keyword in text_lower:
                    score += weight
            
            scores[template_name] = score
        
        print(f"🔍 Template Detection Scores:")
        for template_name, score in scores.items():
            print(f"   {template_name}: {score}")
        
        if scores:
            max_template = max(scores, key=scores.get)
            max_score = scores[max_template]
            
            if max_score > 0:
                return max_template
        
        return list(self.templates.keys())[0] if self.templates else "payment_advice"
    
    def get_template(self, template_name: str) -> Dict:
        """Get template configuration by name"""
        return self.templates.get(template_name, {})
    
    def set_current_template(self, template_name: str):
        """Set the current active template"""
        if template_name in self.templates:
            self.current_template = template_name
            print(f"✅ Template set to: {template_name}")
        else:
            print(f"❌ Template {template_name} not found")
            raise ValueError(f"Unknown template type: {template_name}")
    
    def get_field_type(self, field_label: str) -> str:
        """Get field type for a given field label"""
        for field_type, field_list in self.field_types.items():
            if field_label in field_list:
                return field_type
        return "generic"

class EnhancedPDFExtractor:
    """Enhanced PDF field extraction system"""
    
    def __init__(self, config_file_path: str = None):
        self.template_config = TemplateConfig(config_file_path)
        self.ner_extractor = None
        self.current_template = None
        self.word_mapping = []
        self.field_positions = {}
        self.full_text = ""
        self.pdf_document = None
        self.detected_headers = {}
        
    def extract_text_with_coordinates(self, file_content: bytes) -> Tuple[str, List[WordInfo]]:
        """Extract text with coordinate information using PyMuPDF"""
        print("🔄 Step 2: Extracting text with coordinates...")
        
        try:
            # Open PDF document
            self.pdf_document = fitz.open(stream=file_content, filetype="pdf")
            text_parts = []
            word_mapping = []
            word_index = 0
            
            for page_num in range(len(self.pdf_document)):
                page = self.pdf_document[page_num]
                
                # Extract text blocks with coordinates
                text_dict = page.get_text("dict")
                
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    # Get bounding box
                                    bbox = BoundingBox(
                                        x0=span["bbox"][0],
                                        y0=span["bbox"][1],
                                        x1=span["bbox"][2],
                                        y1=span["bbox"][3],
                                        page_num=page_num
                                    )
                                    
                                    # Split text into words while preserving coordinates
                                    words = text.split()
                                    word_width = bbox.width() / len(words) if words else 0
                                    
                                    for i, word in enumerate(words):
                                        word_bbox = BoundingBox(
                                            x0=bbox.x0 + (i * word_width),
                                            y0=bbox.y0,
                                            x1=bbox.x0 + ((i + 1) * word_width),
                                            y1=bbox.y1,
                                            page_num=page_num
                                        )
                                        
                                        word_info = self._create_enhanced_word_info(
                                            word_index, word, word_bbox, page_num
                                        )
                                        
                                        word_mapping.append(word_info)
                                        text_parts.append(word)
                                        word_index += 1
            
            full_text = " ".join(text_parts)
            self.full_text = full_text
            self.word_mapping = word_mapping
            
            print(f"✅ Step 2: Extracted {len(text_parts)} words with coordinates from {len(self.pdf_document)} pages")
            return full_text, word_mapping
            
        except Exception as e:
            print(f"❌ Step 2 Error: {e}")
            return "", []
    
    def _create_enhanced_word_info(self, index: int, word: str, bbox: BoundingBox, page_num: int) -> WordInfo:
        """Create enhanced word information with coordinates"""
        word_clean = word.strip(string.punctuation)
        word_norm = word_clean.lower()
        
        months = {'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                  'january', 'february', 'march', 'april', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december'}
        org_words = {'bank', 'corp', 'company', 'ltd', 'limited', 'inc', 'pvt', 'services',
                     'private', 'public', 'group', 'industries', 'enterprise', 'retail', 'international'}
        
        return WordInfo(
            index=index,
            original=word,
            clean=word_clean,
            normalized=word_norm,
            bbox=bbox,
            page_num=page_num,
            length=len(word_clean),
            has_alpha=any(c.isalpha() for c in word_clean),
            has_numeric=any(c.isdigit() for c in word_clean),
            is_capitalized=word_clean[0].isupper() if word_clean else False,
            is_all_caps=word_clean.isupper() and len(word_clean) > 1,
            is_all_lower=word_clean.islower() and len(word_clean) > 1,
            is_separator=word in [':', '|', '-', '=', '_', ','],
            is_month=word_norm in months,
            is_organization_word=word_norm in org_words,
            is_email_like='@' in word and '.' in word and len(word) > 6,
            is_currency=any(curr in word.upper() for curr in self.template_config.currency_indicators),
            is_amount_like=self._is_amount_pattern(word),
            is_date_like=self._is_date_pattern(word),
            is_code_like=self._is_code_pattern(word),
            is_phone_like=self._is_phone_pattern(word),
            is_pan_like=self._is_pan_pattern(word_clean),
            is_gst_like=self._is_gst_pattern(word_clean),
            entity_types=[]
        )
    
    def _is_pan_pattern(self, word: str) -> bool:
        return bool(self.template_config.pan_pattern.fullmatch(word.upper().replace(' ', '').replace('-', '')))
    
    def _is_gst_pattern(self, word: str) -> bool:
        return bool(self.template_config.gst_pattern.fullmatch(word.upper().replace(' ', '').replace('-', '')))
    
    def _is_amount_pattern(self, word: str) -> bool:
        word_clean = word.strip(string.punctuation)
        has_currency = any(curr in word.upper() for curr in self.template_config.currency_indicators)
        has_amount_structure = (any(c.isdigit() for c in word_clean) and
                               (',' in word_clean or '.' in word_clean) and
                               len(word_clean) > 3)
        is_pure_numeric_amount = (word_clean.replace(',', '').replace('.', '').isdigit() and
                                 len(word_clean) >= 3)
        return has_currency or has_amount_structure or is_pure_numeric_amount
    
    def _is_date_pattern(self, word: str) -> bool:
        word_clean = word.strip(string.punctuation)
        date_separators = ['/', '.', '-']
        for sep in date_separators:
            if sep in word_clean and word_clean.count(sep) >= 1:
                parts = word_clean.split(sep)
                if len(parts) >= 2:
                    numeric_parts = 0
                    for part in parts:
                        if part.isdigit():
                            num = int(part) if part.isdigit() else 0
                            if (1 <= num <= 31) or (1 <= num <= 12) or (1900 <= num <= 2100):
                                numeric_parts += 1
                    if numeric_parts >= 2:
                        return True
        return False
    
    def _is_code_pattern(self, word: str) -> bool:
        word_clean = word.strip(string.punctuation)
        if (word_clean.isalnum() and
            any(c.isalpha() for c in word_clean) and
            any(c.isdigit() for c in word_clean) and
            6 <= len(word_clean) <= 20):
            return True
        return False
    
    def _is_phone_pattern(self, word: str) -> bool:
        word_clean = word.strip(string.punctuation + ' ')
        clean_digits = ''.join(c for c in word_clean if c.isdigit())
        if 10 <= len(clean_digits) <= 15:
            return True
        return False
    
    def initialize_ner_extractor(self):
        """Initialize NER model for entity extraction"""
        print("🔄 Step 3: Initializing NER model...")
        try:
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_extractor = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Step 3: BERT NER model loaded successfully")
        except Exception as e:
            print(f"⚠️ Step 3 Warning: Could not load BERT model: {e}")
            self.ner_extractor = None
    
    def detect_field_headers_with_bboxes(self) -> Dict[str, Dict]:
        """Detect field headers and create bounding boxes"""
        print("🔄 Step 4: Detecting field headers...")
        
        template = self.template_config.get_template(self.current_template)
        head_keys = template.get('head_keys', {})
        trigger_words = template.get('trigger_words', {})
        
        detected_headers = {}
        used_indices = set()
        
        # First pass: Direct header key matching
        print("   🔍 First pass: Direct header key matching...")
        for display_key, field_label in head_keys.items():
            match_result = self._find_direct_header_match_with_bbox(display_key, field_label, used_indices)
            if match_result:
                detected_headers[field_label] = match_result
                for i in range(match_result['start_index'], match_result['end_index'] + 1):
                    used_indices.add(i)
                print(f"   ✅ Direct match: {field_label} -> '{display_key}'")
        
        # Second pass: Trigger word matching
        print("   🔍 Second pass: Trigger word matching...")
        remaining_fields = [(k, v) for k, v in trigger_words.items() if k not in detected_headers]
        
        for field_label, triggers in remaining_fields:
            match_result = self._find_trigger_match_with_bbox(triggers, field_label, used_indices)
            if match_result:
                detected_headers[field_label] = match_result
                for i in range(match_result['start_index'], match_result['end_index'] + 1):
                    used_indices.add(i)
                print(f"   ✅ Trigger match: {field_label}")
        
        self.detected_headers = detected_headers
        
        print(f"✅ Step 4: Detected {len(detected_headers)} field headers")
        return detected_headers
    
    def _find_direct_header_match_with_bbox(self, header_key: str, field_label: str, used_indices: Set[int]) -> Optional[Dict]:
        """Find direct header matches with bounding box calculation"""
        header_words = header_key.lower().split()
        
        for start_idx in range(len(self.word_mapping) - len(header_words) + 1):
            if start_idx in used_indices:
                continue
            
            match_score = 0
            matched_indices = []
            
            for i, header_word in enumerate(header_words):
                word_idx = start_idx + i
                if word_idx >= len(self.word_mapping):
                    break
                
                word_info = self.word_mapping[word_idx]
                word_norm = word_info.normalized
                
                if header_word == word_norm:
                    match_score += 100
                    matched_indices.append(word_idx)
                elif header_word in word_norm or word_norm in header_word:
                    match_score += 70
                    matched_indices.append(word_idx)
            
            required_score = len(header_words) * 60
            if match_score >= required_score and len(matched_indices) >= len(header_words) * 0.7:
                # Calculate combined bounding box
                combined_bbox = self._calculate_combined_bbox(matched_indices)
                
                return {
                    'start_index': start_idx,
                    'end_index': start_idx + len(header_words) - 1,
                    'score': match_score,
                    'matched_indices': matched_indices,
                    'field_label': field_label,
                    'match_type': 'direct_header',
                    'bbox': combined_bbox,
                    'header_text': header_key
                }
        return None
    
    def _find_trigger_match_with_bbox(self, triggers: List[str], field_label: str, used_indices: Set[int]) -> Optional[Dict]:
        """Find trigger word matches with bounding box calculation"""
        best_match = None
        best_score = 0
        
        for start_idx in range(len(self.word_mapping)):
            if start_idx in used_indices:
                continue
            
            match_result = self._evaluate_trigger_sequence_with_bbox(triggers, start_idx, field_label)
            if match_result and match_result['score'] > best_score:
                overlap = any(i in used_indices for i in range(match_result['start_index'], match_result['end_index'] + 1))
                if not overlap:
                    best_score = match_result['score']
                    best_match = match_result
        
        return best_match
    
    def _evaluate_trigger_sequence_with_bbox(self, triggers: List[str], start_idx: int, field_label: str) -> Optional[Dict]:
        """Evaluate trigger word sequence match with bounding box calculation"""
        if start_idx >= len(self.word_mapping):
            return None
        
        score = 0
        matched_indices = []
        trigger_idx = 0
        current_idx = start_idx
        max_scan_distance = min(len(triggers) * 8, 40)
        consecutive_misses = 0
        max_misses = 3
        
        while trigger_idx < len(triggers) and current_idx < len(self.word_mapping):
            if current_idx - start_idx > max_scan_distance:
                break
            
            word_info = self.word_mapping[current_idx]
            current_trigger = triggers[trigger_idx].lower()
            word_norm = word_info.normalized
            
            if current_trigger == word_norm:
                score += 100
                matched_indices.append(current_idx)
                trigger_idx += 1
                consecutive_misses = 0
            elif len(word_norm) > 2 and current_trigger in word_norm:
                score += 80
                matched_indices.append(current_idx)
                trigger_idx += 1
                consecutive_misses = 0
            elif len(current_trigger) > 2 and word_norm in current_trigger:
                score += 60
                matched_indices.append(current_idx)
                trigger_idx += 1
                consecutive_misses = 0
            elif (word_info.is_separator or
                  word_info.length <= 2 or
                  word_norm in ['the', 'of', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'by']):
                pass
            else:
                consecutive_misses += 1
                if consecutive_misses >= max_misses:
                    break
            
            current_idx += 1
        
        completeness = trigger_idx / len(triggers) if len(triggers) > 0 else 0
        required_score = len(triggers) * 40
        proximity_bonus = max(0, 30 - (current_idx - start_idx - len(matched_indices)) * 2)
        final_score = score + proximity_bonus
        
        if final_score >= required_score and completeness >= 0.5 and matched_indices:
            combined_bbox = self._calculate_combined_bbox(matched_indices)
            trigger_text = " ".join([self.word_mapping[i].original for i in matched_indices])
            
            return {
                'start_index': start_idx,
                'end_index': current_idx - 1,
                'score': final_score,
                'matched_indices': matched_indices,
                'field_label': field_label,
                'completeness': completeness,
                'trigger_matches': trigger_idx,
                'bbox': combined_bbox,
                'header_text': trigger_text
            }
        return None
    
    def _calculate_combined_bbox(self, word_indices: List[int]) -> BoundingBox:
        """Calculate combined bounding box for multiple words"""
        if not word_indices:
            return BoundingBox(0, 0, 0, 0)
        
        word_bboxes = [self.word_mapping[i].bbox for i in word_indices if i < len(self.word_mapping)]
        
        if not word_bboxes:
            return BoundingBox(0, 0, 0, 0)
        
        # Find min and max coordinates
        min_x0 = min(bbox.x0 for bbox in word_bboxes)
        min_y0 = min(bbox.y0 for bbox in word_bboxes)
        max_x1 = max(bbox.x1 for bbox in word_bboxes)
        max_y1 = max(bbox.y1 for bbox in word_bboxes)
        
        # Use page number from first word
        page_num = word_bboxes[0].page_num
        
        return BoundingBox(min_x0, min_y0, max_x1, max_y1, page_num)
    
    def extract_field_values_with_spatial_context(self) -> Dict[str, Dict]:
        """Extract field values using spatial context from bounding boxes"""
        print("🔄 Step 5: Extracting field values...")
        
        results = {}
        
        for field_label, header_info in self.detected_headers.items():
            print(f"   🔍 Extracting {field_label}...")
            
            header_bbox = header_info['bbox']
            field_type = self.template_config.get_field_type(field_label)
            
            # Find candidate value words based on spatial proximity
            candidate_words = self._find_spatially_adjacent_words(header_bbox, field_type)
            
            # Extract value using field-specific logic
            extracted_value, value_bbox = self._extract_value_from_candidates(
                candidate_words, field_label, field_type
            )
            
            results[field_label] = {
                'value': extracted_value.strip(),
                'header_bbox': header_bbox,
                'value_bbox': value_bbox,
                'header_text': header_info['header_text'],
                'confidence': header_info['score'] / 100.0,
                'extraction_method': 'spatial_context'
            }
            
            print(f"   ✅ {field_label}: '{extracted_value.strip()}'")
        
        print(f"✅ Step 5: Extracted {len(results)} field values")
        return results
    
    def _find_spatially_adjacent_words(self, header_bbox: BoundingBox, field_type: str) -> List[WordInfo]:
        """Find words spatially adjacent to header based on field type"""
        candidates = []
        
        # Define search parameters based on field type
        if field_type in ["date", "code", "reference"]:
            max_distance = 200
            max_candidates = 10
        elif field_type in ["amount"]:
            max_distance = 150
            max_candidates = 5
        elif field_type in ["name", "address"]:
            max_distance = 300
            max_candidates = 20
        else:
            max_distance = 250
            max_candidates = 15
        
        # Search for words on the same page
        for word_info in self.word_mapping:
            if word_info.page_num != header_bbox.page_num:
                continue
            
            # Calculate distance from header
            distance = self._calculate_spatial_distance(header_bbox, word_info.bbox)
            
            if distance <= max_distance:
                candidates.append((word_info, distance))
        
        # Sort by distance and return top candidates
        candidates.sort(key=lambda x: x[1])
        return [word_info for word_info, _ in candidates[:max_candidates]]
    
    def _calculate_spatial_distance(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate spatial distance between two bounding boxes"""
        # Calculate center points
        center1 = bbox1.center()
        center2 = bbox2.center()
        
        # Euclidean distance
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    def _extract_value_from_candidates(self, candidates: List[WordInfo], field_label: str, field_type: str) -> Tuple[str, BoundingBox]:
        """Extract value from candidate words based on field type"""
        if not candidates:
            return "", BoundingBox(0, 0, 0, 0)
        
        # Field-specific extraction logic
        if field_type == "email":
            return self._extract_email_from_candidates(candidates)
        elif field_type == "name":
            return self._extract_name_from_candidates(candidates, field_label)
        elif field_type == "date":
            return self._extract_date_from_candidates(candidates)
        elif field_type == "code":
            return self._extract_code_from_candidates(candidates, field_label)
        elif field_type == "reference":
            return self._extract_reference_from_candidates(candidates, field_label)
        elif field_type == "address":
            return self._extract_address_from_candidates(candidates)
        elif field_type == "amount":
            return self._extract_amount_from_candidates(candidates)
        elif field_type == "phone":
            return self._extract_phone_from_candidates(candidates)
        else:
            return self._extract_generic_from_candidates(candidates)
    
    def _extract_email_from_candidates(self, candidates: List[WordInfo]) -> Tuple[str, BoundingBox]:
        """Extract email from candidates"""
        for word_info in candidates:
            if word_info.is_email_like:
                return word_info.clean, word_info.bbox
        return "", BoundingBox(0, 0, 0, 0)
    
    def _extract_name_from_candidates(self, candidates: List[WordInfo], field_label: str) -> Tuple[str, BoundingBox]:
        """Extract name from candidates"""
        name_words = []
        
        for word_info in candidates:
            # Stop conditions
            if (word_info.is_amount_like or word_info.is_currency or
                word_info.is_date_like or word_info.is_email_like):
                break
            
            # Include conditions
            if (word_info.is_capitalized and word_info.has_alpha and word_info.length > 1) or \
               (word_info.is_all_caps and word_info.length > 2) or \
               word_info.is_organization_word or \
               word_info.normalized in ['and', 'of', 'the', '&', ',']:
                name_words.append(word_info)
            
            # Limit number of words
            if len(name_words) >= 10:
                break
        
        if name_words:
            combined_text = " ".join([w.original for w in name_words])
            combined_bbox = self._calculate_combined_bbox([w.index for w in name_words])
            return combined_text, combined_bbox
        
        return "", BoundingBox(0, 0, 0, 0)
    
    def _extract_date_from_candidates(self, candidates: List[WordInfo]) -> Tuple[str, BoundingBox]:
        """Extract date from candidates with enhanced multi-word support"""
        date_words = []
        
        # First pass: Look for complete date patterns in a single word
        for word_info in candidates:
            if word_info.is_date_like:
                return word_info.clean, word_info.bbox
        
        # Second pass: Look for date components (day, month, year)
        for word_info in candidates:
            # Check if word is part of a date
            if (word_info.is_month or 
                word_info.has_numeric or  # Day or year
                word_info.normalized in ['st', 'nd', 'rd', 'th']):  # Day suffixes
                date_words.append(word_info)
            
            # Stop if we encounter a word that's clearly not part of a date
            if (word_info.is_amount_like or word_info.is_currency or 
                word_info.is_email_like or word_info.is_code_like):
                break
            
            # Limit to reasonable date length (e.g., "11th October 2024")
            if len(date_words) >= 5:
                break
        
        if date_words:
            # Try to form a complete date from the collected words
            combined_text = " ".join([w.original for w in date_words])
            combined_bbox = self._calculate_combined_bbox([w.index for w in date_words])
            return combined_text, combined_bbox
        
        return "", BoundingBox(0, 0, 0, 0)
    
    def _extract_code_from_candidates(self, candidates: List[WordInfo], field_label: str) -> Tuple[str, BoundingBox]:
        """Extract code from candidates"""
        # Priority search for specific field types
        if field_label == "pan_number":
            for word_info in candidates:
                if word_info.is_pan_like:
                    return word_info.clean.upper(), word_info.bbox
        elif field_label in ["gst_number", "gstin", "seller_gst_number", "buyer_gst_number"]:
            for word_info in candidates:
                if word_info.is_gst_like:
                    return word_info.clean.upper(), word_info.bbox
        
        # General code pattern search
        for word_info in candidates:
            if word_info.is_code_like:
                return word_info.clean, word_info.bbox
        
        return "", BoundingBox(0, 0, 0, 0)
    
    def _extract_reference_from_candidates(self, candidates: List[WordInfo], field_label: str) -> Tuple[str, BoundingBox]:
        """Extract reference from candidates"""
        ref_words = []
        
        for word_info in candidates:
            # Include alphanumeric sequences and hyphenated references
            if ((word_info.has_alpha and word_info.has_numeric and word_info.length >= 6) or 
                (word_info.has_numeric and not word_info.has_alpha and word_info.length >= 8) or 
                ('-' in word_info.original and word_info.length >= 6)):
                ref_words.append(word_info)
            
            # Stop if we encounter a word that's clearly not part of a reference
            if (word_info.is_amount_like or word_info.is_currency or 
                word_info.is_date_like or word_info.is_email_like):
                break
            
            # Limit to reasonable reference length
            if len(ref_words) >= 5:
                break
        
        if ref_words:
            combined_text = " ".join([w.original for w in ref_words])
            combined_bbox = self._calculate_combined_bbox([w.index for w in ref_words])
            return combined_text, combined_bbox
        
        return "", BoundingBox(0, 0, 0, 0)
    
    def _extract_address_from_candidates(self, candidates: List[WordInfo]) -> Tuple[str, BoundingBox]:
        """Extract address from candidates"""
        address_words = []
        
        for word_info in candidates:
            # Stop conditions
            if (word_info.is_amount_like or word_info.is_currency):
                break
            
            # Include conditions
            if (word_info.has_alpha and word_info.length > 1) or \
               (word_info.has_numeric and word_info.length <= 6):
                address_words.append(word_info)
            
            # Limit to reasonable address length
            if len(address_words) >= 15:
                break
        
        if address_words:
            combined_text = " ".join([w.original for w in address_words])
            combined_bbox = self._calculate_combined_bbox([w.index for w in address_words])
            return combined_text, combined_bbox
        
        return "", BoundingBox(0, 0, 0, 0)
    
    def _extract_amount_from_candidates(self, candidates: List[WordInfo]) -> Tuple[str, BoundingBox]:
        """Extract amount from candidates"""
        amount_words = []
        
        for word_info in candidates:
            # Include currency indicators and numeric values
            if word_info.is_amount_like or (word_info.is_currency and word_info.has_numeric):
                amount_words.append(word_info)
            
            # Stop if we encounter a word that's clearly not part of an amount
            if (word_info.is_date_like or word_info.is_email_like or 
                word_info.is_code_like or word_info.is_phone_like):
                break
            
            # Limit to reasonable amount length (e.g., "INR 50,758.00")
            if len(amount_words) >= 3:
                break
        
        if amount_words:
            combined_text = " ".join([w.original for w in amount_words])
            combined_bbox = self._calculate_combined_bbox([w.index for w in amount_words])
            return combined_text, combined_bbox
        
        return "", BoundingBox(0, 0, 0, 0)
    
    def _extract_phone_from_candidates(self, candidates: List[WordInfo]) -> Tuple[str, BoundingBox]:
        """Extract phone from candidates"""
        phone_words = []
        
        for word_info in candidates:
            # Include phone-like patterns
            if word_info.is_phone_like:
                phone_words.append(word_info)
            
            # Stop if we encounter a word that's clearly not part of a phone number
            if (word_info.is_amount_like or word_info.is_currency or 
                word_info.is_date_like or word_info.is_email_like):
                break
            
            # Limit to reasonable phone length
            if len(phone_words) >= 3:
                break
        
        if phone_words:
            combined_text = " ".join([w.original for w in phone_words])
            combined_bbox = self._calculate_combined_bbox([w.index for w in phone_words])
            return combined_text, combined_bbox
        
        return "", BoundingBox(0, 0, 0, 0)
    
    def _extract_generic_from_candidates(self, candidates: List[WordInfo]) -> Tuple[str, BoundingBox]:
        """Extract generic value from candidates"""
        if candidates:
            # Take first few relevant words
            relevant_words = []
            for word_info in candidates[:5]:
                if word_info.length > 0 and not word_info.is_separator:
                    relevant_words.append(word_info)
            
            if relevant_words:
                combined_text = " ".join([w.original for w in relevant_words])
                combined_bbox = self._calculate_combined_bbox([w.index for w in relevant_words])
                return combined_text, combined_bbox
        
        return "", BoundingBox(0, 0, 0, 0)
    
    def extract_fields_with_autotagging(self, file_content: bytes, template_name: str = None) -> Dict:
        """Main extraction function with autotagging and bounding boxes for headers AND values"""
        print("🚀 Starting PDF field extraction...")
        
        # Step 1: Extract text with coordinates
        text, word_mapping = self.extract_text_with_coordinates(file_content)
        if not text:
            return {"error": "Failed to extract text from PDF"}
        
        # Step 2: Detect template
        if template_name:
            self.template_config.set_current_template(template_name)
        else:
            detected_template = self.template_config.detect_template(text)
            self.template_config.set_current_template(detected_template)
            print(f"📋 Detected template: {detected_template}")
        
        self.current_template = self.template_config.current_template
        
        # Step 3: Initialize NER extractor
        self.initialize_ner_extractor()
        
        # Step 4: Detect field headers with bounding boxes
        detected_headers = self.detect_field_headers_with_bboxes()
        
        # Step 5: Extract field values with spatial context
        extracted_data = self.extract_field_values_with_spatial_context()
        
        # Step 6: Prepare results - Convert BoundingBox objects to dictionaries
        serializable_extracted_fields = {}
        for field_label, field_data in extracted_data.items():
            serializable_extracted_fields[field_label] = {
                'value': field_data['value'],
                'header_text': field_data['header_text'],
                'confidence': field_data['confidence'],
                'extraction_method': field_data['extraction_method']
            }
        
        results = {
            'template': self.current_template,
            'extracted_fields': serializable_extracted_fields,
            'detected_headers': len(detected_headers)
        }
        
        print("✅ Extraction completed successfully!")
        return results

def main():
    st.markdown('<h1 class="main-header">PDF Field Extraction System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'extractor' not in st.session_state:
        st.session_state.extractor = EnhancedPDFExtractor()
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'current_template' not in st.session_state:
        st.session_state.current_template = None
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sub-header">Configuration</h2>', unsafe_allow_html=True)
    
    # Template selection
    template_options = list(st.session_state.extractor.template_config.templates.keys())
    selected_template = st.sidebar.selectbox(
        "Select Document Template",
        options=["Auto-detect"] + template_options,
        index=0
    )
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload a PDF document for field extraction"
    )
    
    # Extract button
    extract_button = st.sidebar.button(
        "Extract Fields",
        type="primary",
        use_container_width=True
    )
    
    # Clear session button
    if st.sidebar.button("Clear Session", key="clear_session", help="Clear all extracted data and start fresh"):
        st.session_state.extracted_data = None
        st.session_state.current_template = None
        st.rerun()
    
    # Main content area
    if uploaded_file is not None:
        st.markdown('<h2 class="sub-header">Document Preview</h2>', unsafe_allow_html=True)
        
        # Display file details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.metric("Type", uploaded_file.type)
        
        # Extraction process
        if extract_button:
            with st.spinner("Extracting fields..."):
                start_time = time.time()
                
                # Read file content
                file_content = uploaded_file.read()
                
                # Determine template
                template_to_use = None if selected_template == "Auto-detect" else selected_template
                
                try:
                    # Extract fields
                    result = st.session_state.extractor.extract_fields_with_autotagging(
                        file_content=file_content,
                        template_name=template_to_use
                    )
                    
                    extraction_time = time.time() - start_time
                    
                    if "error" in result:
                        st.error(f"Extraction Error: {result['error']}")
                        st.markdown('<div class="error-box">An error occurred during extraction. Please try again.</div>', unsafe_allow_html=True)
                    else:
                        # Store results in session state
                        st.session_state.extracted_data = result
                        st.session_state.current_template = result.get('template', 'unknown')
                        
                        # Display extraction summary
                        st.markdown('<h2 class="sub-header">Extraction Results</h2>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Template Used", st.session_state.current_template)
                        with col2:
                            st.metric("Fields Detected", result.get('detected_headers', 0))
                        with col3:
                            st.metric("Extraction Time", f"{extraction_time:.2f} seconds")
                        
                        # Display extracted fields in a table
                        st.markdown('<h3 class="sub-header">Extracted Fields</h3>', unsafe_allow_html=True)
                        
                        if 'extracted_fields' in result and result['extracted_fields']:
                            # Prepare data for display
                            field_data = []
                            for field_label, field_info in result['extracted_fields'].items():
                                field_data.append({
                                    "Field Label": field_label,
                                    "Extracted Value": field_info['value'],
                                    "Header Text": field_info['header_text'],
                                    "Confidence": f"{field_info['confidence']:.2f}",
                                    "Extraction Method": field_info['extraction_method']
                                })
                            
                            # Create DataFrame
                            df = pd.DataFrame(field_data)
                            
                            # Display table
                            st.dataframe(
                                df,
                                use_container_width=True,
                                column_config={
                                    "Field Label": st.column_config.TextColumn("Field Label", width="medium"),
                                    "Extracted Value": st.column_config.TextColumn("Extracted Value", width="large"),
                                    "Header Text": st.column_config.TextColumn("Header Text", width="medium"),
                                    "Confidence": st.column_config.ProgressColumn(
                                        "Confidence",
                                        help="Confidence score of extraction",
                                        format="%.2f",
                                        min_value=0,
                                        max_value=1
                                    ),
                                    "Extraction Method": st.column_config.TextColumn("Extraction Method", width="medium")
                                }
                            )
                            
                            # Download button for extracted data
                            json_data = json.dumps(result['extracted_fields'], indent=2)
                            st.download_button(
                                label="Download Extracted Data (JSON)",
                                data=json_data,
                                file_name=f"{uploaded_file.name}_extracted_fields.json",
                                mime="application/json"
                            )
                        else:
                            st.markdown('<div class="info-box">No fields were extracted from the document.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")
                    st.markdown('<div class="error-box">An error occurred during extraction. Please try again.</div>', unsafe_allow_html=True)
    else:
        # Initial state when no file is uploaded
        st.markdown("""
        <div class="info-box">
            <h3>How to use this tool</h3>
            <ol>
                <li>Upload a PDF document using the sidebar</li>
                <li>Select a document template or choose auto-detect</li>
                <li>Click "Extract Fields"</li>
                <li>View extracted fields</li>
                <li>Download results as needed</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()