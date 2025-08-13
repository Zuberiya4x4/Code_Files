import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from PyPDF2 import PdfReader
from collections import defaultdict, OrderedDict
import string
from typing import Dict, List, Tuple, Optional, Any, Set
import os
import re
import streamlit as st
import pandas as pd
import io
import time

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
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .clear-session-btn {
        background-color: #dc3545 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 0.25rem !important;
        font-weight: bold !important;
        width: 100% !important;
        margin-top: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

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
        
        # Enhanced regex patterns for PAN and GST with more strict validation
        self.pan_pattern = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b')
        self.gst_pattern = re.compile(r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b')
        
        # Alternative patterns for finding PAN/GST in various formats
        self.pan_loose_pattern = re.compile(r'[A-Z]{5}\d{4}[A-Z]')
        self.gst_loose_pattern = re.compile(r'\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]')
        
        if config_file_path and os.path.exists(config_file_path):
            self.load_from_file(config_file_path)
        else:
            self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default templates from your specialized code"""
        
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
        
        # OMGS Invoice Template
        self.templates["omgs_invoice"] = {
            "field_order": [
                "invoice_number", "invoice_date", "order_number", "order_date",
                "payment_method", "bill_to", "ship_to", "sold_by", "gstin", "state_code"
            ],
            "head_keys": {
                "Invoice Number": "invoice_number",
                "Invoice": "invoice_number",
                "Invoice Date": "invoice_date",
                "Order Number": "order_number",
                "Order": "order_number",
                "Order Date": "order_date",
                "Payment Method": "payment_method",
                "Billing Address": "bill_to",
                "Bill To": "bill_to",
                "Ship To": "ship_to",
                "Sold By": "sold_by",
                "GSTIN No": "gstin",
                "GSTIN": "gstin",
                "State Code": "state_code"
            },
            "trigger_words": {
                "invoice_number": ["invoice", "number"],
                "invoice_date": ["invoice", "date"],
                "order_number": ["order", "number"],
                "order_date": ["order", "date"],
                "payment_method": ["payment", "method"],
                "bill_to": ["bill", "to"],
                "ship_to": ["ship", "to"],
                "sold_by": ["sold", "by"],
                "gstin": ["gstin"],
                "state_code": ["state", "code"]
            },
            "detection_keywords": [
                ('maxgrade', 4), ('omgs', 4), ('mgbg', 4), ('kssidc', 3), ('yelahanka', 3), ('razorpay', 2)
            ],
            "output_structure": "nested",
            "output_formatter": "omgs_formatter"
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
    
    def save_to_file(self, config_file_path: str):
        """Save current template configuration to JSON file"""
        try:
            data = {
                'templates': self.templates,
                'field_types': self.field_types
            }
            with open(config_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"✅ Saved templates to {config_file_path}")
        except Exception as e:
            print(f"❌ Failed to save config file: {e}")
    
    def add_template(self, template_name: str, template_config: Dict):
        """Add a new template configuration"""
        required_keys = ['field_order', 'head_keys', 'trigger_words', 'detection_keywords']
        if all(key in template_config for key in required_keys):
            self.templates[template_name] = template_config
            print(f"✅ Added template: {template_name}")
        else:
            print(f"❌ Template {template_name} missing required keys: {required_keys}")
    
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
        
        # Fallback detection logic
        for template_name, template_config in self.templates.items():
            detection_keywords = template_config.get('detection_keywords', [])
            for keyword, _ in detection_keywords:
                if keyword in text_lower:
                    return template_name
        
        # Default fallback
        return list(self.templates.keys())[0] if self.templates else "payment_advice"
    
    def get_field_type(self, field_label: str) -> str:
        """Get field type for a given field label"""
        for field_type, field_list in self.field_types.items():
            if field_label in field_list:
                return field_type
        return "generic"
    
    def extract_pan_from_text(self, text: str) -> str:
        """Extract PAN number from text using regex"""
        # Try strict pattern first
        match = self.pan_pattern.search(text)
        if match:
            return match.group()
        
        # Try loose pattern
        match = self.pan_loose_pattern.search(text.replace(' ', '').replace('-', ''))
        if match:
            return match.group()
        
        return ""
    
    def extract_gst_from_text(self, text: str) -> str:
        """Extract GST number from text using regex"""
        # Try strict pattern first
        match = self.gst_pattern.search(text)
        if match:
            return match.group()
        
        # Try loose pattern
        match = self.gst_loose_pattern.search(text.replace(' ', '').replace('-', ''))
        if match:
            return match.group()
        
        return ""

class GeneralizedPDFExtractor:
    """Generalized PDF field extraction system"""
    
    def __init__(self, config_file_path: str = None):
        self.template_config = TemplateConfig(config_file_path)
        self.ner_extractor = None
        self.current_template = None
        self.word_mapping = []
        self.field_positions = {}
        self.full_text = ""  # Store full text for global pattern matching
        
    def extract_text_from_pdf(self, file_path: str = None, file_content: bytes = None) -> str:
        """Extract text from PDF file or file content"""
        try:
            if file_content:
                pdf_reader = PdfReader(io.BytesIO(file_content))
            else:
                with open(file_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"{page_text}\n"
            
            self.full_text = text  # Store full text
            print(f"✅ Step 2: Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            print(f"❌ Step 2 Error: {e}")
            return ""
    
    def create_word_mapping(self, text: str) -> List[Dict]:
        """Create detailed word mapping with enhanced features"""
        print("🔄 Step 3: Creating word mapping...")
        words = text.split()
        word_mapping = []
        months = {'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                  'january', 'february', 'march', 'april', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december'}
        org_words = {'bank', 'corp', 'company', 'ltd', 'limited', 'inc', 'pvt', 'services',
                     'private', 'public', 'group', 'industries', 'enterprise', 'retail', 'international'}
        
        for i, word in enumerate(words):
            word_clean = word.strip(string.punctuation)
            word_norm = word_clean.lower()
            word_info = {
                'index': i,
                'original': word,
                'clean': word_clean,
                'normalized': word_norm,
                'length': len(word_clean),
                'has_alpha': any(c.isalpha() for c in word_clean),
                'has_numeric': any(c.isdigit() for c in word_clean),
                'is_capitalized': word_clean[0].isupper() if word_clean else False,
                'is_all_caps': word_clean.isupper() and len(word_clean) > 1,
                'is_all_lower': word_clean.islower() and len(word_clean) > 1,
                'is_separator': word in [':', '|', '-', '=', '_', ','],
                'is_month': word_norm in months,
                'is_organization_word': word_norm in org_words,
                'is_email_like': '@' in word and '.' in word and len(word) > 6,
                'is_currency': any(curr in word.upper() for curr in self.template_config.currency_indicators),
                'is_amount_like': self._is_amount_pattern(word),
                'is_date_like': self._is_date_pattern(word),
                'is_code_like': self._is_code_pattern(word),
                'is_phone_like': self._is_phone_pattern(word),
                'is_pan_like': self._is_pan_pattern(word_clean),
                'is_gst_like': self._is_gst_pattern(word_clean),
                'entity_types': []
            }
            word_mapping.append(word_info)
        
        print(f"✅ Step 3: Created mapping for {len(word_mapping)} words")
        return word_mapping
    
    def _is_pan_pattern(self, word: str) -> bool:
        """Check if word matches PAN pattern using fixed regex"""
        return bool(self.template_config.pan_pattern.fullmatch(word.upper().replace(' ', '').replace('-', '')))
    
    def _is_gst_pattern(self, word: str) -> bool:
        """Check if word matches GST pattern using fixed regex"""
        return bool(self.template_config.gst_pattern.fullmatch(word.upper().replace(' ', '').replace('-', '')))
    
    def _is_amount_pattern(self, word: str) -> bool:
        """Check if word matches amount pattern"""
        word_clean = word.strip(string.punctuation)
        has_currency = any(curr in word.upper() for curr in self.template_config.currency_indicators)
        has_amount_structure = (any(c.isdigit() for c in word_clean) and
                               (',' in word_clean or '.' in word_clean) and
                               len(word_clean) > 3)
        is_pure_numeric_amount = (word_clean.replace(',', '').replace('.', '').isdigit() and
                                 len(word_clean) >= 3)
        return has_currency or has_amount_structure or is_pure_numeric_amount
    
    def _is_date_pattern(self, word: str) -> bool:
        """Check if word matches date pattern"""
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
        """Check if word matches code pattern"""
        word_clean = word.strip(string.punctuation)
        if (word_clean.isalnum() and
            any(c.isalpha() for c in word_clean) and
            any(c.isdigit() for c in word_clean) and
            6 <= len(word_clean) <= 20):
            return True
        return False
    
    def _is_phone_pattern(self, word: str) -> bool:
        """Check if word matches phone pattern"""
        word_clean = word.strip(string.punctuation + ' ')
        clean_digits = ''.join(c for c in word_clean if c.isdigit())
        if 10 <= len(clean_digits) <= 15:
            return True
        return False
    
    def initialize_ner_extractor(self):
        """Initialize NER model for entity extraction"""
        print("🔄 Step 4: Initializing NER model...")
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
            print("✅ Step 4: BERT NER model loaded successfully")
        except Exception as e:
            print(f"⚠️ Step 4 Warning: Could not load BERT model: {e}")
            self.ner_extractor = None
    
    def extract_ner_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text"""
        if not self.ner_extractor:
            return []
        
        try:
            chunks = self._create_text_chunks(text, max_length=400)
            all_entities = []
            for chunk_start, chunk_text in chunks:
                entities = self.ner_extractor(chunk_text)
                for entity in entities:
                    entity['start'] += chunk_start
                    entity['end'] += chunk_start
                    entity['confidence'] = entity.get('score', 0.0)
                    all_entities.append(entity)
            return self._filter_entities(all_entities)
        except Exception as e:
            print(f"⚠️ NER extraction failed: {e}")
            return []
    
    def _create_text_chunks(self, text: str, max_length: int = 400) -> List[Tuple[int, str]]:
        """Create overlapping text chunks for NER processing"""
        words = text.split()
        chunks = []
        overlap = 50
        for i in range(0, len(words), max_length - overlap):
            chunk_words = words[i:i + max_length]
            chunk_text = " ".join(chunk_words)
            char_start = len(" ".join(words[:i])) + (1 if i > 0 else 0)
            chunks.append((char_start, chunk_text))
        return chunks
    
    def _filter_entities(self, entities: List[Dict]) -> List[Dict]:
        """Filter NER entities by confidence threshold"""
        filtered = []
        for entity in entities:
            if entity.get('confidence', 0) >= 0.3:
                entity_text = entity['word'].replace('##', '').strip()
                if len(entity_text) >= 2:
                    entity['cleaned_word'] = entity_text
                    filtered.append(entity)
        return filtered
    
    def map_ner_entities(self, ner_entities: List[Dict]):
        """Map NER entities to word indices"""
        print("🔄 Step 5: Mapping NER entities to word indices...")
        text_full = ' '.join([w['original'] for w in self.word_mapping])
        
        for entity in ner_entities:
            entity_text = entity.get('word', '').replace('##', '').strip()
            entity_start = entity.get('start', 0)
            entity_end = entity.get('end', 0)
            entity_type = entity.get('entity_group', entity.get('label', ''))
            confidence = entity.get('score', entity.get('confidence', 0))
            
            current_pos = 0
            for word_info in self.word_mapping:
                word_start = current_pos
                word_end = current_pos + len(word_info['original'])
                
                if not (entity_end <= word_start or entity_start >= word_end):
                    word_info['entity_types'].append({
                        'type': entity_type,
                        'confidence': confidence,
                        'text': entity_text
                    })
                
                current_pos = word_end + 1
        
        entities_mapped = sum(1 for w in self.word_mapping if w['entity_types'])
        print(f"✅ Step 5: Mapped {len(ner_entities)} entities to {entities_mapped} words")
    
    def find_field_headers(self) -> Dict[str, Dict]:
        """Find field headers using template configuration"""
        print("🔄 Step 6: Detecting field headers using trigger words and direct matching...")
        field_positions = {}
        used_indices = set()
        
        template = self.template_config.get_template(self.current_template)
        head_keys = template.get('head_keys', {})
        trigger_words = template.get('trigger_words', {})
        
        # First pass: Direct header key matching
        print("   🔍 First pass: Direct header key matching...")
        for display_key, field_label in head_keys.items():
            direct_match = self._find_direct_header_match(display_key, field_label, used_indices)
            if direct_match:
                field_positions[field_label] = direct_match
                for i in range(direct_match['start_index'], direct_match['end_index'] + 1):
                    used_indices.add(i)
                print(f"   ✅ Direct match: {field_label} -> '{display_key}'")
        
        # Second pass: Trigger word matching
        print("   🔍 Second pass: Trigger word matching...")
        remaining_fields = [(k, v) for k, v in trigger_words.items() if k not in field_positions]
        sorted_fields = sorted(remaining_fields, key=lambda x: len(x[1]), reverse=True)
        
        for field_label, triggers in sorted_fields:
            print(f"      🔍 Searching for {field_label} with triggers: {triggers}")
            best_match = self._find_best_trigger_match(triggers, field_label, used_indices)
            if best_match:
                field_positions[field_label] = best_match
                for i in range(best_match['start_index'], best_match['end_index'] + 1):
                    used_indices.add(i)
                print(f"      ✅ Found {field_label} at indices {best_match['start_index']}-{best_match['end_index']} (score: {best_match['score']})")
            else:
                print(f"      ⚠️ Could not locate {field_label}")
        
        print(f"✅ Step 6: Detected {len(field_positions)} field headers")
        return field_positions
    
    def _find_direct_header_match(self, header_key: str, field_label: str, used_indices: Set[int]) -> Optional[Dict]:
        """Find direct header matches"""
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
                word_norm = word_info['normalized']
                
                if header_word == word_norm:
                    match_score += 100
                    matched_indices.append(word_idx)
                elif header_word in word_norm or word_norm in header_word:
                    match_score += 70
                    matched_indices.append(word_idx)
            
            required_score = len(header_words) * 60
            if match_score >= required_score and len(matched_indices) >= len(header_words) * 0.7:
                return {
                    'start_index': start_idx,
                    'end_index': start_idx + len(header_words) - 1,
                    'score': match_score,
                    'matched_indices': matched_indices,
                    'field_label': field_label,
                    'match_type': 'direct_header'
                }
        return None
    
    def _find_best_trigger_match(self, triggers: List[str], field_label: str, used_indices: Set[int]) -> Optional[Dict]:
        """Find best trigger word match"""
        best_match = None
        best_score = 0
        
        for start_idx in range(len(self.word_mapping)):
            if start_idx in used_indices:
                continue
            
            match_result = self._evaluate_trigger_sequence(triggers, start_idx, field_label)
            if match_result and match_result['score'] > best_score:
                overlap = any(i in used_indices for i in range(match_result['start_index'], match_result['end_index'] + 1))
                if not overlap:
                    best_score = match_result['score']
                    best_match = match_result
        
        return best_match
    
    def _evaluate_trigger_sequence(self, triggers: List[str], start_idx: int, field_label: str) -> Optional[Dict]:
        """Evaluate trigger word sequence match"""
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
            word_norm = word_info['normalized']
            
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
            elif (word_info['is_separator'] or
                  word_info['length'] <= 2 or
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
        
        if final_score >= required_score and completeness >= 0.5:
            return {
                'start_index': start_idx,
                'end_index': current_idx - 1,
                'score': final_score,
                'matched_indices': matched_indices,
                'field_label': field_label,
                'completeness': completeness,
                'trigger_matches': trigger_idx
            }
        return None
    
    def create_field_boundaries(self) -> Dict[str, Tuple[int, int]]:
        """Create field boundaries for value extraction"""
        print("🔄 Step 7: Creating enhanced field boundaries...")
        boundaries = {}
        total_words = len(self.word_mapping)
        
        sorted_fields = sorted(self.field_positions.items(), key=lambda x: x[1]['start_index'])
        
        for i, (field_label, position_info) in enumerate(sorted_fields):
            header_end = position_info['end_index']
            value_start = self._find_value_start_after_header(header_end + 1, total_words)
            
            if i + 1 < len(sorted_fields):
                next_field_start = sorted_fields[i + 1][1]['start_index']
                value_end = self._find_value_end_before_next_field(value_start, next_field_start - 1)
            else:
                value_end = self._find_logical_field_end(value_start, total_words - 1, field_label)
            
            value_start = max(0, min(value_start, total_words - 1))
            value_end = max(value_start, min(value_end, total_words - 1))
            
            boundaries[field_label] = (value_start, value_end)
            print(f"   📍 {field_label}: header[{position_info['start_index']}:{position_info['end_index']}] -> value[{value_start}:{value_end}]")
        
        print(f"✅ Step 7: Created boundaries for {len(boundaries)} fields")
        return boundaries
    
    def _find_value_start_after_header(self, start_idx: int, total_words: int) -> int:
        """Find value start position after header"""
        current_idx = start_idx
        while current_idx < total_words and current_idx < len(self.word_mapping):
            word_info = self.word_mapping[current_idx]
            if (word_info['is_separator'] or
                word_info['length'] <= 1 or
                word_info['normalized'] in ['the', 'of', 'and', 'a', 'an', 'is', 'are', 'was', 'were']):
                current_idx += 1
                continue
            break
        return min(current_idx, total_words - 1)
    
    def _find_value_end_before_next_field(self, start_idx: int, max_end_idx: int) -> int:
        """Find value end position before next field"""
        current_idx = max_end_idx
        while current_idx >= start_idx:
            word_info = self.word_mapping[current_idx]
            if (word_info['length'] > 1 and
                not word_info['is_separator'] and
                word_info['normalized'] not in ['the', 'of', 'and', 'a', 'an', 'is', 'are', 'was', 'were']):
                return current_idx
            current_idx -= 1
        return max(start_idx, max_end_idx)
    
    def _find_logical_field_end(self, start_idx: int, max_end_idx: int, field_label: str) -> int:
        """Find logical field end based on field type"""
        field_type = self.template_config.get_field_type(field_label)
        
        if field_type in ["date", "code", "reference"]:
            limited_end = min(start_idx + 8, max_end_idx)
            return self._find_value_end_before_next_field(start_idx, limited_end)
        elif field_type in ["name"]:
            limited_end = min(start_idx + 15, max_end_idx)
            return self._find_value_end_before_next_field(start_idx, limited_end)
        elif field_type in ["amount"]:
            limited_end = min(start_idx + 5, max_end_idx)
            return self._find_value_end_before_next_field(start_idx, limited_end)
        elif field_type in ["phone"]:
            limited_end = min(start_idx + 3, max_end_idx)
            return self._find_value_end_before_next_field(start_idx, limited_end)
        else:
            return self._find_value_end_before_next_field(start_idx, max_end_idx)
    
    def extract_field_values(self, full_text: str = "") -> Dict[str, Dict]:
        """Extract field values using enhanced logic"""
        print("🔄 Step 8: Extracting field values with enhanced logic...")
        results = {}
        field_boundaries = self.create_field_boundaries()
        
        for field_label, position_info in self.field_positions.items():
            print(f"   🔍 Extracting {field_label}...")
            header_start = position_info['start_index']
            header_end = position_info['end_index']
            
            if field_label in field_boundaries:
                value_start_idx, value_end_idx = field_boundaries[field_label]
            else:
                value_start_idx = header_end + 1
                value_end_idx = len(self.word_mapping) - 1
            
            actual_value_start = self._find_value_start(value_start_idx, value_end_idx)
            extracted_value, value_indices = self._extract_complete_field_value_enhanced(
                actual_value_start, value_end_idx, field_label, full_text
            )
            
            results[field_label] = {
                'value': extracted_value.strip(),
                'value_indices': value_indices,
                'header_indices': position_info['matched_indices'],
                'header_range': (header_start, header_end),
                'value_range': (actual_value_start, value_end_idx)
            }
            print(f"   ✅ {field_label}: '{extracted_value.strip()}'")
        
        print(f"✅ Step 8: Extracted {len(results)} field values")
        return results
    
    def _find_value_start(self, start_idx: int, end_idx: int) -> int:
        """Find actual value start position"""
        current_idx = start_idx
        while current_idx <= end_idx and current_idx < len(self.word_mapping):
            word_info = self.word_mapping[current_idx]
            if (word_info['is_separator'] or
                word_info['length'] <= 1 or
                word_info['normalized'] in ['the', 'of', 'and', 'a', 'an']):
                current_idx += 1
                continue
            break
        return min(current_idx, end_idx)
    
    def _extract_complete_field_value_enhanced(self, start_idx: int, end_idx: int, 
                                             field_label: str, full_text: str = "") -> Tuple[str, List[int]]:
        """Extract complete field value based on field type"""
        field_type = self.template_config.get_field_type(field_label)
        
        # Route to appropriate extraction method based on field type
        if field_type == "email":
            return self._extract_email_value(start_idx, end_idx)
        elif field_type == "name":
            return self._extract_name_value(start_idx, end_idx, field_label)
        elif field_type == "date":
            return self._extract_date_value(start_idx, end_idx)
        elif field_type == "code":
            return self._extract_code_value(start_idx, end_idx, field_label, full_text)
        elif field_type == "reference":
            return self._extract_reference_value(start_idx, end_idx, field_label, full_text)
        elif field_type == "address":
            return self._extract_address_value(start_idx, end_idx)
        elif field_type == "place":
            return self._extract_place_value(start_idx, end_idx)
        elif field_type == "amount":
            return self._extract_amount_value(start_idx, end_idx)
        elif field_type == "phone":
            return self._extract_phone_value(start_idx, end_idx)
        elif field_type == "type":
            return self._extract_type_value(start_idx, end_idx, field_label)
        else:
            return self._extract_generic_value(start_idx, end_idx, field_label)
    
    def _extract_email_value(self, start_idx: int, end_idx: int) -> Tuple[str, List[int]]:
        """Extract email value"""
        for i in range(start_idx, min(start_idx + 5, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            if word_info['is_email_like']:
                return word_info['clean'], [i]
        return "", []
    
    def _extract_name_value(self, start_idx: int, end_idx: int, field_label: str) -> Tuple[str, List[int]]:
        """Extract name value with context awareness"""
        name_parts = []
        name_indices = []
        
        # Dynamic limits based on field type
        if field_label == "buyer_name":
            max_words = 3
        else:
            max_words = 20
        
        i = start_idx
        consecutive_non_name = 0
        max_consecutive_non_name = 2 if field_label == "buyer_name" else 3
        words_collected = 0
        
        # Template-specific adjustments
        if self.current_template == "amazon_invoice" and field_label == "seller_name":
            max_consecutive_non_name = 1
        
        while (i <= end_idx and i < len(self.word_mapping) and words_collected < max_words):
            word_info = self.word_mapping[i]
            word_norm = word_info['normalized']
            include_word = False
            
            # Template-specific stop conditions
            if (self.current_template == "amazon_invoice" and field_label == "seller_name" and
                (word_norm in ['pan', 'gst', 'gstin'] or 
                 (word_info['is_code_like'] and len(word_info['clean']) >= 10))):
                break
            
            if (field_label == "buyer_name" and 
                (word_info['is_amount_like'] or word_info['is_currency'] or
                 word_norm in ['hsn', 'qty', 'price', 'discount', 'taxable', 'sgst', 'cgst', 'igst', 'total', 'shipping', 'charges', '₹'])):
                break
            
            # Name detection logic
            if any(entity['type'] in ['PER', 'ORG'] and entity['confidence'] > 0.4
                   for entity in word_info['entity_types']):
                include_word = True
                consecutive_non_name = 0
            elif word_info['is_capitalized'] and word_info['has_alpha'] and word_info['length'] > 1:
                include_word = True
                consecutive_non_name = 0
            elif (word_info['is_all_caps'] and word_info['length'] > 2 and
                  word_norm not in ['FOR', 'AND', 'THE', 'OF', 'IN', 'ON', 'AT', 'TO', 'PAN', 'GST', 'GSTIN', 'HSN']):
                include_word = True
                consecutive_non_name = 0
            elif word_info['is_organization_word']:
                include_word = True
                consecutive_non_name = 0
            elif (word_norm in ['and', 'of', 'the', '&', ','] and
                  name_parts and consecutive_non_name == 0):
                include_word = True
            elif (word_info['has_numeric'] and not word_info['has_alpha'] and
                  word_info['length'] <= 6 and field_label in ['seller_name', 'seller_address', 'buyer_address'] and
                  not word_info['is_code_like']):
                include_word = True
                consecutive_non_name = 0
            elif (word_info['is_amount_like'] or word_info['is_currency'] or
                  word_info['is_date_like'] or word_info['is_email_like'] or
                  word_norm in ['rs', 'inr', 'total', 'amount', 'date', 'invoice', 'order']):
                break
            else:
                consecutive_non_name += 1
                if consecutive_non_name > max_consecutive_non_name:
                    break
            
            if include_word:
                name_parts.append(word_info['original'])
                name_indices.append(i)
                words_collected += 1
            
            i += 1
        
        return " ".join(name_parts), name_indices
    
    def _extract_date_value(self, start_idx: int, end_idx: int) -> Tuple[str, List[int]]:
        """Extract date value"""
        # Look for complete date patterns first
        for i in range(start_idx, min(start_idx + 3, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            if word_info['is_date_like']:
                return word_info['clean'], [i]
        
        # Look for date components
        date_parts = []
        date_indices = []
        for i in range(start_idx, min(start_idx + 6, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            if word_info['is_month']:
                date_parts.append(word_info['original'])
                date_indices.append(i)
            elif word_info['has_numeric'] and not word_info['has_alpha']:
                try:
                    num_value = int(''.join(c for c in word_info['clean'] if c.isdigit()))
                    if (1 <= num_value <= 31) or (1900 <= num_value <= 2100):
                        date_parts.append(word_info['original'])
                        date_indices.append(i)
                except:
                    pass
        
        return " ".join(date_parts), date_indices
    
    def _extract_code_value(self, start_idx: int, end_idx: int, field_label: str, full_text: str = "") -> Tuple[str, List[int]]:
        """Extract code value with field-specific validation using enhanced regex patterns"""
        
        # For PAN and GST, first try global text search using regex
        if field_label == "pan_number":
            pan_value = self.template_config.extract_pan_from_text(full_text or self.full_text)
            if pan_value:
                # Find the word index containing this PAN
                for i, word_info in enumerate(self.word_mapping):
                    if pan_value in word_info['clean'].upper() or word_info['clean'].upper() in pan_value:
                        return pan_value, [i]
                return pan_value, []
        
        elif field_label in ["gst_number", "seller_gst_number", "buyer_gst_number", "gstin"]:
            gst_value = self.template_config.extract_gst_from_text(full_text or self.full_text)
            if gst_value:
                # Find the word index containing this GST
                for i, word_info in enumerate(self.word_mapping):
                    if gst_value in word_info['clean'].upper() or word_info['clean'].upper() in gst_value:
                        return gst_value, [i]
                return gst_value, []
        
        # Search in local range
        for i in range(start_idx, min(start_idx + 15, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            
            if field_label == "pan_number":
                if word_info['is_pan_like']:
                    return word_info['clean'].upper(), [i]
                # Also check for PAN patterns in combined words
                if i + 1 < len(self.word_mapping):
                    combined = word_info['clean'] + self.word_mapping[i + 1]['clean']
                    if self._is_pan_pattern(combined):
                        return combined.upper(), [i, i + 1]
            elif field_label in ["gst_number", "seller_gst_number", "buyer_gst_number", "gstin"]:
                if word_info['is_gst_like']:
                    return word_info['clean'].upper(), [i]
                # Also check for GST patterns in combined words
                if i + 1 < len(self.word_mapping):
                    combined = word_info['clean'] + self.word_mapping[i + 1]['clean']
                    if self._is_gst_pattern(combined):
                        return combined.upper(), [i, i + 1]
            elif field_label in ["state_ut_code", "state_code"]:
                if (word_info['has_numeric'] and not word_info['has_alpha'] and
                    1 <= len(word_info['clean']) <= 3):
                    return word_info['clean'], [i]
            elif word_info['is_code_like']:
                return word_info['clean'], [i]
        
        # Extended global search for PAN/GST patterns across the entire document
        if field_label in ["pan_number", "gst_number", "gstin", "seller_gst_number", "buyer_gst_number"]:
            for i in range(len(self.word_mapping)):
                word_info = self.word_mapping[i]
                if field_label == "pan_number":
                    if word_info['is_pan_like']:
                        return word_info['clean'].upper(), [i]
                elif field_label in ["gst_number", "gstin", "seller_gst_number", "buyer_gst_number"]:
                    if word_info['is_gst_like']:
                        return word_info['clean'].upper(), [i]
        
        return "", []
    
    def _extract_reference_value(self, start_idx: int, end_idx: int, field_label: str, full_text: str = "") -> Tuple[str, List[int]]:
        """Extract reference value"""
        # Local search
        for i in range(start_idx, min(start_idx + 5, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            if (word_info['has_alpha'] and word_info['has_numeric'] and
                word_info['length'] >= 6 and not word_info['is_email_like']):
                return word_info['clean'], [i]
            elif (word_info['has_numeric'] and not word_info['has_alpha'] and
                  word_info['length'] >= 8):
                return word_info['clean'], [i]
            elif ('-' in word_info['original'] and
                  any(c.isalnum() for c in word_info['original']) and
                  word_info['length'] >= 6):
                return word_info['clean'], [i]
        
        # Global search for specific reference types
        if field_label in ["invoice_number", "order_number"]:
            for i in range(len(self.word_mapping)):
                word_info = self.word_mapping[i]
                if field_label == "invoice_number":
                    if (('-' in word_info['original'] or word_info['has_alpha']) and 
                        word_info['has_numeric'] and word_info['length'] >= 6):
                        return word_info['clean'], [i]
                elif field_label == "order_number":
                    if ((word_info['has_numeric'] and word_info['length'] >= 7) or
                        ('-' in word_info['original'] and word_info['has_numeric'])):
                        return word_info['clean'], [i]
        
        return "", []
    
    def _extract_address_value(self, start_idx: int, end_idx: int) -> Tuple[str, List[int]]:
        """Extract address value"""
        address_parts = []
        address_indices = []
        
        for i in range(start_idx, min(end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            word_norm = word_info['normalized']
            
            # Stop conditions
            if (word_info['is_amount_like'] or word_info['is_currency'] or
                word_norm in ['total', 'amount', 'price', 'cost', 'invoice', 'order', 'hsn', 'qty']):
                break
            
            # Include conditions
            if ((word_info['has_alpha'] and word_info['length'] > 1) or
                (word_info['has_numeric'] and word_info['length'] <= 6)):
                address_parts.append(word_info['original'])
                address_indices.append(i)
            elif word_norm in ['and', 'of', 'the', '&', 'at', 'near', 'in', 'on', ','] and address_parts:
                address_parts.append(word_info['original'])
                address_indices.append(i)
        
        return " ".join(address_parts), address_indices
    
    def _extract_place_value(self, start_idx: int, end_idx: int) -> Tuple[str, List[int]]:
        """Extract place value"""
        place_parts = []
        place_indices = []
        
        for i in range(start_idx, min(start_idx + 8, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            if ((word_info['is_capitalized'] or word_info['is_all_caps']) and
                word_info['has_alpha'] and word_info['length'] > 2):
                place_parts.append(word_info['original'])
                place_indices.append(i)
        
        return " ".join(place_parts), place_indices
    
    def _extract_amount_value(self, start_idx: int, end_idx: int) -> Tuple[str, List[int]]:
        """Extract amount value"""
        for i in range(start_idx, min(start_idx + 5, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            if word_info['is_amount_like']:
                return word_info['clean'], [i]
            elif word_info['is_currency'] and word_info['has_numeric']:
                return word_info['clean'], [i]
            elif (word_info['has_numeric'] and not word_info['has_alpha'] and
                  len(word_info['clean']) >= 3):
                if i + 1 < len(self.word_mapping):
                    next_word = self.word_mapping[i + 1]
                    if ('.' in next_word['original'] or next_word['is_currency']):
                                            return combined, [i, i+1]
                # If no next word or next word doesn't form an amount, return the current word
                return word_info['clean'], [i]
        
        # If no amount-like word is found, try to look for numbers with decimal points or commas
        for i in range(start_idx, min(start_idx + 5, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            # Check if the word is a number with decimal or comma
            if word_info['has_numeric'] and (',' in word_info['original'] or '.' in word_info['original']):
                return word_info['clean'], [i]
        
        return "", []
    
    def _extract_phone_value(self, start_idx: int, end_idx: int) -> Tuple[str, List[int]]:
        """Extract phone value"""
        for i in range(start_idx, min(start_idx + 5, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            if word_info['is_phone_like']:
                # Return the original word if it's phone-like
                return word_info['original'], [i]
            # Also check for phone numbers split by spaces or hyphens
            if word_info['has_numeric'] and (word_info['length'] >= 10 and word_info['length'] <= 15):
                # Check if it's a sequence of digits with possible separators
                clean_digits = ''.join(c for c in word_info['clean'] if c.isdigit())
                if 10 <= len(clean_digits) <= 15:
                    return word_info['original'], [i]
        
        # Check for phone numbers that might be split across two words
        if start_idx + 1 < len(self.word_mapping):
            word1 = self.word_mapping[start_idx]
            word2 = self.word_mapping[start_idx+1]
            combined = word1['original'] + word2['original']
            clean_combined = ''.join(c for c in combined if c.isdigit())
            if 10 <= len(clean_combined) <= 15:
                return combined, [start_idx, start_idx+1]
        
        return "", []
    
    def _extract_type_value(self, start_idx: int, end_idx: int, field_label: str) -> Tuple[str, List[int]]:
        """Extract type value (e.g., transaction type, document type)"""
        type_parts = []
        type_indices = []
        
        for i in range(start_idx, min(start_idx + 5, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            word_norm = word_info['normalized']
            
            # Stop conditions
            if (word_info['is_amount_like'] or word_info['is_currency'] or
                word_info['is_date_like'] or word_info['is_email_like'] or
                word_info['is_code_like'] or word_info['is_phone_like']):
                break
            
            # Include conditions
            if (word_info['has_alpha'] and word_info['length'] > 1 and
                word_norm not in ['the', 'of', 'and', 'a', 'an', 'is', 'are', 'was', 'were']):
                type_parts.append(word_info['original'])
                type_indices.append(i)
            elif word_norm in ['-', '/', '|'] and type_parts:
                type_parts.append(word_info['original'])
                type_indices.append(i)
        
        return " ".join(type_parts), type_indices
    
    def _extract_generic_value(self, start_idx: int, end_idx: int, field_label: str) -> Tuple[str, List[int]]:
        """Extract generic value (fallback for other field types)"""
        value_parts = []
        value_indices = []
        
        for i in range(start_idx, min(end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            word_norm = word_info['normalized']
            
            # Stop conditions
            if (word_info['is_amount_like'] or word_info['is_currency'] or
                word_info['is_date_like'] or word_info['is_email_like'] or
                word_info['is_code_like'] or word_info['is_phone_like']):
                break
            
            # Include conditions
            if (word_info['has_alpha'] or word_info['has_numeric']) and word_info['length'] > 0:
                value_parts.append(word_info['original'])
                value_indices.append(i)
            elif word_norm in [':', '-', '/', '|', ','] and value_parts:
                value_parts.append(word_info['original'])
                value_indices.append(i)
        
        return " ".join(value_parts), value_indices
    
    def format_output(self, extracted_data: Dict[str, Dict]) -> Dict:
        """Format extracted data according to template configuration"""
        template = self.template_config.get_template(self.current_template)
        output_structure = template.get('output_structure', 'flat')
        formatter_name = template.get('output_formatter', None)
        
        if output_structure == 'nested' and formatter_name:
            if formatter_name == 'amazon_formatter':
                return self._amazon_formatter(extracted_data)
            elif formatter_name == 'omgs_formatter':
                return self._omgs_formatter(extracted_data)
        
        # Default flat structure
        return {field: data['value'] for field, data in extracted_data.items()}
    
    def _amazon_formatter(self, extracted_data: Dict[str, Dict]) -> Dict:
        """Format Amazon invoice data into nested structure"""
        result = {
            "document_type": extracted_data.get("document_type", {}).get("value", ""),
            "seller": {
                "name": extracted_data.get("sold_by", {}).get("value", ""),
                "pan": extracted_data.get("pan_number", {}).get("value", ""),
                "gst": extracted_data.get("gst_number", {}).get("value", ""),
                "address": extracted_data.get("billing_address", {}).get("value", "")
            },
            "buyer": {
                "name": extracted_data.get("buyer_name", {}).get("value", ""),
                "address": extracted_data.get("shipping_address", {}).get("value", "")
            },
            "order": {
                "number": extracted_data.get("order_number", {}).get("value", ""),
                "date": extracted_data.get("order_date", {}).get("value", "")
            },
            "invoice": {
                "number": extracted_data.get("invoice_number", {}).get("value", ""),
                "date": extracted_data.get("invoice_date", {}).get("value", ""),
                "details": extracted_data.get("invoice_details", {}).get("value", "")
            },
            "place_of_supply": extracted_data.get("place_of_supply", {}).get("value", ""),
            "place_of_delivery": extracted_data.get("place_of_delivery", {}).get("value", ""),
            "state_ut_code": extracted_data.get("state_ut_code", {}).get("value", "")
        }
        return result
    
    def _omgs_formatter(self, extracted_data: Dict[str, Dict]) -> Dict:
        """Format OMGS invoice data into nested structure"""
        result = {
            "invoice": {
                "number": extracted_data.get("invoice_number", {}).get("value", ""),
                "date": extracted_data.get("invoice_date", {}).get("value", "")
            },
            "order": {
                "number": extracted_data.get("order_number", {}).get("value", ""),
                "date": extracted_data.get("order_date", {}).get("value", "")
            },
            "payment_method": extracted_data.get("payment_method", {}).get("value", ""),
            "bill_to": extracted_data.get("bill_to", {}).get("value", ""),
            "ship_to": extracted_data.get("ship_to", {}).get("value", ""),
            "sold_by": extracted_data.get("sold_by", {}).get("value", ""),
            "gstin": extracted_data.get("gstin", {}).get("value", ""),
            "state_code": extracted_data.get("state_code", {}).get("value", "")
        }
        return result
    
    def extract_fields(self, file_path: str = None, file_content: bytes = None, 
                      template_name: str = None) -> Dict:
        """Main extraction function"""
        print("🚀 Starting PDF field extraction...")
        
        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf(file_path, file_content)
        if not text:
            return {"error": "Failed to extract text from PDF"}
        
        # Step 2: Detect template if not provided
        if template_name:
            self.template_config.set_current_template(template_name)
        else:
            detected_template = self.template_config.detect_template(text)
            self.template_config.set_current_template(detected_template)
            print(f"📋 Detected template: {detected_template}")
        
        self.current_template = self.template_config.current_template
        
        # Step 3: Create word mapping
        self.word_mapping = self.create_word_mapping(text)
        
        # Step 4: Initialize NER extractor
        self.initialize_ner_extractor()
        
        # Step 5: Extract NER entities
        ner_entities = self.extract_ner_entities(text)
        
        # Step 6: Map NER entities to word indices
        self.map_ner_entities(ner_entities)
        
        # Step 7: Find field headers
        self.field_positions = self.find_field_headers()
        
        # Step 8: Extract field values
        extracted_data = self.extract_field_values(text)
        
        # Step 9: Format output
        formatted_data = self.format_output(extracted_data)
        
        print("✅ Extraction completed successfully!")
        return formatted_data


def main():
    st.markdown('<h1 class="main-header">PDF Field Extraction System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'extractor' not in st.session_state:
        st.session_state.extractor = GeneralizedPDFExtractor()
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
        file_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB",
            "File Type": uploaded_file.type
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.metric("Type", uploaded_file.type)
        
        # Extraction process
        if extract_button:
            with st.spinner("Extracting fields from document..."):
                start_time = time.time()
                
                # Read file content
                file_content = uploaded_file.read()
                
                # Determine template
                template_to_use = None if selected_template == "Auto-detect" else selected_template
                
                try:
                    # Extract fields
                    result = st.session_state.extractor.extract_fields(
                        file_content=file_content,
                        template_name=template_to_use
                    )
                    
                    extraction_time = time.time() - start_time
                    
                    if "error" in result:
                        st.markdown(f'<div class="error-box">Error: {result["error"]}</div>', unsafe_allow_html=True)
                    else:
                        st.session_state.extracted_data = result
                        st.session_state.current_template = st.session_state.extractor.current_template
                        
                        st.markdown(f'<div class="success-box">Extraction completed in {extraction_time:.2f} seconds!</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="info-box">Detected template: {st.session_state.current_template}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-box">Error during extraction: {str(e)}</div>', unsafe_allow_html=True)
    
    # Display extracted data
    if st.session_state.extracted_data is not None:
        st.markdown('<h2 class="sub-header">Extracted Fields</h2>', unsafe_allow_html=True)
        
        # Display as JSON
        st.json(st.session_state.extracted_data)
        
        # Download button
        json_str = json.dumps(st.session_state.extracted_data, indent=4)
        st.download_button(
            label="Download Extracted Data (JSON)",
            data=json_str,
            file_name=f"extracted_data_{st.session_state.current_template}.json",
            mime="application/json"
        )
        
        # Display as table if flat structure
        if isinstance(st.session_state.extracted_data, dict):
            # Check if it's a flat structure (no nested dicts)
            is_flat = all(not isinstance(v, dict) for v in st.session_state.extracted_data.values())
            
            if is_flat:
                st.markdown('<h3 class="sub-header">Data Table</h3>', unsafe_allow_html=True)
                
                # Convert to DataFrame
                df = pd.DataFrame([
                    {"Field": k, "Value": v} 
                    for k, v in st.session_state.extracted_data.items()
                ])
                
                st.dataframe(df, use_container_width=True)
                
                # Download as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"extracted_data_{st.session_state.current_template}.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()