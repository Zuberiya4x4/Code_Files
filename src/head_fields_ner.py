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
        
        # Fixed regex patterns for PAN and GST
        self.pan_pattern = re.compile(r'[A-Z]{5}[0-9]{4}[A-Z]{1}')
        self.gst_pattern = re.compile(r'[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z]{1}[0-9A-Z]{1}')
        
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
                "document_type", "sold_by", "billing_address", "shipping_address",
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
                "GST Registration No": "gst_number",
                "GSTIN": "gst_number",
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
                "sold_by": ["sold", "by"],
                "pan_number": ["pan", "no"],
                "gst_number": ["gst", "registration", "no"],
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

class GeneralizedPDFExtractor:
    """Generalized PDF field extraction system"""
    
    def __init__(self, config_file_path: str = None):
        self.template_config = TemplateConfig(config_file_path)
        self.ner_extractor = None
        self.current_template = None
        self.word_mapping = []
        self.field_positions = {}
        
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
        return bool(self.template_config.pan_pattern.fullmatch(word.upper()))
    
    def _is_gst_pattern(self, word: str) -> bool:
        """Check if word matches GST pattern using fixed regex"""
        return bool(self.template_config.gst_pattern.fullmatch(word.upper()))
    
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
        """Extract code value with field-specific validation using fixed regex patterns"""
        # Search in local range first
        for i in range(start_idx, min(start_idx + 15, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            
            if field_label == "pan_number":
                if word_info['is_pan_like']:  # Use the fixed regex pattern
                    return word_info['clean'], [i]
            elif field_label in ["gst_number", "seller_gst_number", "buyer_gst_number", "gstin"]:
                if word_info['is_gst_like']:  # Use the fixed regex pattern
                    return word_info['clean'], [i]
            elif field_label in ["state_ut_code", "state_code"]:
                if (word_info['has_numeric'] and not word_info['has_alpha'] and
                    1 <= len(word_info['clean']) <= 3):
                    return word_info['clean'], [i]
            elif word_info['is_code_like']:
                return word_info['clean'], [i]
        
        # Global search for specific patterns using regex
        if field_label in ["pan_number", "gst_number", "gstin"]:
            for i in range(len(self.word_mapping)):
                word_info = self.word_mapping[i]
                if field_label == "pan_number":
                    if word_info['is_pan_like']:  # Use the fixed regex pattern
                        return word_info['clean'], [i]
                elif field_label in ["gst_number", "gstin"]:
                    if word_info['is_gst_like']:  # Use the fixed regex pattern
                        return word_info['clean'], [i]
        
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
                        combined_amount = word_info['original'] + " " + next_word['original']
                        return combined_amount, [i, i + 1]
                return word_info['clean'], [i]
        return "", []
    
    def _extract_phone_value(self, start_idx: int, end_idx: int) -> Tuple[str, List[int]]:
        """Extract phone value"""
        for i in range(start_idx, min(start_idx + 3, end_idx + 1, len(self.word_mapping))):
            word_info = self.word_mapping[i]
            if word_info['is_phone_like']:
                return word_info['clean'], [i]
        return "", []
    
    def _extract_type_value(self, start_idx: int, end_idx: int, field_label: str) -> Tuple[str, List[int]]:
        """Extract type value"""
        if field_label == "document_type":
            type_parts = []
            type_indices = []
            for i in range(start_idx, min(start_idx + 8, end_idx + 1, len(self.word_mapping))):
                word_info = self.word_mapping[i]
                word_norm = word_info['normalized']
                if word_norm in ['tax', 'invoice', 'bill', 'supply', 'cash', 'memo']:
                    type_parts.append(word_info['original'])
                    type_indices.append(i)
            if type_parts:
                return " ".join(type_parts), type_indices
        elif field_label == "payment_method":
            for i in range(start_idx, min(start_idx + 3, end_idx + 1, len(self.word_mapping))):
                word_info = self.word_mapping[i]
                word_norm = word_info['normalized']
                if word_norm in ['razorpay', 'paytm', 'cash', 'card', 'upi', 'netbanking']:
                    return word_info['original'], [i]
        
        return self._extract_generic_value(start_idx, end_idx, field_label)
    
    def _extract_generic_value(self, start_idx: int, end_idx: int, field_label: str) -> Tuple[str, List[int]]:
        """Extract generic value"""
        value_parts = []
        value_indices = []
        
        if "details" in field_label:
            max_words = 15
        else:
            max_words = 12
        
        words_collected = 0
        for i in range(start_idx, min(end_idx + 1, len(self.word_mapping))):
            if words_collected >= max_words:
                break
            
            word_info = self.word_mapping[i]
            
            # Field-specific stop conditions
            if (field_label == "payment_method" and 
                word_info['normalized'] in ['product', 'hsn', 'qty', 'price', 'discount']):
                break
            
            if ((word_info['has_alpha'] and word_info['length'] > 1) or
                (word_info['has_numeric'] and word_info['length'] > 1)):
                value_parts.append(word_info['original'])
                value_indices.append(i)
                words_collected += 1
        
        return " ".join(value_parts), value_indices
    
    def validate_extractions(self, extracted_data: Dict[str, Dict], full_text: str = "") -> Dict[str, Dict]:
        """Validate extracted data and find missed words"""
        print("🔄 Step 9: Validating extractions and detecting missed words...")
        validation_results = {}
        all_used_indices = set()
        
        # Collect all used indices
        for field_data in extracted_data.values():
                        all_used_indices.update(field_data.get('value_indices', []))
        
        # Find missed words (words that are not used in any extraction)
        missed_words = []
        for i, word_info in enumerate(self.word_mapping):
            if i not in all_used_indices and word_info['length'] > 1:
                # Skip common words that are not important
                if word_info['normalized'] not in ['the', 'of', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by']:
                    missed_words.append((i, word_info['original']))
        
        # Validate each field
        for field_label, field_data in extracted_data.items():
            value = field_data['value']
            value_indices = field_data['value_indices']
            field_type = self.template_config.get_field_type(field_label)
            
            # Basic validation
            is_valid = True
            validation_issues = []
            
            if not value:
                is_valid = False
                validation_issues.append("Empty value")
            elif field_type == "email" and '@' not in value:
                is_valid = False
                validation_issues.append("Invalid email format")
            elif field_type == "date" and not self._is_date_pattern(value):
                is_valid = False
                validation_issues.append("Invalid date format")
            elif field_type == "amount" and not self._is_amount_pattern(value):
                is_valid = False
                validation_issues.append("Invalid amount format")
            elif field_type == "phone" and not self._is_phone_pattern(value):
                is_valid = False
                validation_issues.append("Invalid phone format")
            elif field_type == "pan_number" and not self._is_pan_pattern(value):
                is_valid = False
                validation_issues.append("Invalid PAN format")
            elif field_type in ["gst_number", "gstin"] and not self._is_gst_pattern(value):
                is_valid = False
                validation_issues.append("Invalid GST format")
            
            validation_results[field_label] = {
                'value': value,
                'is_valid': is_valid,
                'issues': validation_issues,
                'field_type': field_type,
                'value_indices': value_indices,
                'header_indices': field_data['header_indices']
            }
        
        print(f"✅ Step 9: Validation complete. {len(missed_words)} missed words detected.")
        return validation_results
    
    def format_output(self, validation_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Format the output according to the template's output structure"""
        template = self.template_config.get_template(self.current_template)
        output_structure = template.get('output_structure', 'flat')
        output_formatter = template.get('output_formatter', None)
        
        if output_structure == 'nested' and output_formatter:
            if output_formatter == 'amazon_formatter':
                return self._format_amazon_output(validation_results)
            elif output_formatter == 'omgs_formatter':
                return self._format_omgs_output(validation_results)
        
        # Default flat structure
        formatted_output = {}
        for field_label, result in validation_results.items():
            formatted_output[field_label] = result['value']
        
        return formatted_output
    
    def _format_amazon_output(self, validation_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Format output for Amazon invoice template"""
        output = {
            "document_type": validation_results.get('document_type', {}).get('value', ''),
            "seller_details": {
                "name": validation_results.get('sold_by', {}).get('value', ''),
                "pan_number": validation_results.get('pan_number', {}).get('value', ''),
                "gst_number": validation_results.get('gst_number', {}).get('value', '')
            },
            "billing_address": validation_results.get('billing_address', {}).get('value', ''),
            "shipping_address": validation_results.get('shipping_address', {}).get('value', ''),
            "place_of_supply": validation_results.get('place_of_supply', {}).get('value', ''),
            "place_of_delivery": validation_results.get('place_of_delivery', {}).get('value', ''),
            "order_details": {
                "order_number": validation_results.get('order_number', {}).get('value', ''),
                "order_date": validation_results.get('order_date', {}).get('value', '')
            },
            "invoice_details": {
                "invoice_number": validation_results.get('invoice_number', {}).get('value', ''),
                "invoice_date": validation_results.get('invoice_date', {}).get('value', ''),
                "details": validation_results.get('invoice_details', {}).get('value', '')
            }
        }
        return output
    
    def _format_omgs_output(self, validation_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Format output for OMGS invoice template"""
        output = {
            "invoice_details": {
                "invoice_number": validation_results.get('invoice_number', {}).get('value', ''),
                "invoice_date": validation_results.get('invoice_date', {}).get('value', '')
            },
            "order_details": {
                "order_number": validation_results.get('order_number', {}).get('value', ''),
                "order_date": validation_results.get('order_date', {}).get('value', '')
            },
            "payment_method": validation_results.get('payment_method', {}).get('value', ''),
            "addresses": {
                "bill_to": validation_results.get('bill_to', {}).get('value', ''),
                "ship_to": validation_results.get('ship_to', {}).get('value', ''),
                "sold_by": validation_results.get('sold_by', {}).get('value', '')
            },
            "tax_details": {
                "gstin": validation_results.get('gstin', {}).get('value', ''),
                "state_code": validation_results.get('state_code', {}).get('value', '')
            }
        }
        return output
    
    def extract_fields(self, file_path: str = None, file_content: bytes = None, template_name: str = None) -> Dict[str, Any]:
        """Main method to extract fields from PDF"""
        start_time = time.time()
        print("🚀 Starting PDF field extraction...")
        
        # Step 1: Set template
        if template_name:
            self.template_config.set_current_template(template_name)
            self.current_template = template_name
        else:
            # Extract text first to auto-detect template
            text = self.extract_text_from_pdf(file_path, file_content)
            if not text:
                return {"error": "Could not extract text from PDF"}
            
            detected_template = self.template_config.detect_template(text)
            self.template_config.set_current_template(detected_template)
            self.current_template = detected_template
            print(f"🔍 Auto-detected template: {detected_template}")
        
        # Step 2: Extract text (if not already extracted)
        if 'text' not in locals():
            text = self.extract_text_from_pdf(file_path, file_content)
            if not text:
                return {"error": "Could not extract text from PDF"}
        
        # Step 3: Create word mapping
        self.word_mapping = self.create_word_mapping(text)
        
        # Step 4: Initialize NER model
        self.initialize_ner_extractor()
        
        # Step 5: Extract and map NER entities
        if self.ner_extractor:
            ner_entities = self.extract_ner_entities(text)
            self.map_ner_entities(ner_entities)
        
        # Step 6: Find field headers
        self.field_positions = self.find_field_headers()
        
        # Step 7: Extract field values
        extracted_data = self.extract_field_values(text)
        
        # Step 8: Validate extractions
        validation_results = self.validate_extractions(extracted_data, text)
        
        # Step 9: Format output
        formatted_output = self.format_output(validation_results)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"✅ Extraction completed in {processing_time:.2f} seconds")
        
        return {
            "template": self.current_template,
            "output": formatted_output,
            "validation": validation_results,
            "processing_time": processing_time,
            "word_count": len(self.word_mapping),
            "fields_detected": len(self.field_positions)
        }

def main():
    st.markdown('<h1 class="main-header">PDF Field Extraction System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'extractor' not in st.session_state:
        st.session_state.extractor = GeneralizedPDFExtractor()
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    if 'current_template' not in st.session_state:
        st.session_state.current_template = None
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sub-header">Settings</h2>', unsafe_allow_html=True)
    
    # Template selection
    template_names = list(st.session_state.extractor.template_config.templates.keys())
    selected_template = st.sidebar.selectbox(
        "Select Template",
        options=["Auto-detect"] + template_names,
        index=0
    )
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF File",
        type=["pdf"],
        help="Upload a PDF file for field extraction"
    )
    
    # Extract button
    extract_button = st.sidebar.button(
        "Extract Fields",
        type="primary",
        disabled=not uploaded_file
    )
    
    # Clear session button
    if st.sidebar.button("Clear Session", key="clear_session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.extractor = GeneralizedPDFExtractor()
        st.rerun()
    
    # Main content area
    if uploaded_file is not None:
        st.markdown('<h2 class="sub-header">Uploaded File</h2>', unsafe_allow_html=True)
        file_details = {
            "Filename": uploaded_file.name,
            "File type": uploaded_file.type,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)
    
    # Extraction process
    if extract_button and uploaded_file:
        with st.spinner("Extracting fields from PDF..."):
            try:
                # Get file content
                file_content = uploaded_file.read()
                
                # Determine template
                template_name = None if selected_template == "Auto-detect" else selected_template
                
                # Extract fields
                results = st.session_state.extractor.extract_fields(
                    file_content=file_content,
                    template_name=template_name
                )
                
                # Store results in session state
                st.session_state.extraction_results = results
                st.session_state.current_template = results.get('template', 'Unknown')
                
                # Show success message
                st.markdown(
                    f'<div class="success-box">Extraction completed successfully! '
                    f'Template: {results.get("template", "Unknown")} | '
                    f'Fields detected: {results.get("fields_detected", 0)} | '
                    f'Processing time: {results.get("processing_time", 0):.2f} seconds</div>',
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.markdown(
                    f'<div class="error-box">Error during extraction: {str(e)}</div>',
                    unsafe_allow_html=True
                )
    
    # Display results
    if st.session_state.extraction_results:
        results = st.session_state.extraction_results
        template = results.get('template', 'Unknown')
        output = results.get('output', {})
        validation = results.get('validation', {})
        
        st.markdown('<h2 class="sub-header">Extraction Results</h2>', unsafe_allow_html=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-container"><h3>Template</h3><p style="font-size: 1.5rem; font-weight: bold;">{}</p></div>'.format(template), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-container"><h3>Fields Detected</h3><p style="font-size: 1.5rem; font-weight: bold;">{}</p></div>'.format(results.get('fields_detected', 0)), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-container"><h3>Processing Time</h3><p style="font-size: 1.5rem; font-weight: bold;">{:.2f} sec</p></div>'.format(results.get('processing_time', 0)), unsafe_allow_html=True)
        with col4:
            valid_fields = sum(1 for v in validation.values() if v.get('is_valid', False))
            st.markdown('<div class="metric-container"><h3>Valid Fields</h3><p style="font-size: 1.5rem; font-weight: bold;">{}/{} ({:.1f}%)</p></div>'.format(
                valid_fields, len(validation), (valid_fields / max(1, len(validation)) * 100)), unsafe_allow_html=True)
        
        # Display output in tabs
        tab1, tab2, tab3 = st.tabs(["Formatted Output", "Validation Details", "Raw Data"])
        
        with tab1:
            st.markdown('<h3 class="sub-header">Formatted Output</h3>', unsafe_allow_html=True)
            st.json(output)
            
            # Download button for JSON
            json_str = json.dumps(output, indent=4)
            st.download_button(
                label="Download as JSON",
                data=json_str,
                file_name=f"{uploaded_file.name.split('.')[0]}_extracted.json",
                mime="application/json"
            )
        
        with tab2:
            st.markdown('<h3 class="sub-header">Validation Details</h3>', unsafe_allow_html=True)
            
            # Create validation dataframe
            validation_data = []
            for field, data in validation.items():
                validation_data.append({
                    "Field": field,
                    "Value": data.get('value', ''),
                    "Field Type": data.get('field_type', ''),
                    "Is Valid": "✅" if data.get('is_valid', False) else "❌",
                    "Issues": ", ".join(data.get('issues', [])) if data.get('issues') else "None"
                })
            
            df = pd.DataFrame(validation_data)
            st.dataframe(df, use_container_width=True)
        
        with tab3:
            st.markdown('<h3 class="sub-header">Raw Extraction Data</h3>', unsafe_allow_html=True)
            st.json(results)
    
    # Footer
    st.markdown("---")
    st.markdown("PDF Field Extraction System | Built with Streamlit, Transformers, and PyPDF2")

if __name__ == "__main__":
    main()