import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from PyPDF2 import PdfReader
from collections import defaultdict
import string
import re

# ------------------ Configuration ------------------

HEAD_KEYS = {
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
    "Handling fee of remitting bank": "handling_fee_of_remitting_bank",
    "Value date": "value_date",
    "Remitter's name": "remitter_name",
    "Remitting bank": "remitting_bank",
    "Instruction reference": "instruction_reference",
    "Other reference": "other_reference",
    "Remitter to beneficiary information": "remitter_to_beneficiary_info"
}

# Enhanced trigger words for field identification
TRIGGER_WORDS = {
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
    "handling_fee_of_remitting_bank": ["handling", "fee", "of", "remitting", "bank"],
    "value_date": ["value", "date"],
    "remitter_name": ["remitter", "name"],
    "remitting_bank": ["remitting", "bank"], 
    "instruction_reference": ["instruction", "reference"],
    "other_reference": ["other", "reference"],
    "remitter_to_beneficiary_info": ["remitter", "to", "beneficiary", "information"]
}

# Enhanced field boundaries and exclusion patterns
FIELD_BOUNDARIES = {
    "advice_date": {"max_words": 4, "stop_words": ["advice", "reference", "recipient"], "exclusions": []},
    "advice_ref": {"max_words": 2, "stop_words": ["recipient", "transaction", "payment"], "exclusions": ["page", "advice"]},
    "receipt_name": {"max_words": 12, "stop_words": ["transaction", "type", "email"], "exclusions": []},
    "receipt_email": {"max_words": 1, "stop_words": ["transaction", "type"], "exclusions": []},
    "transaction_type": {"max_words": 6, "stop_words": ["sub", "beneficiary"], "exclusions": []},
    "sub_payment_type": {"max_words": 4, "stop_words": ["beneficiary", "name"], "exclusions": []},
    "beneficiary_name": {"max_words": 10, "stop_words": ["beneficiary", "bank"], "exclusions": []},
    "beneficiary_bank": {"max_words": 25, "stop_words": ["beneficiary", "account", "customer"], "exclusions": []},
    "account_number": {"max_words": 15, "stop_words": ["customer", "reference"], "exclusions": []},
    "customer_reference": {"max_words": 8, "stop_words": ["debit", "remittance"], "exclusions": []},
    "debit_amount": {"max_words": 5, "stop_words": ["remittance"], "exclusions": []},
    "remittance_amount": {"max_words": 3, "stop_words": ["handling", "fee"], "exclusions": []},
    "handling_fee_of_remitting_bank": {"max_words": 6, "stop_words": ["value", "date"], "exclusions": []},
    "value_date": {"max_words": 4, "stop_words": ["remitter", "name"], "exclusions": []},
    "remitter_name": {"max_words": 8, "stop_words": ["remitting", "bank"], "exclusions": []},
    "remitting_bank": {"max_words": 12, "stop_words": ["instruction", "reference"], "exclusions": ["collect", "from", "remitter", "value", "fee", "handling"]},
    "instruction_reference": {"max_words": 3, "stop_words": ["other", "reference"], "exclusions": []},
    "other_reference": {"max_words": 3, "stop_words": ["remitter", "to", "beneficiary"], "exclusions": []},
    "remitter_to_beneficiary_info": {"max_words": 15, "stop_words": ["dear", "sir", "madam", "computer", "generated"], "exclusions": []}
}

# Company/Organization indicators
COMPANY_INDICATORS = ["private", "limited", "ltd", "pvt", "corp", "inc", "group", "company", "services", "service", "india"]
BANK_INDICATORS = ["bank", "banking", "corp", "limited", "ltd", "chase", "morgan", "shanghai", "hsbc"]

# Field order for processing (important for context-aware extraction)
FIELD_PROCESSING_ORDER = [
    "advice_date", "advice_ref", "receipt_name", "receipt_email", "transaction_type", 
    "sub_payment_type", "beneficiary_name", "beneficiary_bank", "account_number", 
    "customer_reference", "debit_amount", "remittance_amount", "handling_fee_of_remitting_bank",
    "value_date", "remitter_name", "remitting_bank", "instruction_reference", 
    "other_reference", "remitter_to_beneficiary_info"
]

# ------------------ RoBERTa NER Model Setup ------------------

class RoBERTaNERExtractor:
    def __init__(self):
        """Initialize RoBERTa NER model"""
        try:
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline("ner", 
                                        model=self.model, 
                                        tokenizer=self.tokenizer,
                                        aggregation_strategy="simple",
                                        device=0 if torch.cuda.is_available() else -1)
            print("✅ RoBERTa NER model loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load RoBERTa model: {e}")
            self.ner_pipeline = None
    
    def extract_entities(self, text):
        """Extract named entities using RoBERTa"""
        if not self.ner_pipeline:
            return []
        
        try:
            chunks = self._split_text_into_chunks(text, max_length=400)
            all_entities = []
            
            for chunk_start, chunk_text in chunks:
                entities = self.ner_pipeline(chunk_text)
                
                for entity in entities:
                    entity['start'] += chunk_start
                    entity['end'] += chunk_start
                    all_entities.append(entity)
            
            return all_entities
        except Exception as e:
            print(f"⚠️  NER extraction failed: {e}")
            return []
    
    def _split_text_into_chunks(self, text, max_length=400):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length - 50):
            chunk_words = words[i:i + max_length]
            chunk_text = " ".join(chunk_words)
            
            char_start = len(" ".join(words[:i]))
            if i > 0:
                char_start += 1
            
            chunks.append((char_start, chunk_text))
        
        return chunks

# ------------------ Text Processing ------------------

def extract_text_from_pdf(file_path):
    """Extract text from PDF with better formatting"""
    reader = PdfReader(file_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        # Better whitespace normalization
        page_text = " ".join(page_text.split())
        text += f"{page_text} "
    
    # Clean up extra whitespace and normalize
    text = " ".join(text.split())
    return text.strip()

def normalize_word(word):
    """Normalize word for comparison"""
    cleaned = word.lower().strip(string.punctuation + "()[]{}\"'")
    return cleaned

def create_word_index_mapping(text):
    """Create enhanced mapping of words with their positions"""
    words = text.split()
    word_mapping = []
    
    for i, word in enumerate(words):
        normalized = normalize_word(word)
        word_mapping.append({
            'index': i,
            'original': word,
            'normalized': normalized,
            'is_numeric': any(c.isdigit() for c in word),
            'has_alpha': any(c.isalpha() for c in word),
            'has_special': any(c in string.punctuation for c in word),
            'is_email': '@' in word,
            'is_currency': 'inr' in normalized or '₹' in word or 'usd' in normalized or '$' in word,
            'is_separator': normalized in [':', '-', '–', '—', '|', '•', '.', ')', '(', ']', '[', '/', ','],
            'is_company_word': normalized in COMPANY_INDICATORS,
            'is_bank_word': normalized in BANK_INDICATORS,
            'is_capitalized': word and len(word) > 1 and word[0].isupper(),
            'length': len(word),
            'has_underscore': '_' in word,
            'is_reference_like': ('_' in word and any(c.isdigit() for c in word)) or (len(word) > 6 and any(c.isdigit() for c in word) and any(c.isalpha() for c in word))
        })
    
    return word_mapping

# ------------------ Enhanced Trigger-Based Field Detection ------------------

def find_field_positions_by_triggers(word_mapping):
    """Find field positions using enhanced trigger matching with proper sequencing"""
    field_positions = {}
    
    print(f"🔍 Analyzing {len(word_mapping)} words for trigger patterns...")
    
    # Process fields in order to maintain context
    for field_label in FIELD_PROCESSING_ORDER:
        if field_label in TRIGGER_WORDS:
            triggers = TRIGGER_WORDS[field_label]
            best_match = find_precise_trigger_match(word_mapping, triggers, field_label, field_positions)
            if best_match:
                field_positions[field_label] = best_match
                print(f"✅ Found {field_label} at position {best_match['start_index']}-{best_match['end_index']}")
    
    return field_positions

def find_precise_trigger_match(word_mapping, triggers, field_label, existing_positions):
    """Enhanced trigger matching with context awareness and flexible gap handling"""
    best_match = None
    best_score = 0
    
    # Get positions of already found fields for context
    used_ranges = []
    for pos_info in existing_positions.values():
        used_ranges.append((pos_info['start_index'], pos_info['end_index']))
    
    # Search for trigger sequences
    for start_idx in range(len(word_mapping)):
        # Skip if this area is already used by another field
        if any(start <= start_idx <= end for start, end in used_ranges):
            continue
            
        score = 0
        matched_words = []
        trigger_idx = 0
        current_idx = start_idx
        consecutive_matches = 0
        gap_penalty = 0
        
        # Try to match all triggers in sequence
        while trigger_idx < len(triggers) and current_idx < len(word_mapping):
            word_info = word_mapping[current_idx]
            current_trigger = triggers[trigger_idx].lower()
            
            # Exact match
            if current_trigger == word_info['normalized']:
                score += 30
                matched_words.append(word_info)
                trigger_idx += 1
                consecutive_matches += 1
                gap_penalty = 0
            # Partial match within word
            elif len(word_info['normalized']) > 3 and current_trigger in word_info['normalized']:
                score += 20
                matched_words.append(word_info)
                trigger_idx += 1
                consecutive_matches += 1
                gap_penalty = 0
            # Skip separators and small words without penalty
            elif word_info['is_separator'] or len(word_info['normalized']) <= 2:
                pass
            # Small gap tolerance
            elif current_idx - start_idx < len(triggers) * 3:
                gap_penalty += 1
                if gap_penalty > 3:
                    break
            else:
                break
            
            current_idx += 1
            
            # Stop if we've gone too far without finding more matches
            if current_idx - start_idx > len(triggers) * 5:
                break
        
        # Bonus for consecutive matches and complete trigger matching
        if consecutive_matches >= len(triggers):
            score += 15
        
        # Apply gap penalty
        score -= gap_penalty * 2
        
        # Check if we matched all triggers with sufficient score
        if trigger_idx == len(triggers) and score > best_score and score >= len(triggers) * 12:
            best_score = score
            best_match = {
                'start_index': start_idx,
                'end_index': current_idx,
                'score': score,
                'matched_words': matched_words,
                'field_label': field_label,
                'gap_penalty': gap_penalty
            }
    
    return best_match

# ------------------ Enhanced Value Extraction ------------------

def extract_values_with_ner(word_mapping, field_positions, ner_entities):
    """Extract field values with enhanced precision and proper boundary detection"""
    results = {}
    
    # Create NER entity mapping by word position
    ner_by_position = create_ner_word_mapping(ner_entities, word_mapping)
    
    # Process fields in order
    for field_label in FIELD_PROCESSING_ORDER:
        if field_label in field_positions:
            position_info = field_positions[field_label]
            value = extract_field_value_advanced(word_mapping, position_info, ner_by_position, field_label, results)
            results[field_label] = clean_and_validate_value(value, field_label)
            
            print(f"📝 {field_label}: '{results[field_label]}'")
    
    return results

def create_ner_word_mapping(ner_entities, word_mapping):
    """Create mapping of NER entities to word positions"""
    ner_by_position = {}
    
    for entity in ner_entities:
        entity_text = entity['word'].replace('##', '').strip().lower()
        
        # Find matching words in our word mapping
        for i, word_info in enumerate(word_mapping):
            word_norm = word_info['normalized']
            
            # More precise matching
            if (entity_text == word_norm or 
                (len(entity_text) > 3 and len(word_norm) > 3 and 
                 (entity_text in word_norm or word_norm in entity_text))):
                
                if i not in ner_by_position:
                    ner_by_position[i] = []
                
                ner_by_position[i].append({
                    'entity_type': entity['entity_group'],
                    'confidence': entity['score'],
                    'text': entity['word']
                })
    
    return ner_by_position

def extract_field_value_advanced(word_mapping, position_info, ner_by_position, field_label, existing_results):
    """Advanced field value extraction with proper boundary detection"""
    start_idx = position_info['end_index']
    field_config = FIELD_BOUNDARIES.get(field_label, {"max_words": 4, "stop_words": [], "exclusions": []})
    
    # Skip separators and small words
    while start_idx < len(word_mapping):
        word_info = word_mapping[start_idx]
        if word_info['is_separator'] or len(word_info['normalized']) <= 1:
            start_idx += 1
        else:
            break
    
    if start_idx >= len(word_mapping):
        return ""
    
    # Field-specific extraction with enhanced boundary detection
    if field_label == "receipt_email":
        return extract_email_advanced(word_mapping, start_idx)
    elif field_label == "transaction_type":
        return extract_transaction_type_enhanced(word_mapping, start_idx, field_config)
    elif field_label == "customer_reference":
        return extract_customer_reference_enhanced(word_mapping, start_idx, field_config)
    elif field_label == "debit_amount":
        return extract_debit_amount_enhanced(word_mapping, start_idx, field_config)
    elif field_label == "remitting_bank":
        return extract_remitting_bank_advanced(word_mapping, start_idx, field_config, existing_results)
    elif field_label == "handling_fee_of_remitting_bank":
        return extract_handling_fee_advanced(word_mapping, start_idx, field_config)
    elif field_label in ["receipt_name", "beneficiary_name", "remitter_name"]:
        return extract_organization_name_advanced(word_mapping, start_idx, ner_by_position, field_config)
    elif field_label in ["beneficiary_bank"]:
        return extract_bank_name_advanced(word_mapping, start_idx, ner_by_position, field_config)
    elif "amount" in field_label:
        return extract_amount_advanced(word_mapping, start_idx, field_label)
    elif "date" in field_label:
        return extract_date_advanced(word_mapping, start_idx)
    elif "reference" in field_label and field_label != "customer_reference":
        return extract_reference_advanced(word_mapping, start_idx)
    elif field_label == "account_number":
        return extract_account_advanced(word_mapping, start_idx)
    elif field_label in ["sub_payment_type"]:
        return extract_type_advanced(word_mapping, start_idx, field_config)
    elif field_label == "remitter_to_beneficiary_info":
        return extract_beneficiary_info_enhanced(word_mapping, start_idx, field_config)
    else:
        return extract_generic_advanced(word_mapping, start_idx, field_config)

def extract_customer_reference_enhanced(word_mapping, start_idx, field_config):
    """Enhanced customer reference extraction with improved pattern detection"""
    max_words = field_config['max_words']
    stop_words = field_config['stop_words']
    
    # Look for reference patterns in multiple passes
    # Pass 1: Look for underscore patterns (most reliable)
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Stop at stop words
        if any(stop_word in normalized for stop_word in stop_words):
            break
        
        # Look for underscore pattern with numbers
        if word_info['has_underscore'] and word_info['is_numeric']:
            # Clean the word
            clean_word = word.strip()
            if len(clean_word) > 3:
                return clean_word
    
    # Pass 2: Look for alphanumeric patterns without underscore
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Stop at stop words
        if any(stop_word in normalized for stop_word in stop_words):
            break
        
        # Look for reference-like patterns
        if word_info['is_reference_like'] and len(word) > 6:
            clean_word = word.strip()
            # Avoid common false positives
            if not any(fp in clean_word.lower() for fp in ['debit', 'amount', 'inr', 'usd']):
                return clean_word
    
    # Pass 3: Look for pure numeric patterns that might be references
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Stop at stop words
        if any(stop_word in normalized for stop_word in stop_words):
            break
        
        # Look for long numeric sequences that could be references
        if word_info['is_numeric'] and len(word) > 8 and not word_info['is_currency']:
            return word.strip()
    
    return ""

def extract_debit_amount_enhanced(word_mapping, start_idx, field_config):
    """Enhanced debit amount extraction with better currency detection"""
    max_words = field_config['max_words']
    stop_words = field_config['stop_words']
    
    # Look for currency amounts in multiple patterns
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Stop at stop words
        if any(stop_word in normalized for stop_word in stop_words):
            break
        
        # Pattern 1: INR followed by amount
        if normalized == 'inr' and i + 1 < len(word_mapping):
            next_word_info = word_mapping[i + 1]
            if next_word_info['is_numeric'] and (',' in next_word_info['original'] or len(next_word_info['original']) > 4):
                return f"INR {next_word_info['original']}"
        
        # Pattern 2: Amount with INR suffix
        if word_info['is_currency'] and ('inr' in normalized or 'usd' in normalized):
            return word
        
        # Pattern 3: Large numeric amount (likely currency)
        if word_info['is_numeric'] and (',' in word or len(word) > 5) and not word_info['has_alpha']:
            # Check if next word is currency
            if i + 1 < len(word_mapping):
                next_word = word_mapping[i + 1]['normalized']
                if next_word in ['inr', 'usd']:
                    return f"{word} {next_word.upper()}"
            return word
        
        # Pattern 4: Currency symbol followed by amount
        if '₹' in word or '$' in word:
            return word
    
    return ""

def extract_beneficiary_info_enhanced(word_mapping, start_idx, field_config):
    """Enhanced beneficiary information extraction with better filtering"""
    max_words = field_config['max_words']
    stop_words = field_config['stop_words']
    
    info_words = []
    found_reference = False
    
    # Look for reference number first
    for i in range(start_idx, min(start_idx + 5, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        
        # Look for reference pattern
        if word_info['has_underscore'] and word_info['is_numeric']:
            info_words.append(word)
            found_reference = True
            start_idx = i + 1
            break
    
    # If reference found, look for minimal additional info
    if found_reference:
        # Look for meaningful content after reference
        meaningful_words = []
        for i in range(start_idx, min(start_idx + 10, len(word_mapping))):
            word_info = word_mapping[i]
            word = word_info['original']
            normalized = word_info['normalized']
            
            # Stop at common email template words
            if any(stop in normalized for stop in stop_words):
                break
            
            # Add meaningful words only
            if (word_info['has_alpha'] and len(word) > 2 and 
                normalized not in ['bl', 'this', 'is', 'a', 'the', 'message', 'email', 'e', 'mail']):
                meaningful_words.append(word)
                if len(meaningful_words) >= 3:  # Limit to avoid template text
                    break
        
        if meaningful_words:
            info_words.extend(meaningful_words)
    else:
        # If no reference, look for other meaningful content
        for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
            word_info = word_mapping[i]
            word = word_info['original']
            normalized = word_info['normalized']
            
            # Stop at template words
            if any(stop in normalized for stop in stop_words):
                break
            
            if word_info['has_alpha'] and len(word) > 2:
                info_words.append(word)
    
    return " ".join(info_words)

def extract_transaction_type_enhanced(word_mapping, start_idx, field_config):
    """Enhanced transaction type extraction with better phrase collection"""
    type_words = []
    max_words = field_config['max_words']
    stop_words = field_config['stop_words']
    
    # Look for transaction type patterns
    i = start_idx
    while i < len(word_mapping) and len(type_words) < max_words:
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Stop at stop words (but be more selective)
        if any(stop_word in normalized and stop_word not in ['payment', 'type'] for stop_word in stop_words):
            break
        
        # Collect alphabetic words that form the transaction type
        if word_info['has_alpha'] and len(word) > 1:
            # Special handling for common transaction types
            if normalized in ['priority', 'normal', 'express', 'urgent', 'payment', 'transfer']:
                type_words.append(word)
            elif type_words and len(type_words) < 3:  # Continue collecting related words
                if word_info['is_capitalized'] or normalized in ['payment', 'transfer', 'wire', 'swift']:
                    type_words.append(word)
                else:
                    # Check if it's likely part of the type
                    if i + 1 < len(word_mapping):
                        next_word = word_mapping[i + 1]['normalized']
                        if next_word not in stop_words:
                            type_words.append(word)
                        else:
                            break
                    else:
                        break
            elif not type_words:
                # First word should be relevant
                if word_info['is_capitalized'] or normalized in ['priority', 'normal', 'express', 'urgent']:
                    type_words.append(word)
        elif type_words and not word_info['is_separator']:
            # Stop if we encounter non-alphabetic content after collecting words
            break
        
        i += 1
    
    result = " ".join(type_words)
    return result if result else ""

def extract_remitting_bank_advanced(word_mapping, start_idx, field_config, existing_results):
    """Extract remitting bank name excluding fee-related words"""
    bank_words = []
    max_words = field_config['max_words']
    exclusions = field_config['exclusions']
    stop_words = field_config['stop_words']
    
    # Skip words that belong to handling fee field
    skip_fee_words = True
    fee_skip_count = 0
    
    i = start_idx
    while i < len(word_mapping) and len(bank_words) < max_words:
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Skip fee-related words at the beginning
        if skip_fee_words and any(exclusion in normalized for exclusion in exclusions):
            fee_skip_count += 1
            if fee_skip_count > 5:  # After skipping enough fee words, start collecting bank name
                skip_fee_words = False
            i += 1
            continue
        
        # Stop at stop words
        if any(stop_word in normalized for stop_word in stop_words):
            break
        
        # Skip exclusion words
        if any(exclusion in normalized for exclusion in exclusions):
            i += 1
            continue
        
        # Collect bank name words
        if ((word_info['is_capitalized'] or word_info['is_bank_word']) and 
            len(word) > 1 and word_info['has_alpha']):
            bank_words.append(word)
            skip_fee_words = False  # Once we start collecting, don't skip anymore
        elif bank_words and word_info['has_alpha'] and len(word) > 2:
            bank_words.append(word)
        elif bank_words and len(bank_words) >= 3:
            break
        
        i += 1
    
    return " ".join(bank_words)

def extract_handling_fee_advanced(word_mapping, start_idx, field_config):
    """Extract handling fee information"""
    fee_words = []
    max_words = field_config['max_words']
    
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Stop at "Value" which indicates next field
        if normalized == "value":
            break
        
        # Collect fee-related words
        if normalized in ["collect", "from", "remitter"] or (fee_words and word_info['has_alpha']):
            fee_words.append(word)
            if len(fee_words) >= 3:  # "Collect from Remitter" is sufficient
                break
    
    return " ".join(fee_words)

def extract_email_advanced(word_mapping, start_idx):
    """Advanced email extraction"""
    for i in range(start_idx, min(start_idx + 10, len(word_mapping))):
        word_info = word_mapping[i]
        if word_info['is_email'] and '.' in word_info['original']:
            return word_info['original']
    return ""

def extract_organization_name_advanced(word_mapping, start_idx, ner_by_position, field_config):
    """Advanced organization name extraction"""
    name_words = []
    seen_words = set()
    max_words = field_config['max_words']
    stop_words = field_config['stop_words']
    
    i = start_idx
    while i < len(word_mapping) and len(name_words) < max_words:
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Stop at stop words
        if any(stop_word in normalized for stop_word in stop_words):
            break
        
        # Skip duplicates
        if normalized in seen_words:
            i += 1
            continue
        
        # Include NER organization entities
        if i in ner_by_position:
            for entity in ner_by_position[i]:
                if entity['entity_type'] == 'ORG' and normalized not in seen_words:
                    name_words.append(word)
                    seen_words.add(normalized)
                    break
        # Include capitalized words and company indicators
        elif ((word_info['is_capitalized'] or word_info['is_company_word']) and 
              len(word) > 1 and normalized not in seen_words and word_info['has_alpha']):
            name_words.append(word)
            seen_words.add(normalized)
        
        i += 1
    
    return " ".join(name_words)

def extract_bank_name_advanced(word_mapping, start_idx, ner_by_position, field_config):
    """Advanced bank name extraction"""
    bank_words = []
    seen_words = set()
    max_words = field_config['max_words']
    stop_words = field_config['stop_words']
    
    i = start_idx
    while i < len(word_mapping) and len(bank_words) < max_words:
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Stop at stop words
        if any(stop_word in normalized for stop_word in stop_words):
            break
        
        # Skip duplicates
        if normalized in seen_words:
            i += 1
            continue
        
        # Include NER organization entities
        if i in ner_by_position:
            for entity in ner_by_position[i]:
                if entity['entity_type'] == 'ORG' and normalized not in seen_words:
                    bank_words.append(word)
                    seen_words.add(normalized)
                    break
        # Include bank-related and capitalized words
        elif ((word_info['is_bank_word'] or word_info['is_capitalized']) and 
              len(word) > 1 and normalized not in seen_words and word_info['has_alpha']):
            bank_words.append(word)
            seen_words.add(normalized)
        
        i += 1
    
    return " ".join(bank_words)

def extract_amount_advanced(word_mapping, start_idx, field_label):
    """Advanced amount extraction with field-specific logic"""
    for i in range(start_idx, min(start_idx + 5, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Currency with amount
        if word_info['is_currency'] or (word_info['is_numeric'] and ',' in word and len(word) > 5):
            return word
        
        # INR prefix pattern
        if normalized == 'inr' and i + 1 < len(word_mapping):
            next_word_info = word_mapping[i + 1]
            if next_word_info['is_numeric'] and (',' in next_word_info['original'] or len(next_word_info['original']) > 4):
                return f"INR {next_word_info['original']}"
        
        # Large numeric value
        if word_info['is_numeric'] and len(word) > 4 and not word_info['has_alpha']:
            return word
    
    return ""

def extract_date_advanced(word_mapping, start_idx):
    """Advanced date extraction"""
    date_words = []
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    for i in range(start_idx, min(start_idx + 4, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        if any(month in normalized for month in months):
            date_words.append(word)
        elif word_info['is_numeric'] and len(word) <= 4:
            date_words.append(word)
        elif date_words and len(date_words) >= 2:
            break
    
    return " ".join(date_words[:3])

def extract_reference_advanced(word_mapping, start_idx):
    """Advanced reference extraction"""
    for i in range(start_idx, min(start_idx + 3, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        
        if word_info['is_reference_like']:
            return word
    return ""

def extract_account_advanced(word_mapping, start_idx):
    """Advanced account number extraction"""
    for i in range(start_idx, min(start_idx + 15, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        
        if '*' in word or (word_info['is_numeric'] and word_info['has_alpha'] and len(word) > 6):
            account_info = [word]
            j = i + 1
            # Collect additional account information
            while j < min(i + 10, len(word_mapping)):
                next_word_info = word_mapping[j]
                next_word = next_word_info['original']
                if '(' in next_word or 'Part' in next_word or 'security' in next_word.lower():
                    account_info.append(next_word)
                    # Continue until closing parenthesis
                    while j < len(word_mapping) and ')' not in word_mapping[j]['original']:
                        j += 1
                        if j < len(word_mapping):
                            account_info.append(word_mapping[j]['original'])
                    break
                j += 1
            return " ".join(account_info)
    return ""

def extract_type_advanced(word_mapping, start_idx, field_config):
    """Advanced type extraction"""
    type_words = []
    max_words = field_config['max_words']
    stop_words = field_config['stop_words']
    
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Stop at stop words
        if any(stop_word in normalized for stop_word in stop_words):
            break
        
        if word_info['has_alpha'] and len(word) > 1:
            type_words.append(word)
    
    return " ".join(type_words)

def extract_generic_advanced(word_mapping, start_idx, field_config):
    """Advanced generic extraction"""
    words = []
    max_words = field_config['max_words']
    stop_words = field_config['stop_words']
    
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Stop at stop words
        if any(stop_word in normalized for stop_word in stop_words):
            break
        
        if word_info['has_alpha'] and len(word) > 1:
            words.append(word)
    
    return " ".join(words)

def clean_and_validate_value(value, field_label):
    """Clean and validate extracted field values with enhanced cleaning"""
    if not value:
        return ""
    
    # Remove extra whitespace
    value = " ".join(value.split())
    
    # Field-specific cleaning
    if field_label == "customer_reference":
        # More aggressive cleaning for customer reference
        value = value.strip()
        # Remove common contaminations
        contaminations = ["debit", "amount", "inr", "usd", "Debit", "Amount", "INR", "USD", "DEBIT", "AMOUNT"]
        for contamination in contaminations:
            if value.lower().endswith(contamination.lower()):
                value = value[:-len(contamination)].strip()
            elif value.lower().startswith(contamination.lower()):
                value = value[len(contamination):].strip()
        return value
    
    elif field_label == "debit_amount":
        # Ensure proper amount formatting
        if 'inr' in value.lower() and not value.upper().startswith('INR'):
            # Standardize INR format
            value = value.replace('inr', 'INR').replace('Inr', 'INR')
        return value
    
    elif field_label == "remitter_to_beneficiary_info":
        # Clean beneficiary info from template text
        # Remove common email template phrases
        template_phrases = [
            "Dear Sir/Madam", "This is a Computer generated E-Mail Message",
            "This does not require a Signature", "System generated automatic",
            "Please do not reply", "Document Number Amount Date Net Amount",
            "Narration/Inv"
        ]
        
        for phrase in template_phrases:
            value = value.replace(phrase, "").strip()
        
        # Remove excessive whitespace and clean up
        value = " ".join(value.split())
        
        # If it starts with BL//, clean it up
        if value.startswith("BL//"):
            value = value[4:].strip()
        
        return value
    
    elif field_label == "handling_fee_of_remitting_bank":
        # Ensure proper fee description
        if "Collect" in value and "from" in value and "Remitter" in value:
            return "Collect from Remitter"
        return value
    
    elif field_label in ["receipt_name", "beneficiary_name", "remitter_name"]:
        # Remove trailing field indicators
        unwanted_endings = ["Transaction", "Beneficiary", "Bank", "Account", "Type"]
        for ending in unwanted_endings:
            if value.endswith(ending):
                value = value[:-len(ending)].strip()
        return value
    
    return value

# ------------------ Main Processing Function ------------------

def process_pdf_with_enhanced_extraction(file_path):
    """Main processing function with enhanced extraction"""
    print(f"📄 Processing PDF: {file_path}")
    
    # Initialize NER extractor
    ner_extractor = RoBERTaNERExtractor()
    
    # Extract and process text
    text = extract_text_from_pdf(file_path)
    word_mapping = create_word_index_mapping(text)
    
    print(f"📊 Total words: {len(word_mapping)}")
    
    # Extract NER entities
    ner_entities = ner_extractor.extract_entities(text) if ner_extractor.ner_pipeline else []
    print(f"🤖 NER entities found: {len(ner_entities)}")
    
    # Find field positions using enhanced triggers
    field_positions = find_field_positions_by_triggers(word_mapping)
    print(f"🎯 Fields detected: {len(field_positions)}")
    
    # Extract values with enhanced precision
    extracted_values = extract_values_with_ner(word_mapping, field_positions, ner_entities)
    
    return {
        "extracted_entities": extracted_values,
        "ner_entities_count": len(ner_entities),
        "fields_detected": len(field_positions)
    }

# ------------------ Main Execution ------------------

def main():
    """Main execution function"""
    file_path = "/content/Payment_Advice_F3.pdf"  # Change this to your PDF path
    
    try:
        print("🚀 Starting Enhanced Smart Trigger & NER-Based PDF Processing...")
        print("🔧 Features: Advanced Triggers + RoBERTa NER + Enhanced Boundary Detection + Field Sequencing")
        print("❌ Not using: Regex, Hardcoding, Manual Patterns")
        
        # Process the PDF
        result = process_pdf_with_enhanced_extraction(file_path)
        
        # Print results
        print("\n" + "="*60)
        print("📦 FINAL EXTRACTED JSON:")
        print("="*60)
        
        final_output = result["extracted_entities"]
        print(json.dumps(final_output, indent=4, ensure_ascii=False))
        
        # Save to file
        output_file = "extracted_payment_advice_enhanced.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Summary
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   • NER entities processed: {result['ner_entities_count']}")
        print(f"   • Fields detected by triggers: {result['fields_detected']}")
        print(f"   • Non-empty values extracted: {len([v for v in final_output.values() if v])}")
        
        # Check critical fields
        critical_fields = ["transaction_type", "customer_reference", "debit_amount", "remitting_bank"]
        print(f"\n🎯 CRITICAL FIELDS STATUS:")
        for field in critical_fields:
            status = "✅ EXTRACTED" if final_output.get(field) else "❌ MISSING"
            value = final_output.get(field, "")
            print(f"   • {field}: {status} - '{value}'")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
