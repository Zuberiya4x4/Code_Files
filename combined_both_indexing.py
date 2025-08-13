#COMBINED CODE FOR BOTH WORD BY WORD AND FIELD BY VALUE INDEXING
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from PyPDF2 import PdfReader
from collections import defaultdict
import string
import re
from typing import Dict, List, Tuple, Optional, Any

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

# Enhanced trigger words with more flexible patterns
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

# Improved field boundaries with better stop words for separation
FIELD_BOUNDARIES = {
    "advice_date": {"max_words": 6, "stop_words": ["advice", "reference", "no", "recipient"]},
    "advice_ref": {"max_words": 4, "stop_words": ["recipient", "name", "email"]},
    "receipt_name": {"max_words": 15, "stop_words": ["recipient", "email", "transaction", "type"]},
    "receipt_email": {"max_words": 3, "stop_words": ["transaction", "type", "sub", "payment"]},
    "transaction_type": {"max_words": 8, "stop_words": ["sub", "payment", "type", "beneficiary"]},
    "sub_payment_type": {"max_words": 8, "stop_words": ["beneficiary", "name", "bank"]},
    "beneficiary_name": {"max_words": 15, "stop_words": ["beneficiary", "bank", "account"]},
    "beneficiary_bank": {"max_words": 25, "stop_words": ["beneficiary", "account", "customer", "reference"]},
    "account_number": {"max_words": 20, "stop_words": ["customer", "reference", "debit", "amount"]},
    "customer_reference": {"max_words": 6, "stop_words": ["debit", "amount", "remittance"]},
    "debit_amount": {"max_words": 8, "stop_words": ["remittance", "amount", "handling", "fee"]},
    "remittance_amount": {"max_words": 8, "stop_words": ["handling", "fee", "remitting", "bank"]},
    "handling_fee_of_remitting_bank": {"max_words": 10, "stop_words": ["value", "date", "remitter"]},
    "value_date": {"max_words": 6, "stop_words": ["remitter", "name", "remitting", "bank"]},
    "remitter_name": {"max_words": 15, "stop_words": ["remitting", "bank", "instruction", "reference"]},
    "remitting_bank": {"max_words": 20, "stop_words": ["instruction", "reference", "other"]},
    "instruction_reference": {"max_words": 6, "stop_words": ["other", "reference", "remitter", "to"]},
    "other_reference": {"max_words": 6, "stop_words": ["remitter", "to", "beneficiary", "information"]},
    "remitter_to_beneficiary_info": {"max_words": 50, "stop_words": ["dear", "sir", "madam", "computer", "generated"]}
}

# Enhanced patterns for better recognition
BANK_KEYWORDS = [
    "bank", "banking", "corp", "corporation", "limited", "ltd", "pvt", "private",
    "chase", "morgan", "shanghai", "hsbc", "hdfc", "icici", "sbi", "axis",
    "standard", "chartered", "citibank", "wells", "fargo", "barclays"
]

COMPANY_KEYWORDS = [
    "limited", "ltd", "pvt", "private", "corp", "corporation", "inc", "incorporated",
    "group", "company", "co", "services", "service", "enterprises", "solutions",
    "technologies", "tech", "india", "international", "global"
]

MONTH_NAMES = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december"
]

# Field processing order optimized for dependency resolution
FIELD_PROCESSING_ORDER = [
    "advice_date", "advice_ref", "receipt_name", "receipt_email", "transaction_type",
    "sub_payment_type", "beneficiary_name", "beneficiary_bank", "account_number",
    "customer_reference", "debit_amount", "remittance_amount", "handling_fee_of_remitting_bank",
    "value_date", "remitter_name", "remitting_bank", "instruction_reference",
    "other_reference", "remitter_to_beneficiary_info"
]

# ------------------ Enhanced RoBERTa NER Model Setup ------------------

class EnhancedRoBERTaNERExtractor:
    def __init__(self):
        """Initialize RoBERTa NER model with enhanced configuration"""
        try:
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Enhanced RoBERTa NER model loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load RoBERTa model: {e}")
            self.ner_pipeline = None

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities with improved processing"""
        if not self.ner_pipeline:
            return []

        try:
            # Split text into manageable chunks
            chunks = self._create_smart_chunks(text, max_length=400)
            all_entities = []

            for chunk_start, chunk_text in chunks:
                try:
                    entities = self.ner_pipeline(chunk_text)
                    
                    for entity in entities:
                        # Adjust positions to global coordinates
                        entity['start'] += chunk_start
                        entity['end'] += chunk_start
                        entity['chunk_start'] = chunk_start
                        entity['confidence_score'] = entity.get('score', 0.0)
                        all_entities.append(entity)
                        
                except Exception as chunk_error:
                    print(f"⚠️  Error processing chunk: {chunk_error}")
                    continue

            # Filter and enhance entities
            filtered_entities = self._filter_and_enhance_entities(all_entities)
            return filtered_entities
            
        except Exception as e:
            print(f"⚠️  NER extraction failed: {str(e)}")
            return []

    def _create_smart_chunks(self, text: str, max_length: int = 400) -> List[Tuple[int, str]]:
        """Create smart chunks with better boundary detection"""
        words = text.split()
        chunks = []
        overlap = 50  # Overlap for context preservation
        
        for i in range(0, len(words), max_length - overlap):
            chunk_words = words[i:i + max_length]
            chunk_text = " ".join(chunk_words)
            
            # Calculate character start position
            char_start = len(" ".join(words[:i]))
            if i > 0:
                char_start += 1
                
            chunks.append((char_start, chunk_text))
            
        return chunks

    def _filter_and_enhance_entities(self, entities: List[Dict]) -> List[Dict]:
        """Filter and enhance NER entities"""
        enhanced_entities = []
        
        for entity in entities:
            # Skip low confidence entities
            if entity.get('confidence_score', 0) < 0.3:
                continue
                
            # Clean entity text
            entity_text = entity['word'].replace('##', '').strip()
            if len(entity_text) < 2:
                continue
                
            # Enhance entity information
            entity['cleaned_word'] = entity_text
            entity['is_bank_related'] = any(bank_word in entity_text.lower() 
                                          for bank_word in BANK_KEYWORDS)
            entity['is_company_related'] = any(comp_word in entity_text.lower() 
                                             for comp_word in COMPANY_KEYWORDS)
            
            enhanced_entities.append(entity)
            
        return enhanced_entities

# ------------------ Enhanced Text Processing ------------------

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF with improved formatting"""
    try:
        reader = PdfReader(file_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Split into lines and clean each line
                lines = page_text.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    # Clean line while preserving structure
                    cleaned_line = ' '.join(line.split())
                    if cleaned_line and len(cleaned_line.strip()) > 0:
                        cleaned_lines.append(cleaned_line)
                
                if cleaned_lines:
                    text_parts.append(' '.join(cleaned_lines))
        
        # Join all pages with space separator
        full_text = ' '.join(text_parts)
        
        # Final normalization
        full_text = ' '.join(full_text.split())
        
        return full_text.strip()
        
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {str(e)}")
        return ""

def create_comprehensive_word_mapping(text: str) -> List[Dict]:
    """Create comprehensive word mapping with enhanced metadata"""
    words = text.split()
    word_mapping = []
    char_position = 0
    
    for i, word in enumerate(words):
        original_word = word
        normalized_word = word.lower().strip(string.punctuation + "()[]{}\"'")
        
        # Enhanced word analysis
        word_info = {
            'index': i,
            'original': original_word,
            'normalized': normalized_word,
            'length': len(original_word),
            'char_start': char_position,
            'char_end': char_position + len(original_word),
            
            # Character type analysis
            'has_alpha': any(c.isalpha() for c in original_word),
            'has_numeric': any(c.isdigit() for c in original_word),
            'has_special': any(c in string.punctuation for c in original_word),
            'is_alpha_only': original_word.isalpha(),
            'is_numeric_only': original_word.isdigit(),
            'is_alphanumeric': original_word.isalnum(),
            
            # Format analysis
            'is_capitalized': original_word and len(original_word) > 1 and original_word[0].isupper(),
            'is_all_caps': original_word.isupper() and len(original_word) > 1,
            'is_title_case': original_word.istitle(),
            
            # Special character analysis
            'has_underscore': '_' in original_word,
            'has_hyphen': '-' in original_word,
            'has_dot': '.' in original_word,
            'has_comma': ',' in original_word,
            'has_at': '@' in original_word,
            'has_asterisk': '*' in original_word,
            'has_slash': '/' in original_word,
            'has_parentheses': '(' in original_word or ')' in original_word,
            
            # Content type analysis
            'is_email_like': '@' in original_word and '.' in original_word,
            'is_currency_symbol': original_word in ['₹', '$', '€', '£'],
            'is_currency_code': normalized_word in ['inr', 'usd', 'eur', 'gbp'],
            'is_month': normalized_word in MONTH_NAMES,
            'is_bank_word': normalized_word in BANK_KEYWORDS,
            'is_company_word': normalized_word in COMPANY_KEYWORDS,
            
            # Pattern analysis
            'is_reference_like': (
                ('_' in original_word and any(c.isdigit() for c in original_word)) or
                (len(original_word) > 6 and any(c.isdigit() for c in original_word) and 
                 any(c.isalpha() for c in original_word))
            ),
            'is_account_like': (
                '*' in original_word or 
                (len(original_word) > 8 and any(c.isdigit() for c in original_word))
            ),
            'is_large_number': original_word.isdigit() and len(original_word) > 4,
            'is_amount_like': (
                (',' in original_word and any(c.isdigit() for c in original_word)) or
                (any(c.isdigit() for c in original_word) and len(original_word) > 3)
            ),
            
            # Date pattern analysis
            'is_date_like': (
                (original_word.isdigit() and 1 <= len(original_word) <= 4) or
                normalized_word in MONTH_NAMES or
                (len(original_word) <= 4 and any(c.isdigit() for c in original_word))
            ),
            
            # Separator analysis
            'is_separator': normalized_word in [':', '-', '–', '—', '|', '•', '.', ')', '(', 
                                               ']', '[', '/', ',', ';', '--'],
            'is_connector': normalized_word in ['and', 'of', 'the', 'in', 'at', 'on', 'to', 'for']
        }
        
        # Update character position for next word
        char_position += len(original_word) + 1  # +1 for space
        
        word_mapping.append(word_info)
    
    return word_mapping

# ------------------ Enhanced Field Detection ------------------

def find_enhanced_field_positions(word_mapping: List[Dict]) -> Dict[str, Dict]:
    """Find field positions using enhanced trigger matching"""
    field_positions = {}
    used_ranges = []  # Track used word ranges to avoid overlap
    
    print(f"🔍 Analyzing {len(word_mapping)} words for field triggers...")
    
    # Process fields in order
    for field_label in FIELD_PROCESSING_ORDER:
        if field_label in TRIGGER_WORDS:
            triggers = TRIGGER_WORDS[field_label]
            
            # Find best match for this field
            best_match = find_best_trigger_match(
                word_mapping, triggers, field_label, used_ranges
            )
            
            if best_match:
                field_positions[field_label] = best_match
                
                # Mark this range as used (only the trigger part)
                used_ranges.append((
                    best_match['start_index'], 
                    best_match['end_index']
                ))
                
                print(f"✅ Located {field_label} at position {best_match['start_index']}-{best_match['end_index']} (score: {best_match['score']:.1f})")
            else:
                print(f"⚠️  Could not locate {field_label}")
    
    return field_positions

def find_best_trigger_match(word_mapping: List[Dict], triggers: List[str], 
                           field_label: str, used_ranges: List[Tuple[int, int]]) -> Optional[Dict]:
    """Find the best trigger match for a field"""
    best_match = None
    best_score = 0
    
    # Search through the document
    for start_idx in range(len(word_mapping)):
        # Skip if this position overlaps with used ranges
        if any(start <= start_idx <= end for start, end in used_ranges):
            continue
        
        # Evaluate match at this position
        match_result = evaluate_enhanced_trigger_sequence(
            word_mapping, triggers, start_idx, field_label
        )
        
        if match_result and match_result['score'] > best_score:
            best_score = match_result['score']
            best_match = match_result
    
    return best_match

def evaluate_enhanced_trigger_sequence(word_mapping: List[Dict], triggers: List[str], 
                                     start_idx: int, field_label: str) -> Optional[Dict]:
    """Evaluate trigger sequence with enhanced scoring"""
    if start_idx >= len(word_mapping):
        return None
    
    score = 0
    matched_words = []
    trigger_idx = 0
    current_idx = start_idx
    gap_count = 0
    max_scan_distance = min(len(triggers) * 5, 20)
    
    while trigger_idx < len(triggers) and current_idx < len(word_mapping):
        if current_idx - start_idx > max_scan_distance:
            break
        
        word_info = word_mapping[current_idx]
        current_trigger = triggers[trigger_idx].lower()
        word_norm = word_info['normalized']
        
        # Scoring logic
        if current_trigger == word_norm:
            # Exact match - highest score
            score += 40
            matched_words.append(word_info)
            trigger_idx += 1
            gap_count = 0
            
        elif len(word_norm) > 3 and current_trigger in word_norm:
            # Partial match within word
            score += 30
            matched_words.append(word_info)
            trigger_idx += 1
            gap_count = 0
            
        elif len(current_trigger) > 3 and current_trigger in word_norm:
            # Trigger contained in word
            score += 25
            matched_words.append(word_info)
            trigger_idx += 1
            gap_count = 0
            
        elif word_info['is_separator'] or word_info['is_connector']:
            # Skip separators and connectors without penalty
            pass
            
        elif len(word_norm) <= 2 and not word_info['has_numeric']:
            # Skip very short non-numeric words
            pass
            
        else:
            # Gap in matching
            gap_count += 1
            if gap_count > 3:  # Allow up to 3 gaps
                break
        
        current_idx += 1
    
    # Bonus scoring
    if trigger_idx == len(triggers):
        score += 25  # Complete match bonus
        if gap_count == 0:
            score += 15  # Consecutive match bonus
    
    # Calculate minimum required score
    min_score = len(triggers) * 20
    
    if score >= min_score and trigger_idx == len(triggers):
        return {
            'start_index': start_idx,
            'end_index': current_idx,
            'score': score,
            'matched_words': matched_words,
            'field_label': field_label,
            'gaps': gap_count,
            'completeness': trigger_idx / len(triggers)
        }
    
    return None

# ------------------ Enhanced Value Extraction ------------------

def extract_values_with_improved_separation(word_mapping: List[Dict], field_positions: Dict[str, Dict], 
                                          ner_entities: List[Dict]) -> Dict[str, str]:
    """Extract field values with improved separation logic"""
    results = {}
    
    # Create NER lookup for efficient access
    ner_lookup = create_ner_word_lookup(ner_entities, word_mapping)
    
    # Process each field
    for field_label in FIELD_PROCESSING_ORDER:
        if field_label in field_positions:
            position_info = field_positions[field_label]
            
            # Extract value using improved strategy
            extracted_value = extract_field_value_with_boundaries(
                word_mapping, position_info, ner_lookup, field_label, results
            )
            
            # Clean and validate
            cleaned_value = clean_field_value(extracted_value, field_label)
            results[field_label] = cleaned_value
            
            print(f"📝 {field_label}: '{cleaned_value}'")
    
    return results

def create_ner_word_lookup(ner_entities: List[Dict], word_mapping: List[Dict]) -> Dict[int, List[Dict]]:
    """Create efficient NER lookup by word index"""
    ner_lookup = defaultdict(list)
    
    for entity in ner_entities:
        entity_text = entity.get('cleaned_word', entity['word']).lower()
        
        # Find matching words in word mapping
        for i, word_info in enumerate(word_mapping):
            word_norm = word_info['normalized']
            word_orig = word_info['original'].lower()
            
            # Enhanced matching criteria
            if (entity_text == word_norm or
                entity_text == word_orig or
                (len(entity_text) > 3 and len(word_norm) > 3 and 
                 (entity_text in word_norm or word_norm in entity_text))):
                
                ner_lookup[i].append({
                    'entity_type': entity['entity_group'],
                    'confidence': entity.get('confidence_score', entity.get('score', 0)),
                    'text': entity['word'],
                    'is_bank_related': entity.get('is_bank_related', False),
                    'is_company_related': entity.get('is_company_related', False)
                })
    
    return ner_lookup

def extract_field_value_with_boundaries(word_mapping: List[Dict], position_info: Dict, 
                                       ner_lookup: Dict[int, List[Dict]], field_label: str, 
                                       existing_results: Dict[str, str]) -> str:
    """Extract field value with improved boundary detection"""
    start_idx = position_info['end_index']
    field_config = FIELD_BOUNDARIES.get(field_label, {
        "max_words": 10, "stop_words": []
    })
    
    # Skip initial separators and small words
    while (start_idx < len(word_mapping) and 
           (word_mapping[start_idx]['is_separator'] or 
            (word_mapping[start_idx]['length'] <= 2 and 
             not word_mapping[start_idx]['has_numeric'] and
             not word_mapping[start_idx]['is_date_like']))):
        start_idx += 1
    
    if start_idx >= len(word_mapping):
        return ""
    
    # Apply field-specific extraction with improved separation
    return extract_field_with_improved_boundaries(
        word_mapping, start_idx, field_config, field_label, ner_lookup
    )

def extract_field_with_improved_boundaries(word_mapping: List[Dict], start_idx: int, 
                                         field_config: Dict, field_label: str,
                                         ner_lookup: Dict[int, List[Dict]]) -> str:
    """Extract field with improved boundary detection"""
    max_words = field_config['max_words']
    stop_words = field_config.get('stop_words', [])
    
    value_parts = []
    
    # Field-specific extraction logic
    if field_label == "receipt_name":
        return extract_organization_name_improved(word_mapping, start_idx, ner_lookup, stop_words, max_words)
    elif field_label == "receipt_email":
        return extract_email_address_improved(word_mapping, start_idx, stop_words, max_words)
    elif field_label == "beneficiary_name":
        return extract_organization_name_improved(word_mapping, start_idx, ner_lookup, stop_words, max_words)
    elif field_label == "beneficiary_bank":
        return extract_bank_name_improved(word_mapping, start_idx, ner_lookup, stop_words, max_words)
    elif field_label == "remitter_name":
        return extract_organization_name_improved(word_mapping, start_idx, ner_lookup, stop_words, max_words)
    elif field_label == "remitting_bank":
        return extract_bank_name_improved(word_mapping, start_idx, ner_lookup, stop_words, max_words)
    elif field_label == "customer_reference":
        return extract_reference_improved(word_mapping, start_idx, stop_words, max_words)
    elif field_label in ["debit_amount", "remittance_amount"]:
        return extract_amount_improved(word_mapping, start_idx, stop_words, max_words)
    elif field_label == "transaction_type":
        return extract_transaction_type_improved(word_mapping, start_idx, stop_words, max_words)
    elif field_label == "sub_payment_type":
        return extract_sub_payment_type_improved(word_mapping, start_idx, stop_words, max_words)
    elif field_label == "account_number":
        return extract_account_number_improved(word_mapping, start_idx, stop_words, max_words)
    elif field_label in ["advice_date", "value_date"]:
        return extract_date_improved(word_mapping, start_idx, stop_words, max_words)
    elif field_label in ["advice_ref", "instruction_reference", "other_reference"]:
        return extract_reference_improved(word_mapping, start_idx, stop_words, max_words)
    elif field_label == "handling_fee_of_remitting_bank":
        return extract_fee_info_improved(word_mapping, start_idx, stop_words, max_words)
    elif field_label == "remitter_to_beneficiary_info":
        return extract_beneficiary_info_improved(word_mapping, start_idx, stop_words, max_words)
    else:
        return extract_generic_value_improved(word_mapping, start_idx, stop_words, max_words)

# ------------------ Improved Extraction Functions ------------------

def extract_organization_name_improved(word_mapping: List[Dict], start_idx: int, 
                                     ner_lookup: Dict[int, List[Dict]], stop_words: List[str], 
                                     max_words: int) -> str:
    """Extract organization name with improved boundary detection"""
    name_parts = []
    
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words to prevent field contamination
        if normalized in stop_words:
            break
        
        # Skip separators initially
        if word_info['is_separator'] and not name_parts:
            continue
        
        # Priority 1: NER organization entities
        if i in ner_lookup:
            org_entities = [e for e in ner_lookup[i] if e['entity_type'] == 'ORG']
            if org_entities and max(org_entities, key=lambda x: x['confidence'])['confidence'] > 0.5:
                name_parts.append(word)
                continue
        
        # Priority 2: All caps words (likely organization names)
        if word_info['is_all_caps'] and word_info['length'] > 1:
            name_parts.append(word)
            continue
        
        # Priority 3: Capitalized words
        if word_info['is_capitalized'] and word_info['has_alpha']:
            name_parts.append(word)
            continue
        
        # Priority 4: Company keywords
        if word_info['is_company_word']:
            name_parts.append(word)
            continue
        
        # Priority 5: Connectors if we have content
        if word_info['is_connector'] and name_parts:
            name_parts.append(word)
            continue
        
        # Stop if pattern breaks and we have content
        if name_parts and not (word_info['is_capitalized'] or word_info['is_all_caps'] or 
                              word_info['is_company_word'] or word_info['is_connector']):
            break
    
    return " ".join(name_parts).strip()

def extract_bank_name_improved(word_mapping: List[Dict], start_idx: int, 
                             ner_lookup: Dict[int, List[Dict]], stop_words: List[str], 
                             max_words: int) -> str:
    """Extract bank name with improved boundary detection"""
    bank_parts = []
    found_bank_keyword = False
    
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words
        if normalized in stop_words:
            break
        
        # Skip separators initially
        if word_info['is_separator'] and not bank_parts:
            continue
        
        # Priority 1: Bank-specific keywords
        if word_info['is_bank_word']:
            bank_parts.append(word)
            found_bank_keyword = True
            continue
        
        # Priority 2: NER organization entities with bank relation
        if i in ner_lookup:
            org_entities = [e for e in ner_lookup[i] if e['entity_type'] == 'ORG']
            if org_entities:
                best_entity = max(org_entities, key=lambda x: x['confidence'])
                if best_entity['confidence'] > 0.4:
                    bank_parts.append(word)
                    continue
        
        # Priority 3: All caps words (bank names often in caps)
        if word_info['is_all_caps'] and word_info['length'] > 1:
            bank_parts.append(word)
            continue
        
        # Priority 4: Capitalized words
        if word_info['is_capitalized'] and word_info['has_alpha']:
            bank_parts.append(word)
            continue
        
        # Priority 5: Company formation words
        if word_info['is_company_word']:
            bank_parts.append(word)
            continue
        
        # Priority 6: Connectors if we have content
        if word_info['is_connector'] and bank_parts:
            bank_parts.append(word)
            continue
        
        # Stop if pattern breaks and we have meaningful content
        if bank_parts and not (word_info['is_capitalized'] or word_info['is_all_caps'] or 
                              word_info['is_bank_word'] or word_info['is_company_word'] or
                              word_info['is_connector']):
            break
    
    return " ".join(bank_parts).strip()

def extract_email_address_improved(word_mapping: List[Dict], start_idx: int, 
                                 stop_words: List[str], max_words: int) -> str:
    """Extract email address with improved boundary detection"""
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words
        if normalized in stop_words:
            break
        
        # Look for email pattern
        if word_info['is_email_like']:
            if '@' in word and '.' in word and len(word) > 5:
                parts = word.split('@')
                if len(parts) == 2 and '.' in parts[1]:
                    return word
    
    return ""

def extract_reference_improved(word_mapping: List[Dict], start_idx: int, 
                             stop_words: List[str], max_words: int) -> str:
    """Extract reference with improved boundary detection"""
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words
        if normalized in stop_words:
            break
        
        # Strategy 1: Look for underscore patterns
        if word_info['is_reference_like'] and word_info['has_underscore']:
            return word
        
        # Strategy 2: Look for long alphanumeric sequences
        if (word_info['has_alpha'] and word_info['has_numeric'] and 
            word_info['length'] > 6 and not word_info['is_email_like']):
            return word
    
    return ""

def extract_amount_improved(word_mapping: List[Dict], start_idx: int, 
                          stop_words: List[str], max_words: int) -> str:
    """Extract amount with improved boundary detection"""
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words
        if normalized in stop_words:
            break
        
        # Strategy 1: Currency code followed by amount
        if word_info['is_currency_code'] and i + 1 < len(word_mapping):
            next_word_info = word_mapping[i + 1]
            next_word = next_word_info['original']
            
            if (next_word_info['is_amount_like'] or next_word_info['is_large_number'] or
                next_word_info['has_comma']):
                return f"{word.upper()}{next_word}"
        
        # Strategy 2: Amount with currency embedded
        if ('INR' in word or 'USD' in word) and word_info['has_numeric']:
            return word
        
        # Strategy 3: Large number that could be amount
        if (word_info['is_large_number'] or 
            (word_info['has_comma'] and word_info['has_numeric'])):
            return word
    
    return ""

def extract_transaction_type_improved(word_mapping: List[Dict], start_idx: int, 
                                    stop_words: List[str], max_words: int) -> str:
    """Extract transaction type with improved boundary detection"""
    type_parts = []
    
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words
        if normalized in stop_words:
            break
        
        # Skip separators initially
        if word_info['is_separator'] and not type_parts:
            continue
        
        # Collect meaningful words
        if word_info['has_alpha'] and word_info['length'] > 1:
            type_parts.append(word)
        elif type_parts and word_info['is_separator']:
            continue
        elif type_parts:
            break
    
    return " ".join(type_parts).strip()

def extract_sub_payment_type_improved(word_mapping: List[Dict], start_idx: int, 
                                    stop_words: List[str], max_words: int) -> str:
    """Extract sub payment type with improved boundary detection"""
    return extract_transaction_type_improved(word_mapping, start_idx, stop_words, max_words)

def extract_account_number_improved(word_mapping: List[Dict], start_idx: int, 
                                  stop_words: List[str], max_words: int) -> str:
    """Extract account number with improved boundary detection"""
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words
        if normalized in stop_words:
            break
        
        # Look for account patterns
        if (word_info['is_account_like'] or word_info['has_asterisk'] or
            (word_info['has_numeric'] and word_info['length'] > 8)):
            
            account_parts = [word]
            
            # Look for additional information in parentheses
            j = i + 1
            while j < min(i + 5, len(word_mapping)):
                next_word_info = word_mapping[j]
                next_word = next_word_info['original']
                
                if '(' in next_word or 'Part' in next_word or 'security' in next_word.lower():
                    account_parts.append(next_word)
                    # Continue until closing parenthesis or stop word
                    while j < len(word_mapping) and ')' not in word_mapping[j]['original']:
                        j += 1
                        if j < len(word_mapping) and word_mapping[j]['normalized'] not in stop_words:
                            account_parts.append(word_mapping[j]['original'])
                        else:
                            break
                    break
                j += 1
            
            return " ".join(account_parts)
    
    return ""

def extract_date_improved(word_mapping: List[Dict], start_idx: int, 
                        stop_words: List[str], max_words: int) -> str:
    """Extract date with improved pattern recognition and year inclusion"""
    date_parts = []
    found_month = False
    found_day = False
    found_year = False
    
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words
        if normalized in stop_words:
            break
        
        # Skip separators initially
        if word_info['is_separator'] and not date_parts:
            continue
        
        # Month names (priority)
        if word_info['is_month']:
            date_parts.append(word)
            found_month = True
            continue
        
        # Numbers that could be day/year
        if word_info['is_numeric_only']:
            num_value = int(word) if word.isdigit() else 0
            
            # Day (1-31)
            if 1 <= num_value <= 31 and not found_day:
                date_parts.append(word)
                found_day = True
                continue
            
            # Year (likely 4 digits or 2 digits > 50)
            elif ((len(word) == 4 and 1900 <= num_value <= 2100) or 
                  (len(word) == 2 and num_value >= 50)) and not found_year:
                date_parts.append(word)
                found_year = True
                continue
            
            # If we already have month and day, this could be year
            elif found_month and found_day and not found_year:
                date_parts.append(word)
                found_year = True
                continue
        
        # Short alphabetic (abbreviated months)
        elif (word_info['has_alpha'] and word_info['length'] <= 3 and
              normalized in ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                           'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
            date_parts.append(word)
            found_month = True
            continue
        
        # Stop if we have reasonable date components
        elif date_parts and len(date_parts) >= 2:
            # Continue looking for year if we don't have it
            if not found_year and word_info['is_numeric_only']:
                continue
            else:
                break
        
        # Stop if pattern breaks
        elif date_parts:
            break
    
    # Post-process to ensure we have a complete date
    result = " ".join(date_parts).strip()
    
    # If we're missing year, try to find it nearby
    if result and not found_year:
        extended_result = find_missing_year(word_mapping, start_idx, max_words + 5, stop_words)
        if extended_result and len(extended_result) > len(result):
            result = extended_result
    
    return result

def find_missing_year(word_mapping: List[Dict], start_idx: int, max_words: int, stop_words: List[str]) -> str:
    """Find missing year component in nearby text"""
    date_parts = []
    
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words
        if normalized in stop_words:
            break
        
        # Collect date-like components
        if (word_info['is_month'] or 
            (word_info['is_numeric_only'] and len(word) <= 4) or
            (word_info['has_alpha'] and word_info['length'] <= 3 and normalized in MONTH_NAMES)):
            date_parts.append(word)
        elif date_parts and word_info['is_separator']:
            continue
        elif date_parts and len(date_parts) >= 3:
            break
    
    return " ".join(date_parts).strip()

def extract_fee_info_improved(word_mapping: List[Dict], start_idx: int, 
                            stop_words: List[str], max_words: int) -> str:
    """Extract fee information with improved boundary detection"""
    fee_parts = []
    fee_keywords = ['collect', 'from', 'remitter', 'charge', 'fee', 'paid', 'by', 'beneficiary']
    
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words
        if normalized in stop_words:
            break
        
        # Skip separators initially
        if word_info['is_separator'] and not fee_parts:
            continue
        
        if (normalized in fee_keywords or 
            (fee_parts and word_info['has_alpha'] and word_info['length'] > 2)):
            fee_parts.append(word)
        elif fee_parts and len(fee_parts) >= 4:
            break
    
    return " ".join(fee_parts).strip()

def extract_beneficiary_info_improved(word_mapping: List[Dict], start_idx: int, 
                                    stop_words: List[str], max_words: int) -> str:
    """Extract beneficiary information with improved boundary detection"""
    info_parts = []
    template_words = ['dear', 'sir', 'madam', 'computer', 'generated', 'email', 
                     'message', 'signature', 'system', 'reply', 'document']
    
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words or template words
        if normalized in stop_words or normalized in template_words:
            break
        
        # Skip separators initially
        if word_info['is_separator'] and not info_parts:
            continue
        
        # Include meaningful content
        if (word_info['has_alpha'] and word_info['length'] > 2 and
            normalized not in ['bl', 'this', 'is', 'a', 'the']):
            info_parts.append(word)
        elif word_info['is_reference_like']:
            info_parts.append(word)
        elif info_parts and len(info_parts) >= 20:
            break
    
    return " ".join(info_parts).strip()

def extract_generic_value_improved(word_mapping: List[Dict], start_idx: int, 
                                 stop_words: List[str], max_words: int) -> str:
    """Generic value extraction with improved boundary detection"""
    value_parts = []
    
    for i in range(start_idx, min(start_idx + max_words, len(word_mapping))):
        word_info = word_mapping[i]
        word = word_info['original']
        normalized = word_info['normalized']
        
        # Check for stop words
        if normalized in stop_words:
            break
        
        # Skip separators initially
        if word_info['is_separator'] and not value_parts:
            continue
        
        if word_info['has_alpha'] and word_info['length'] > 1:
            value_parts.append(word)
        elif value_parts and word_info['is_separator']:
            continue
        elif value_parts:
            break
    
    return " ".join(value_parts).strip()

# ------------------ Enhanced Cleaning and Validation ------------------

def clean_field_value(value: str, field_label: str) -> str:
    """Enhanced field value cleaning and validation"""
    if not value or not value.strip():
        return ""
    
    # Basic cleanup
    value = " ".join(value.split())
    
    # Field-specific cleaning
    if field_label in ["receipt_name", "beneficiary_name", "remitter_name"]:
        return clean_organization_name(value)
    elif field_label in ["beneficiary_bank", "remitting_bank"]:
        return clean_bank_name(value)
    elif field_label == "receipt_email":
        return clean_email(value)
    elif field_label in ["debit_amount", "remittance_amount"]:
        return clean_amount(value)
    elif field_label == "customer_reference":
        return clean_reference(value)
    elif field_label == "remitter_to_beneficiary_info":
        return clean_beneficiary_info(value)
    elif field_label in ["advice_date", "value_date"]:
        return clean_date(value)
    
    return value.strip()

def clean_organization_name(value: str) -> str:
    """Clean organization name"""
    if len(value) < 2:
        return ""
    
    # Remove field indicators that might have leaked in
    contaminating_words = ["Beneficiary", "Bank", "Account", "Type", "Name", "Email", "Transaction"]
    words = value.split()
    cleaned_words = []
    
    for word in words:
        if word not in contaminating_words:
            cleaned_words.append(word)
    
    value = " ".join(cleaned_words)
    
    # Remove leading/trailing separators
    value = value.strip(":-()[]{}.,;")
    
    return value.strip()

def clean_bank_name(value: str) -> str:
    """Clean bank name"""
    if len(value) < 3:
        return ""
    
    # Remove common false inclusions
    contaminating_words = ["Beneficiary", "Account", "Customer", "Reference", "Value", "Fee", "Handling", "Date"]
    words = value.split()
    cleaned_words = []
    
    for word in words:
        if word not in contaminating_words:
            cleaned_words.append(word)
    
    value = " ".join(cleaned_words)
    
    # Ensure minimum meaningful content
    if len(value) < 3:
        return ""
    
    return value.strip()

def clean_email(value: str) -> str:
    """Clean and validate email"""
    if '@' not in value or '.' not in value or len(value) < 5:
        return ""
    
    # Basic email validation
    parts = value.split('@')
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return ""
    
    if '.' not in parts[1]:
        return ""
    
    return value.strip()

def clean_amount(value: str) -> str:
    """Clean amount value"""
    if not any(c.isdigit() for c in value):
        return ""
    
    # Remove contaminating words
    contaminating_words = ["Remittance", "Amount", "Handling", "Fee", "Debit"]
    for word in contaminating_words:
        value = value.replace(word, "").replace(word.lower(), "")
    
    # Standardize currency codes
    value = value.replace('inr', 'INR').replace('usd', 'USD')
    
    # Clean up extra spaces
    value = " ".join(value.split())
    
    return value.strip()

def clean_reference(value: str) -> str:
    """Clean reference value"""
    if len(value) < 3:
        return ""
    
    # Remove contaminating words
    contaminating_words = ["Debit", "Amount", "INR", "USD", "Remittance", "Transaction", "Customer", "Reference"]
    for word in contaminating_words:
        value = value.replace(word, "").replace(word.lower(), "")
    
    value = " ".join(value.split())
    
    if len(value) < 3:
        return ""
    
    return value.strip()

def clean_beneficiary_info(value: str) -> str:
    """Clean beneficiary information"""
    # Remove template text
    template_phrases = [
        "Dear Sir/Madam", "This is a Computer generated E-Mail Message",
        "This does not require a Signature", "System generated",
        "Please do not reply", "Document Number Amount Date"
    ]
    
    for phrase in template_phrases:
        value = value.replace(phrase, "")
    
    # Remove BL// prefix
    if value.startswith("BL//"):
        value = value[4:]
    
    value = " ".join(value.split())
    return value.strip()

def clean_date(value: str) -> str:
    """Clean date value"""
    if not value:
        return ""
    
    # Remove contaminating words
    contaminating_words = ["Value", "Date", "Advice", "Sending"]
    words = value.split()
    cleaned_words = []
    
    for word in words:
        if word not in contaminating_words:
            cleaned_words.append(word)
    
    return " ".join(cleaned_words).strip()

# ------------------ Main Processing Function ------------------

def process_pdf_with_enhanced_extraction(file_path: str) -> Dict[str, Any]:
    """Main processing function with enhanced field extraction"""
    print(f"📄 Processing PDF: {file_path}")
    print("🔧 Enhanced Features: Improved Separation + Complete Date Extraction")
    
    try:
        # Initialize NER extractor
        ner_extractor = EnhancedRoBERTaNERExtractor()
        
        # Extract and process text
        text = extract_text_from_pdf(file_path)
        if not text:
            print("❌ Failed to extract text from PDF")
            return {"extracted_entities": {}, "error": "Failed to extract text"}
        
        print(f"📝 Extracted text length: {len(text)} characters")
        
        # Create comprehensive word mapping
        word_mapping = create_comprehensive_word_mapping(text)
        print(f"📊 Total words analyzed: {len(word_mapping)}")
        
        # Extract NER entities
        ner_entities = ner_extractor.extract_entities(text) if ner_extractor.ner_pipeline else []
        print(f"🤖 NER entities found: {len(ner_entities)}")
        
        # Find field positions
        field_positions = find_enhanced_field_positions(word_mapping)
        print(f"🎯 Fields detected by triggers: {len(field_positions)}")
        
        # Extract values with improved separation
        extracted_values = extract_values_with_improved_separation(word_mapping, field_positions, ner_entities)
        
        # Post-processing fixes
        extracted_values = apply_post_processing_fixes(extracted_values, word_mapping)
        
        return {
            "extracted_entities": extracted_values,
            "ner_entities_count": len(ner_entities),
            "fields_detected": len(field_positions),
            "text_length": len(text),
            "word_count": len(word_mapping)
        }
        
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        return {"extracted_entities": {}, "error": str(e)}

def apply_post_processing_fixes(extracted_values: Dict[str, str], 
                               word_mapping: List[Dict]) -> Dict[str, str]:
    """Apply post-processing fixes for common issues"""
    print("🔧 Applying post-processing fixes...")
    
    # Fix 1: Extract customer reference from beneficiary info if missing
    if not extracted_values.get("customer_reference") and extracted_values.get("remitter_to_beneficiary_info"):
        beneficiary_info = extracted_values["remitter_to_beneficiary_info"]
        parts = beneficiary_info.split()
        for part in parts:
            if ('_' in part and any(c.isdigit() for c in part) and len(part) > 6):
                extracted_values["customer_reference"] = part
                print(f"   ✅ Found customer reference in beneficiary info: {part}")
                break
    
    # Fix 2: Set debit amount equal to remittance amount if missing
    if not extracted_values.get("debit_amount") and extracted_values.get("remittance_amount"):
        extracted_values["debit_amount"] = extracted_values["remittance_amount"]
        print(f"   ✅ Set debit amount equal to remittance amount")
    
    # Fix 3: Extended search for missing critical fields
    critical_fields = ["receipt_email", "customer_reference"]
    for field in critical_fields:
        if not extracted_values.get(field):
            found_value = perform_extended_search(field, word_mapping)
            if found_value:
                extracted_values[field] = found_value
                print(f"   ✅ Found {field} through extended search: {found_value}")
    
    return extracted_values

def perform_extended_search(field_label: str, word_mapping: List[Dict]) -> str:
    """Perform extended search for missing critical fields"""
    if field_label == "receipt_email":
        # Search entire document for email patterns
        for word_info in word_mapping:
            if word_info['is_email_like']:
                word = word_info['original']
                if '@' in word and '.' in word and len(word) > 5:
                    return word
    
    elif field_label == "customer_reference":
        # Search for reference patterns throughout document
        for word_info in word_mapping:
            word = word_info['original']
            if (word_info['has_underscore'] and word_info['has_numeric'] and 
                len(word) > 6 and len(word) < 25):
                # Exclude common false positives
                if not any(exclude in word.lower() for exclude in 
                          ['advice', 'instruction', 'other', 'email', 'amount']):
                    return word
    
    return ""

# ------------------ Main Execution ------------------

def main():
    """Main execution function"""
    file_path = r"C:\Users\SyedaZuberiya\Desktop\PyMuPDF Testing\Payment_Advice_F2.pdf"  # Update this path as needed
    
    try:
        print("🚀 Starting Enhanced PDF Field Extraction...")
        print("🎯 Target: Clean field separation with complete date extraction")
        print("🔧 Technology: Improved Boundaries + NER + Smart Indexing")
        print("="*70)
        
        # Process the PDF
        result = process_pdf_with_enhanced_extraction(file_path)
        
        if "error" in result:
            print(f"❌ Processing failed: {result['error']}")
            return
        
        # Extract results
        final_output = result["extracted_entities"]
        
        # Print results
        print("\n" + "="*70)
        print("📦 ENHANCED EXTRACTION RESULTS:")
        print("="*70)
        
        print(json.dumps(final_output, indent=4, ensure_ascii=False))
        
        # Save results
        output_file = "enhanced_payment_advice_extraction.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Statistics
        print(f"\n📊 EXTRACTION STATISTICS:")
        print(f"   • Text processed: {result['text_length']:,} characters")
        print(f"   • Words analyzed: {result['word_count']:,}")
        print(f"   • NER entities: {result['ner_entities_count']}")
        print(f"   • Fields detected: {result['fields_detected']}")
        print(f"   • Non-empty extractions: {len([v for v in final_output.values() if v])}")
        
        # Critical field analysis
        critical_fields = ["receipt_name", "receipt_email", "beneficiary_name", 
                          "beneficiary_bank", "remitter_name", "remitting_bank", 
                          "customer_reference", "debit_amount", "advice_date", "value_date"]
        
        print(f"\n🎯 CRITICAL FIELDS STATUS:")
        success_count = 0
        
        for field in critical_fields:
            value = final_output.get(field, "")
            if value:
                status = "✅ EXTRACTED"
                print(f"   • {field:25}: {status} - '{value}'")
                success_count += 1
            else:
                status = "❌ MISSING"
                print(f"   • {field:25}: {status}")
        
        # Success rate
        success_rate = (success_count / len(critical_fields)) * 100
        print(f"\n🏆 SUCCESS RATE: {success_rate:.1f}% ({success_count}/{len(critical_fields)})")
        
        # Overall completion
        total_completion = (len([v for v in final_output.values() if v]) / len(final_output)) * 100
        print(f"🎯 OVERALL COMPLETION: {total_completion:.1f}%")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        print("   Please update the file_path variable with the correct path to your PDF.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()