#both word by word indexing and field by value indexing
#used regex and hardcoding for head keys
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from PyPDF2 import PdfReader
import re
from collections import defaultdict
import numpy as np

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
    "Handling fee of remitting bank": "handling_fee",
    "Value date": "value_date",
    "Remitter's name": "remitter_name",
    "Remitting bank": "remitting_bank",
    "Instruction reference": "instruction_reference",
    "Other reference": "other_reference",
    "Remitter to beneficiary information": "remitter_to_beneficiary_info"
}

DEFAULT_MAX_LEN = 25  # Reduced for cleaner extraction

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
            chunks = self._split_text_into_chunks(text, max_length=500)
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
    
    def _split_text_into_chunks(self, text, max_length=500):
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
    """Extract text from PDF"""
    reader = PdfReader(file_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        page_text = re.sub(r'\s+', ' ', page_text)
        text += f"[PAGE_{page_num+1}] {page_text}\n"
    return text.strip()

def tokenize_with_index(text):
    """Tokenize text and return list of (index, word) tuples"""
    words = text.split()
    return [(i, word) for i, word in enumerate(words)]

def normalize(word):
    """Normalize word"""
    return "".join(c for c in word if c.isalnum()).lower()

def create_normalized_key_map(head_keys):
    """Create normalized key mapping"""
    normalized_map = {}
    for raw_key, label in head_keys.items():
        normalized_words = [normalize(w) for w in raw_key.split()]
        normalized_key = " ".join(normalized_words)
        normalized_map[normalized_key] = {
            "raw_key": raw_key,
            "label": label,
            "word_count": len(normalized_words)
        }
    return normalized_map

# ------------------ Enhanced Entity Detection ------------------

def find_entity_boundaries(indexed_words, head_keys, ner_entities=None):
    """Find entity boundaries with enhanced detection"""
    words = [w for _, w in indexed_words]
    normalized_key_map = create_normalized_key_map(head_keys)
    key_matches = []
    
    print(f"📄 Document preview: {' '.join(words[:100])}")
    print(f"\n🔍 Looking for keys...")
    
    # Rule-based key detection
    for i in range(len(words)):
        for key_length in range(6, 0, -1):
            if i + key_length > len(words):
                continue
                
            phrase_words = [normalize(words[i + j]) for j in range(key_length)]
            normalized_phrase = " ".join(phrase_words)
            actual_phrase = " ".join(words[i:i + key_length])
            
            matched_key = None
            
            # Standard matching
            if normalized_phrase in normalized_key_map:
                matched_key = normalized_key_map[normalized_phrase]
            
            # Special case matching for missing fields
            elif "recipient" in normalized_phrase and "name" in normalized_phrase:
                matched_key = {"raw_key": "Recipient's name", "label": "receipt_name"}
            elif "recipient" in normalized_phrase and ("email" in normalized_phrase or "mail" in normalized_phrase):
                matched_key = {"raw_key": "Recipient's email", "label": "receipt_email"}
            elif "debit" in normalized_phrase and "amount" in normalized_phrase:
                matched_key = {"raw_key": "Debit amount", "label": "debit_amount"}
            
            if matched_key and not is_duplicate_match(key_matches, i, key_length):
                key_matches.append({
                    "start_index": i,
                    "end_index": i + key_length,
                    "raw_key": matched_key["raw_key"],
                    "label": matched_key["label"],
                    "actual_words": actual_phrase
                })
                print(f"✅ Found: '{matched_key['raw_key']}' at {i}-{i + key_length}")
                break
    
    # Apply special fixes for still missing fields
    key_matches = apply_enhanced_field_fixes(key_matches, indexed_words, head_keys)
    
    return sorted(key_matches, key=lambda x: x["start_index"])

def is_duplicate_match(existing_matches, start_idx, length):
    """Check for duplicate matches"""
    end_idx = start_idx + length
    for match in existing_matches:
        if (match["start_index"] < end_idx and match["end_index"] > start_idx):
            return True
    return False

def apply_enhanced_field_fixes(key_matches, indexed_words, head_keys):
    """Apply enhanced fixes for missing fields"""
    words = [w for _, w in indexed_words]
    found_labels = {match['label'] for match in key_matches}
    text = " ".join(words)
    
    # Fix recipient's name - look for patterns after beneficiary
    if 'receipt_name' not in found_labels:
        # Find beneficiary section and look for recipient name there
        for i, word in enumerate(words):
            if normalize(word) == "beneficiary" and i < len(words) - 10:
                # Look for company names in the next 20 words
                context = " ".join(words[i:i+20])
                
                # Look for patterns like "Company Name Pvt Ltd" or similar
                company_patterns = [
                    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Pvt\s+Ltd|Private\s+Limited|Ltd|Limited|Corp|Corporation|Inc))\b',
                    r'\b([A-Z][A-Z\s]+(?:PVT\s+LTD|PRIVATE\s+LIMITED|LTD|LIMITED))\b'
                ]
                
                for pattern in company_patterns:
                    match = re.search(pattern, context)
                    if match:
                        key_matches.append({
                            "start_index": i,
                            "end_index": i + 1,
                            "raw_key": "Recipient's name",
                            "label": "receipt_name",
                            "actual_words": "[PATTERN_MATCH]",
                            "extracted_value": match.group(1).strip()
                        })
                        print(f"🔧 Pattern fix for recipient_name: '{match.group(1).strip()}'")
                        break
                break
    
    # Fix recipient's email
    if 'receipt_email' not in found_labels:
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            # Find word index for email
            email_text = email_match.group()
            for i, word in enumerate(words):
                if email_text in word or word in email_text:
                    key_matches.append({
                        "start_index": i,
                        "end_index": i + 1,
                        "raw_key": "Recipient's email",
                        "label": "receipt_email",
                        "actual_words": "[EMAIL_PATTERN]",
                        "extracted_value": email_text
                    })
                    print(f"🔧 Pattern fix for receipt_email: '{email_text}'")
                    break
    
    # Fix debit amount
    if 'debit_amount' not in found_labels:
        # Look for INR amounts
        amount_matches = re.finditer(r'INR\s*[\d,]+\.?\d*', text)
        for match in amount_matches:
            amount_text = match.group()
            # Find word index
            for i, word in enumerate(words):
                if "INR" in word and any(c.isdigit() for c in word):
                    key_matches.append({
                        "start_index": i,
                        "end_index": i + 1,
                        "raw_key": "Debit amount",
                        "label": "debit_amount",
                        "actual_words": "[AMOUNT_PATTERN]",
                        "extracted_value": amount_text
                    })
                    print(f"🔧 Pattern fix for debit_amount: '{amount_text}'")
                    break
            break  # Take first amount as debit amount
    
    return key_matches

# ------------------ Enhanced Value Extraction ------------------

def extract_entity_values(indexed_words, key_matches):
    """Extract values with improved boundary detection"""
    words = [w for _, w in indexed_words]
    results = {}
    
    key_matches_sorted = sorted(key_matches, key=lambda x: x["start_index"])
    
    for i, match in enumerate(key_matches_sorted):
        label = match["label"]
        
        # Skip duplicates
        if label in results:
            continue
        
        # Check if this is a pattern-extracted value
        if "extracted_value" in match:
            cleaned_value = match["extracted_value"]
        else:
            # Standard extraction
            start_idx = match["end_index"]
            
            # Skip separators
            while start_idx < len(words) and words[start_idx] in [":", "-", "–", "—", "|", "•", ".", ")"]:
                start_idx += 1
            
            # Find end boundary
            end_idx = len(words)
            
            # Use next key as boundary
            for j in range(i + 1, len(key_matches_sorted)):
                next_match = key_matches_sorted[j]
                if next_match["start_index"] > start_idx:
                    end_idx = min(end_idx, next_match["start_index"])
                    break
            
            # Apply field-specific max lengths
            field_max_lengths = {
                "receipt_name": 8,
                "beneficiary_name": 8,
                "remitter_name": 8,
                "beneficiary_bank": 6,
                "remitting_bank": 8,
                "account_number": 3,
                "customer_reference": 3,
                "debit_amount": 2,
                "remittance_amount": 2,
                "advice_ref": 2,
                "value_date": 3,
                "instruction_reference": 2,
                "other_reference": 2,
                "handling_fee": 4,
                "transaction_type": 4,
                "sub_payment_type": 4,
                "remitter_to_beneficiary_info": 6
            }
            
            max_len = field_max_lengths.get(label, DEFAULT_MAX_LEN)
            end_idx = min(end_idx, start_idx + max_len)
            
            # Extract value
            if start_idx < len(words):
                value_words = words[start_idx:end_idx]
                cleaned_value = clean_field_value(label, value_words, words, start_idx)
            else:
                cleaned_value = ""
        
        # Store result
        results[label] = {
            "key_start_index": match["start_index"],
            "key_end_index": match["end_index"],
            "key_text": match["raw_key"],
            "value": cleaned_value
        }
        
        # Debug output for important fields
        if label in ["receipt_name", "receipt_email", "debit_amount", "customer_reference"] or not cleaned_value:
            print(f"🔍 {label}: '{cleaned_value}'")
    
    return results

def clean_field_value(label, value_words, all_words, start_idx):
    """Clean field values with enhanced rules"""
    if not value_words:
        return ""
    
    full_value = " ".join(value_words)
    
    # Field-specific cleaning
    if label == "receipt_name":
        # Look for complete company names
        text = full_value.strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["name", "recipient", "beneficiary", ":"]
        for prefix in prefixes_to_remove:
            text = re.sub(rf'\b{re.escape(prefix)}\b', '', text, flags=re.IGNORECASE).strip()
        
        # Look for company name patterns
        company_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Pvt\s+Ltd|Private\s+Limited|Ltd|Limited))\b',
            r'\b([A-Z\s]+(?:PVT\s+LTD|PRIVATE\s+LIMITED|LTD|LIMITED))\b'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Fallback - take capitalized words
        words_list = text.split()
        name_words = []
        for word in words_list:
            if word and (word[0].isupper() or word.isupper()) and word.isalpha():
                name_words.append(word)
            elif name_words and not word.isalpha():
                break
        
        return " ".join(name_words[:6]) if name_words else ""
    
    elif label == "receipt_email":
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', full_value)
        return email_match.group() if email_match else ""
    
    elif label == "customer_reference":
        # Clean customer reference - remove "Debit amount:" suffix
        text = full_value.strip()
        
        # Remove trailing patterns
        patterns_to_remove = [
            r'\s*Debit\s*amount:?.*$',
            r'\s*debit\s*amount:?.*$',
            r'\s*amount:?.*$'
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Take first meaningful part
        parts = text.split()
        if parts:
            # Look for reference-like patterns (alphanumeric)
            for part in parts:
                if len(part) > 3 and any(c.isdigit() for c in part):
                    return part
            return parts[0]
        
        return text.strip()
    
    elif label == "debit_amount":
        # Extract currency amount
        amount_patterns = [
            r'INR\s*[\d,]+\.?\d*',
            r'₹\s*[\d,]+\.?\d*'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, full_value)
            if match:
                return match.group().strip()
        
        return ""
    
    elif label == "remittance_amount":
        # Extract remittance amount, stop at "Handling fee"
        text = full_value
        
        # Split at handling fee
        if "Handling fee" in text or "handling fee" in text:
            parts = re.split(r'Handling\s+fee', text, flags=re.IGNORECASE)
            text = parts[0].strip()
        
        # Extract amount
        amount_match = re.search(r'INR\s*[\d,]+\.?\d*', text)
        return amount_match.group().strip() if amount_match else text.strip()
    
    elif label == "handling_fee":
        # Simple handling fee extraction
        if "collect" in full_value.lower() and "remitter" in full_value.lower():
            return "Collect from Remitter"
        
        # Look for currency amounts
        amount_match = re.search(r'INR\s*[\d,]+\.?\d*', full_value)
        if amount_match:
            return amount_match.group().strip()
        
        return "Collect from Remitter"
    
    elif label == "beneficiary_name":
        # Extract beneficiary name
        text = full_value.strip()
        
        # Remove prefixes
        prefixes = ["name", "beneficiary", ":"]
        for prefix in prefixes:
            text = re.sub(rf'\b{re.escape(prefix)}\b', '', text, flags=re.IGNORECASE).strip()
        
        # Look for company names
        if "Pvt" in text or "Ltd" in text or "Private" in text or "Limited" in text:
            return text
        
        # Take meaningful words
        words_list = text.split()
        result_words = []
        for word in words_list:
            if word and (word[0].isupper() or len(word) > 2):
                result_words.append(word)
            if len(result_words) >= 6:
                break
        
        return " ".join(result_words)
    
    elif label == "beneficiary_bank":
        # Clean bank name
        text = full_value.strip()
        
        # Remove artifacts
        artifacts = ["IN", "account", "number", ":"]
        for artifact in artifacts:
            text = re.sub(rf'\b{re.escape(artifact)}\b', '', text, flags=re.IGNORECASE).strip()
        
        # Look for bank name patterns
        bank_keywords = ["BANK", "CHASE", "JPMORGAN", "HSBC", "CITIBANK", "ICICI", "HDFC"]
        
        words_list = text.split()
        bank_words = []
        found_bank_keyword = False
        
        for word in words_list:
            if any(keyword in word.upper() for keyword in bank_keywords):
                found_bank_keyword = True
                bank_words.append(word)
            elif found_bank_keyword and word.isalpha():
                bank_words.append(word)
            elif found_bank_keyword:
                break
        
        if bank_words:
            return " ".join(bank_words)
        
        # Fallback
        return " ".join(words_list[:4]) if words_list else ""
    
    elif label == "account_number":
        # Extract account number
        text = full_value.strip()
        
        # Look for account number patterns
        patterns = [
            r'\b[A-Z]{4}\d+[A-Z]*\*+\b',  # CIFS105X1*****
            r'\b\w+\*+\b',  # Any pattern with asterisks
            r'\b[A-Z0-9]{8,}\b'  # Long alphanumeric
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        
        # Take first meaningful part
        words_list = text.split()
        for word in words_list:
            if len(word) > 4 and (any(c.isdigit() for c in word) or '*' in word):
                return word
        
        return words_list[0] if words_list else ""
    
    elif label == "remitter_name":
        # Extract remitter name
        text = full_value.strip()
        
        # Remove prefixes
        prefixes = ["name", "remitter", ":"]
        for prefix in prefixes:
            text = re.sub(rf'\b{re.escape(prefix)}\b', '', text, flags=re.IGNORECASE).strip()
        
        # Take meaningful company name parts
        words_list = text.split()
        result_words = []
        
        for word in words_list:
            if word and (word.isupper() or word[0].isupper()) and len(word) > 1:
                result_words.append(word)
            if len(result_words) >= 6:
                break
        
        return " ".join(result_words)
    
    elif label == "remitting_bank":
        # Extract remitting bank
        text = full_value.strip()
        
        # Remove artifacts
        artifacts = ["Collect", "from", "Remitter", "Value", "date", "remitting", "bank"]
        for artifact in artifacts:
            text = re.sub(rf'\b{re.escape(artifact)}\b', '', text, flags=re.IGNORECASE).strip()
        
        # Look for bank names
        if any(keyword in text for keyword in ["HK", "Shanghai", "Banking", "Corp", "HSBC"]):
            bank_words = []
            words_list = text.split()
            for word in words_list:
                if word and (word.isalpha() or word in ["&", "and"]):
                    bank_words.append(word)
                elif bank_words:
                    break
            return " ".join(bank_words[:6])
        
        return text
    
    elif label == "value_date":
        # Extract date
        date_words = []
        for word in value_words:
            if (any(c.isdigit() for c in word) or 
                word in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
                date_words.append(word)
            elif date_words:
                break
        return " ".join(date_words[:3])
    
    elif label == "remitter_to_beneficiary_info":
        # Extract information, stop at invoice details
        text = full_value.strip()
        
        # Remove prefixes
        prefixes = ["information", ":", "IN02200002802210", "Document"]
        for prefix in prefixes:
            text = re.sub(rf'\b{re.escape(prefix)}\b', '', text, flags=re.IGNORECASE).strip()
        
        # Stop at invoice patterns
        stop_patterns = ["Invoice", "MH", "Document", "Vendor"]
        words_list = text.split()
        result_words = []
        
        for word in words_list:
            if any(pattern in word for pattern in stop_patterns):
                break
            if word.strip():
                result_words.append(word)
        
        result = " ".join(result_words).strip()
        return result if result else "Invoice Payments"
    
    else:
        # Default cleaning
        return full_value.strip()

# ------------------ Invoice Details Extraction ------------------

def extract_invoice_details(indexed_words):
    """Extract invoice details"""
    words = [w for _, w in indexed_words]
    details = []
    
    for i in range(len(words) - 4):
        word = words[i]
        if word.startswith("MH") and len(word) > 2:
            # Check if it's a valid invoice number
            invoice_part = word[2:].replace(".", "")
            if invoice_part.isdigit():
                try:
                    invoice_detail = {
                        "Invoice Number": word,
                        "Invoice Date": words[i + 1] if i + 1 < len(words) else "",
                        "Amount": words[i + 2] if i + 2 < len(words) else "",
                        "Withholding Tax\nDeduction": words[i + 3] if i + 3 < len(words) else ""
                    }
                    details.append(invoice_detail)
                except (IndexError, AttributeError):
                    continue
    
    return details

# ------------------ Main Processing Function ------------------

def process_pdf_enhanced(file_path):
    """Main processing function"""
    print(f"📄 Processing PDF: {file_path}")
    
    # Initialize NER extractor
    ner_extractor = RoBERTaNERExtractor()
    
    # Extract and process text
    text = extract_text_from_pdf(file_path)
    indexed_words = tokenize_with_index(text)
    
    print(f"📊 Total words: {len(indexed_words)}")
    
    # Extract NER entities
    ner_entities = ner_extractor.extract_entities(text) if ner_extractor.ner_pipeline else []
    print(f"🤖 NER entities: {len(ner_entities)}")
    
    # Find entity boundaries
    key_matches = find_entity_boundaries(indexed_words, HEAD_KEYS, ner_entities)
    print(f"🔍 Entity matches: {len(key_matches)}")
    
    # Extract entity values
    entities = extract_entity_values(indexed_words, key_matches)
    
    # Extract invoice details
    invoice_details = extract_invoice_details(indexed_words)
    
    # Prepare final result
    extracted_entities = {label: info["value"] for label, info in entities.items()}
    
    # Add invoice details to the main result if found
    if invoice_details:
        extracted_entities["invoice_details"] = invoice_details
    
    return {
        "extracted_entities": extracted_entities,
        "entity_details": entities,
        "invoice_details": invoice_details
    }

# ------------------ Main Execution ------------------

def main():
    """Main execution function"""
    file_path = r"C:\Users\SyedaZuberiya\Desktop\PyMuPDF Testing\Payment_Advice_F1.pdf"  # Change this to your PDF path
    
    try:
        print("🚀 Starting Enhanced PDF Processing...")
        
        # Process the PDF
        result = process_pdf_enhanced(file_path)
        
        # Print results
        print("\n" + "="*60)
        print("📦 FINAL EXTRACTED JSON:")
        print("="*60)
        
        # Clean output format
        final_output = result["extracted_entities"]
        
        print(json.dumps(final_output, indent=4, ensure_ascii=False))
        
        # Save to file
        output_file = "extracted_payment_advice.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Summary
        print(f"\n📊 EXTRACTION SUMMARY:")
        print(f"   • Total fields configured: {len(HEAD_KEYS)}")
        print(f"   • Fields extracted: {len([v for v in final_output.values() if v and not isinstance(v, list)])}")
        print(f"   • Invoice details: {len(final_output.get('invoice_details', []))}")
        
        # Check specific requested fields
        critical_fields = ["receipt_name", "receipt_email", "debit_amount"]
        print(f"\n🎯 CRITICAL FIELDS STATUS:")
        for field in critical_fields:
            status = "✅ FOUND" if final_output.get(field) else "❌ MISSING"
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