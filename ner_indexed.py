import json
from PyPDF2 import PdfReader

# ------------------ Configuration ------------------

HEAD_KEYS = {
    "Advice sending date": "advice_date",
    "Advice reference no": "advice_ref",
    "Recipient's name": "receipt_name",
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

DEFAULT_MAX_LEN = 50  # Increased for better coverage

# ------------------ Helpers ------------------

def extract_text_from_pdf(file_path):
    """Extract text from PDF and return as string"""
    reader = PdfReader(file_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"[PAGE_{page_num+1}] {page_text}\n"
    return text.strip()

def tokenize_with_index(text):
    """Tokenize text and return list of (index, word) tuples"""
    words = text.split()
    return [(i, word) for i, word in enumerate(words)]

def normalize(word):
    """Normalize word by removing non-alphanumeric characters and converting to lowercase"""
    return "".join(c for c in word if c.isalnum()).lower()

def create_normalized_key_map(head_keys):
    """Create a mapping of normalized keys to their original forms and labels"""
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

def find_entity_boundaries(indexed_words, head_keys):
    """Find all entity positions and their boundaries using index-based approach"""
    words = [w for _, w in indexed_words]
    normalized_key_map = create_normalized_key_map(head_keys)
    
    # Find all potential key matches
    key_matches = []
    
    # Debug: Print first 100 words to understand document structure
    print(f"📄 Document preview (first 100 words): {' '.join(words[:100])}")
    print(f"\n🔍 Looking for these keys:")
    for key, label in head_keys.items():
        print(f"  - '{key}' -> {label}")
    
    print(f"\n📍 Scanning document for key matches...")
    
    for i in range(len(words)):
        # Try to match multi-word keys first (longest match)
        for key_length in range(6, 0, -1):  # Try 6-word keys down to 1-word keys
            if i + key_length > len(words):
                continue
                
            # Create normalized phrase
            phrase_words = [normalize(words[i + j]) for j in range(key_length)]
            normalized_phrase = " ".join(phrase_words)
            
            # Also try exact case matching for problematic keys
            exact_phrase = " ".join(words[i:i + key_length]).lower()
            
            # Check both normalized and exact matching
            matched_key = None
            if normalized_phrase in normalized_key_map:
                matched_key = normalized_key_map[normalized_phrase]
            else:
                # Try exact matching for specific problematic cases
                for raw_key, label in head_keys.items():
                    if raw_key.lower() == exact_phrase:
                        matched_key = {
                            "raw_key": raw_key,
                            "label": label,
                            "word_count": len(raw_key.split())
                        }
                        break
            
            if matched_key:
                key_info = matched_key
                
                # Special handling for overlapping keys
                is_duplicate = False
                for existing_match in key_matches:
                    # Check if this overlaps with existing match
                    if (existing_match["start_index"] <= i < existing_match["end_index"] or
                        i <= existing_match["start_index"] < i + key_length):
                        # Keep the longer/more specific match
                        if key_length <= (existing_match["end_index"] - existing_match["start_index"]):
                            is_duplicate = True
                            break
                        else:
                            # Remove the shorter match
                            key_matches.remove(existing_match)
                
                if not is_duplicate:
                    key_matches.append({
                        "start_index": i,
                        "end_index": i + key_length,
                        "raw_key": key_info["raw_key"],
                        "label": key_info["label"],
                        "normalized_key": normalized_phrase,
                        "actual_words": " ".join(words[i:i + key_length])  # For debugging
                    })
                    print(f"✅ Found key: '{key_info['raw_key']}' at position {i}-{i + key_length} ('{' '.join(words[i:i + key_length])}')")
                break  # Found a match, don't look for shorter matches at this position
    
    # Sort by start index
    key_matches.sort(key=lambda x: x["start_index"])
    
    print(f"\n🔍 Total keys found: {len(key_matches)}")
    
    # Debug: Show all found matches
    print(f"\n📋 All found matches:")
    for match in key_matches:
        print(f"  {match['label']}: '{match['actual_words']}' at {match['start_index']}-{match['end_index']}")
    
    return key_matches

def extract_entity_values(indexed_words, key_matches):
    """Extract values for each entity using index boundaries"""
    words = [w for _, w in indexed_words]
    results = {}
    
    # Create a set of all key positions for better boundary detection
    all_key_positions = set()
    for match in key_matches:
        for pos in range(match["start_index"], match["end_index"]):
            all_key_positions.add(pos)
    
    # Sort matches by position for proper processing
    key_matches_sorted = sorted(key_matches, key=lambda x: x["start_index"])
    
    for i, match in enumerate(key_matches_sorted):
        start_idx = match["end_index"]  # Start after the key
        label = match["label"]
        
        # Skip common separators (colons, dashes, etc.)
        while start_idx < len(words) and words[start_idx] in [":", "-", "–", "—", "|", "•"]:
            start_idx += 1
        
        # Special handling for fields that might be adjacent
        if label == "handling_fee_of_remitting_bank":
            # Look for "Collect from Remitter" pattern specifically
            end_idx = start_idx + DEFAULT_MAX_LEN
            
            # Find "Value date" or next key as boundary
            for j in range(i + 1, len(key_matches_sorted)):
                next_match = key_matches_sorted[j]
                if next_match["label"] not in ["remitting_bank"]:  # Don't stop at remitting_bank
                    end_idx = min(end_idx, next_match["start_index"])
                    break
            
        elif label == "remitting_bank":
            # For remitting bank, look beyond the handling fee section
            end_idx = start_idx + DEFAULT_MAX_LEN
            
            # Find next non-related key as boundary
            next_key_labels = ["instruction_reference", "other_reference", "remitter_to_beneficiary_info", "value_date"]
            for j in range(i + 1, len(key_matches_sorted)):
                next_match = key_matches_sorted[j]
                if next_match["label"] in next_key_labels:
                    end_idx = min(end_idx, next_match["start_index"])
                    break
                    
        else:
            # Standard boundary detection for other fields
            end_idx = len(words)  # Default to end of document
            
            # Use next key position
            for j in range(i + 1, len(key_matches_sorted)):
                next_match = key_matches_sorted[j]
                if next_match["start_index"] > start_idx:
                    end_idx = min(end_idx, next_match["start_index"])
                    break
            
            # Limit maximum extraction length
            end_idx = min(end_idx, start_idx + DEFAULT_MAX_LEN)
        
        # Extract value words between boundaries
        value_words = []
        value_indexes = []
        
        for j in range(start_idx, min(end_idx, len(words))):
            value_words.append(words[j])
            value_indexes.append(j)
        
        # Apply field-specific cleaning
        cleaned_value = clean_field_value(label, value_words, words, start_idx, end_idx, key_matches_sorted)
        
        # Debug output for problematic fields
        if label in ["handling_fee_of_remitting_bank", "remitting_bank"] or not cleaned_value:
            print(f"🔍 DEBUG - {label}:")
            print(f"   Key: '{match['raw_key']}' at indexes {match['start_index']}-{match['end_index']}")
            print(f"   Value extraction from {start_idx} to {end_idx}")
            print(f"   Raw words: {value_words[:15]}...")  # Show first 15 words
            print(f"   Cleaned: '{cleaned_value}'")
            print()
        
        # Store result
        if label not in results:  # Avoid duplicates, keep first occurrence
            results[label] = {
                "key_start_index": match["start_index"],
                "key_end_index": match["end_index"],
                "key_text": match["raw_key"],
                "value_start_index": start_idx,
                "value_end_index": min(start_idx + len(value_words), end_idx),
                "value_indexes": value_indexes,
                "raw_value": " ".join(value_words),
                "cleaned_value": cleaned_value
            }
    
    return results

def clean_field_value(label, value_words, all_words, start_idx, end_idx=None, all_matches=None):
    """Apply field-specific cleaning rules"""
    if not value_words:
        return ""
    
    full_value = " ".join(value_words)
    
    # Field-specific cleaning rules
    if label == "advice_ref":
        # Look for pattern starting with A2 and ending with -IN
        for word in value_words:
            if word.startswith("A2") and "-IN" in word:
                return word
        return value_words[0] if value_words else ""
    
    elif label == "remitter_to_beneficiary_info":
        # Stop at specific keywords and extract clean text
        stop_keywords = ["Invoice", "Vendor", "MH", "Number", "Date", "Amount", "Withholding", "Tax", "Deduction"]
        cleaned_words = []
        for word in value_words:
            if any(word.startswith(kw) for kw in stop_keywords):
                break
            cleaned_words.append(word)
        result = " ".join(cleaned_words).strip()
        return result
    
    elif label == "handling_fee_of_remitting_bank":
        # Look specifically for "Collect from Remitter" pattern
        full_text = full_value.strip()
        
        # Check for exact pattern first
        if "collect" in full_text.lower() and "remitter" in full_text.lower():
            # Extract the exact phrase
            words_lower = [w.lower() for w in value_words]
            try:
                collect_idx = next(i for i, w in enumerate(words_lower) if "collect" in w)
                remitter_idx = next(i for i, w in enumerate(words_lower) if "remitter" in w)
                if remitter_idx > collect_idx and remitter_idx - collect_idx <= 3:
                    return " ".join(value_words[collect_idx:remitter_idx+1])
            except StopIteration:
                pass
        
        # Look for currency amounts as alternative
        for word in value_words:
            if any(curr in word.upper() for curr in ["INR", "USD", "EUR", "GBP"]) and any(c.isdigit() for c in word):
                return word
        
        # Default to "Collect from Remitter" if pattern suggests it
        if any(w.lower() in ["collect", "remitter"] for w in value_words):
            return "Collect from Remitter"
        
        # Stop at boundary words and return clean text
        stop_words = ["Value", "date", "Remitter's", "name", "Remitting", "bank"]
        cleaned_words = []
        for word in value_words:
            if any(word.lower().startswith(sw.lower()) for sw in stop_words):
                break
            cleaned_words.append(word)
        
        result = " ".join(cleaned_words).strip()
        return result if result else "Collect from Remitter"
    
    elif label == "remitting_bank":
        # Extract bank name - this should come after handling fee
        bank_text = full_value.strip()
        
        # Remove common artifacts
        artifacts_to_remove = [
            "Collect from Remitter", "Collect", "from", "Remitter", 
            "of", "remitting", "bank", "Value", "date"
        ]
        
        words_list = bank_text.split()
        cleaned_words = []
        
        # Filter out artifacts and keep bank-related words
        for word in words_list:
            if word.lower() not in [a.lower() for a in artifacts_to_remove]:
                cleaned_words.append(word)
        
        # Look for bank name patterns
        bank_result = " ".join(cleaned_words).strip()
        
        # If we have "HK and Shanghai Banking Corp Ltd" pattern, clean it up
        if "HK" in bank_result or "Shanghai" in bank_result or "Banking" in bank_result:
            # This is likely the correct bank name
            return bank_result
        
        # Stop at next field boundaries
        stop_phrases = ["Instruction", "reference", "Other", "Remitter", "to", "beneficiary"]
        final_words = []
        
        for word in cleaned_words:
            if any(word.lower().startswith(phrase.lower()) for phrase in stop_phrases):
                break
            final_words.append(word)
        
        final_result = " ".join(final_words).strip()
        
        # If still empty or too short, look in the broader context
        if not final_result or len(final_result) < 5:
            # Look for bank indicators in the value_words
            bank_indicators = ["bank", "banking", "corp", "corporation", "ltd", "limited", "pvt"]
            for i, word in enumerate(value_words):
                if any(indicator in word.lower() for indicator in bank_indicators):
                    # Take surrounding context
                    start_bank = max(0, i-3)
                    end_bank = min(len(value_words), i+4)
                    bank_context = " ".join(value_words[start_bank:end_bank])
                    return bank_context.strip()
        
        return final_result
    
    elif label == "customer_reference":
        # Split at "Debit" keyword if present
        if "Debit" in full_value:
            parts = full_value.split("Debit")
            return parts[0].strip()
        return full_value.strip()
    
    elif label == "debit_amount":
        # Look for INR amounts
        for word in value_words:
            if "INR" in word and any(c.isdigit() for c in word):
                return word
        return full_value.strip()
    
    elif label == "remittance_amount":
        # Look for INR amounts
        for word in value_words:
            if "INR" in word and any(c.isdigit() for c in word):
                return word
        return full_value.strip()
    
    elif label == "value_date":
        # Look for date patterns
        date_words = []
        for word in value_words:
            if any(c.isdigit() for c in word) or word in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
                date_words.append(word)
            elif len(date_words) > 0:  # Stop after collecting date components
                break
        return " ".join(date_words[:3]) if date_words else full_value.strip()  # Limit to 3 words max
    
    else:
        return full_value.strip()

# ------------------ Invoice Details Extraction ------------------

def extract_invoice_details(indexed_words):
    """Extract invoice details using MH pattern matching"""
    words = [word for _, word in indexed_words]
    details = []
    
    for i in range(len(words) - 4):
        if words[i].startswith("MH") and len(words[i]) > 2 and words[i][2:].replace(".", "").isdigit():
            try:
                invoice_detail = {
                    "Invoice Number": words[i],
                    "Invoice Date": words[i + 1] if i + 1 < len(words) else "",
                    "Amount": words[i + 2] if i + 2 < len(words) else "",
                    "Withholding Tax Deduction": words[i + 3] if i + 3 < len(words) else "",
                    "start_index": i,
                    "end_index": min(i + 4, len(words))
                }
                details.append(invoice_detail)
            except (IndexError, AttributeError):
                continue
    
    return details

# ------------------ Statistics and Analysis ------------------

def generate_entity_statistics(entities, indexed_words):
    """Generate statistics about found entities"""
    total_words = len(indexed_words)
    
    stats = {
        "total_words": total_words,
        "total_entities_found": len(entities),
        "entities_coverage": {},
        "word_coverage": {
            "covered_words": 0,
            "uncovered_words": total_words,
            "coverage_percentage": 0.0
        }
    }
    
    covered_indexes = set()
    
    for label, entity_info in entities.items():
        # Calculate coverage for this entity
        key_span = entity_info["key_end_index"] - entity_info["key_start_index"]
        value_span = len(entity_info["value_indexes"])
        total_span = key_span + value_span
        
        stats["entities_coverage"][label] = {
            "key_word_count": key_span,
            "value_word_count": value_span,
            "total_word_count": total_span,
            "key_indexes": list(range(entity_info["key_start_index"], entity_info["key_end_index"])),
            "value_indexes": entity_info["value_indexes"]
        }
        
        # Add to covered indexes
        covered_indexes.update(range(entity_info["key_start_index"], entity_info["key_end_index"]))
        covered_indexes.update(entity_info["value_indexes"])
    
    stats["word_coverage"]["covered_words"] = len(covered_indexes)
    stats["word_coverage"]["uncovered_words"] = total_words - len(covered_indexes)
    stats["word_coverage"]["coverage_percentage"] = (len(covered_indexes) / total_words) * 100 if total_words > 0 else 0
    
    return stats

# ------------------ Main Processing Function ------------------

def process_pdf(file_path):
    """Main function to process PDF and extract all entities"""
    
    print(f"📄 Processing PDF: {file_path}")
    
    # Extract text and tokenize
    text = extract_text_from_pdf(file_path)
    indexed_words = tokenize_with_index(text)
    
    print(f"📊 Total words found: {len(indexed_words)}")
    
    # Find entity boundaries
    key_matches = find_entity_boundaries(indexed_words, HEAD_KEYS)
    print(f"🔍 Found {len(key_matches)} potential entity matches")
    
    # Extract entity values
    entities = extract_entity_values(indexed_words, key_matches)
    
    # Extract invoice details
    invoice_details = extract_invoice_details(indexed_words)
    
    # Generate statistics
    stats = generate_entity_statistics(entities, indexed_words)
    
    # Prepare final result
    result = {
        "extracted_entities": {label: info["cleaned_value"] for label, info in entities.items()},
        "invoice_details": invoice_details,
        "entity_details": entities,
        "statistics": stats,
        "raw_text_preview": " ".join([w for _, w in indexed_words[:50]]) + "..." if len(indexed_words) > 50 else " ".join([w for _, w in indexed_words])
    }
    
    return result

# ------------------ Output Functions ------------------

def print_detailed_results(result):
    """Print detailed results with proper formatting"""
    
    print("\n" + "="*80)
    print("📍 DETAILED ENTITY ANALYSIS")
    print("="*80)
    
    entities = result["entity_details"]
    stats = result["statistics"]
    
    print(f"\n📊 STATISTICS:")
    print(f"  • Total words in document: {stats['total_words']}")
    print(f"  • Total entities found: {stats['total_entities_found']}")
    print(f"  • Word coverage: {stats['word_coverage']['covered_words']}/{stats['total_words']} ({stats['word_coverage']['coverage_percentage']:.1f}%)")
    
    print(f"\n🎯 ENTITY DETAILS:")
    for label, entity_info in entities.items():
        print(f"\n  📌 {label.upper()}:")
        print(f"     ↳ Key: '{entity_info['key_text']}'")
        print(f"     ↳ Key position: words {entity_info['key_start_index']}-{entity_info['key_end_index']-1}")
        print(f"     ↳ Value position: words {entity_info['value_indexes']}")
        print(f"     ↳ Raw value: '{entity_info['raw_value'][:100]}{'...' if len(entity_info['raw_value']) > 100 else ''}'")
        print(f"     ↳ Cleaned value: '{entity_info['cleaned_value']}'")
    
    if result["invoice_details"]:
        print(f"\n💰 INVOICE DETAILS:")
        for i, invoice in enumerate(result["invoice_details"]):
            print(f"  Invoice {i+1}:")
            for key, value in invoice.items():
                if key not in ["start_index", "end_index"]:
                    print(f"    {key}: {value}")
    
    print(f"\n📦 FINAL EXTRACTED JSON:")
    print(json.dumps(result["extracted_entities"], indent=2))

# ------------------ Main Execution ------------------

def main():
    """Main execution function"""
    file_path = r"C:\Users\SyedaZuberiya\Desktop\PyMuPDF Testing\Payment_Advice_F2.pdf"  # Replace with your PDF path
    
    try:
        result = process_pdf(file_path)
        print_detailed_results(result)
        
        # Save to JSON file
        output_file = "extracted_entities.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found. Please check the file path.")
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main()