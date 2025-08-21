# document_processor.py

import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
import re
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Configuration
DEFAULT_HEAD_KEYS = {
    "Advice sending date": "advice_date",
    "Advice reference no": "advice_ref",
    "Recipient's name and contact information": "receipt_name",
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

DEFAULT_FIELD_TYPES = {
    "advice_date": "date",
    "advice_ref": "reference",
    "receipt_name": "split_text",
    "transaction_type": "text",
    "sub_payment_type": "text",
    "beneficiary_name": "text",
    "beneficiary_bank": "text",
    "account_number": "alphanumeric",
    "customer_reference": "text",
    "debit_amount": "amount",
    "remittance_amount": "amount",
    "handling_fee_of_remitting_bank": "text",
    "value_date": "date",
    "remitter_name": "text",
    "remitting_bank": "text",
    "instruction_reference": "text",
    "other_reference": "text",
    "remitter_to_beneficiary_info": "text"
}

DEFAULT_MAX_LEN = 30

DEFAULT_FIELD_COLORS = {
    "advice_date": (255, 0, 0),
    "advice_ref": (0, 255, 0),
    "receipt_name": (0, 0, 255),
    "transaction_type": (255, 165, 0),
    "sub_payment_type": (255, 0, 255),
    "beneficiary_name": (0, 255, 255),
    "beneficiary_bank": (128, 0, 128),
    "account_number": (255, 192, 203),
    "customer_reference": (0, 128, 0),
    "debit_amount": (255, 215, 0),
    "remittance_amount": (230, 230, 250),
    "handling_fee_of_remitting_bank": (255, 69, 0),
    "value_date": (135, 206, 235),
    "remitter_name": (238, 130, 238),
    "remitting_bank": (34, 139, 34),
    "instruction_reference": (255, 218, 185),
    "other_reference": (75, 0, 130),
    "remitter_to_beneficiary_info": (154, 205, 50),
}

# Text extraction functions
def extract_text_with_coordinates(file_path, file_type="pdf"):
    """Extract text with coordinates from PDF or Image"""
    words_with_coords = []
    
    if file_type == "pdf":
        doc = fitz.open(file_path)
        
        for page_num, page in enumerate(doc):
            words = page.get_text("words")
            
            for word_info in words:
                x0, y0, x1, y1, word, block_no, line_no, word_no = word_info
                words_with_coords.append({
                    "word": word,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "page": page_num,
                    "block": block_no,
                    "line": line_no,
                    "word_no": word_no
                })
        
        doc.close()
    
    elif file_type == "image":
        img = Image.open(file_path)
        ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)
        
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                word = ocr_data['text'][i].strip()
                
                words_with_coords.append({
                    "word": word,
                    "x0": x,
                    "y0": y,
                    "x1": x + w,
                    "y1": y + h,
                    "page": 0,
                    "block": ocr_data['block_num'][i],
                    "line": ocr_data['line_num'][i],
                    "word_no": ocr_data['word_num'][i]
                })
    
    return words_with_coords

def tokenize_with_index(words_with_coords):
    """Tokenize text and return list of (index, word_info) tuples"""
    return [(i, word_info) for i, word_info in enumerate(words_with_coords)]

# Entity detection functions
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

def find_entity_boundaries(indexed_words, head_keys):
    """Find all entity positions and their boundaries using index-based approach"""
    words = [w_info["word"] for _, w_info in indexed_words]
    normalized_key_map = create_normalized_key_map(head_keys)
    
    key_matches = []
    
    for i in range(len(words)):
        for key_length in range(6, 0, -1):
            if i + key_length > len(words):
                continue
                
            phrase_words = [normalize(words[i + j]) for j in range(key_length)]
            normalized_phrase = " ".join(phrase_words)
            exact_phrase = " ".join(words[i:i + key_length]).lower()
            
            matched_key = None
            if normalized_phrase in normalized_key_map:
                matched_key = normalized_key_map[normalized_phrase]
            else:
                for raw_key, label in head_keys.items():
                    if raw_key.lower() == exact_phrase:
                        matched_key = {
                            "raw_key": raw_key,
                            "label": label,
                            "word_count": len(raw_key.split())
                        }
                        break
            
            if not matched_key and "recipient" in normalized_phrase and "name" in normalized_phrase:
                if key_length >= 2:
                    matched_key = {
                        "raw_key": "Recipient's name",
                        "label": "receipt_name",
                        "word_count": 2
                    }
            
            if matched_key:
                key_info = matched_key
                
                is_duplicate = False
                for existing_match in key_matches:
                    if (existing_match["start_index"] <= i < existing_match["end_index"] or
                        i <= existing_match["start_index"] < i + key_length):
                        if key_length <= (existing_match["end_index"] - existing_match["start_index"]):
                            is_duplicate = True
                            break
                        else:
                            key_matches.remove(existing_match)
                
                if not is_duplicate:
                    key_matches.append({
                        "start_index": i,
                        "end_index": i + min(key_length, key_info["word_count"]),
                        "raw_key": key_info["raw_key"],
                        "label": key_info["label"],
                        "normalized_key": normalized_phrase,
                        "actual_words": " ".join(words[i:i + key_length])
                    })
                break
    
    key_matches.sort(key=lambda x: x["start_index"])
    return key_matches

def is_line_break_between(words_with_info, idx1, idx2):
    """Check if there's a significant line break between two word indices"""
    if idx1 >= len(words_with_info) or idx2 >= len(words_with_info) or idx1 >= idx2:
        return False
    
    word1_info = words_with_info[idx1]
    word2_info = words_with_info[idx2]
    
    if word1_info.get("line", 0) != word2_info.get("line", 0):
        return True
    
    y_diff = abs(word2_info["y0"] - word1_info["y0"])
    avg_height = (word1_info["y1"] - word1_info["y0"] + word2_info["y1"] - word2_info["y0"]) / 2
    
    if y_diff > avg_height * 1.5:
        return True
    
    return False

def find_next_key_boundary(current_index, key_matches_sorted, current_match_idx):
    """Find the next key boundary, excluding certain close fields"""
    next_boundary = float('inf')
    
    for j in range(current_match_idx + 1, len(key_matches_sorted)):
        next_match = key_matches_sorted[j]
        if next_match["start_index"] > current_index:
            next_boundary = next_match["start_index"]
            break
    
    return next_boundary

def clean_field_value(label, value_words, all_words, start_idx, end_idx=None, all_matches=None, field_type="text"):
    """Apply field-specific cleaning rules with improved logic based on field type"""
    if not value_words:
        return ""
    
    full_value = " ".join(value_words)
    
    if field_type == "reference":
        for word in value_words:
            if len(word) > 5 and any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                return word
        
        for word in value_words:
            if len(word) > 2 and word not in [":", "-", "–", "—", "|", "•"]:
                return word
        
        return value_words[0] if value_words else ""
    
    elif field_type == "date":
        date_words = []
        for word in value_words:
            if any(c.isdigit() for c in word) or word in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
                date_words.append(word)
            elif len(date_words) > 0:
                break
        
        if date_words:
            return " ".join(date_words[:3])
        
        return " ".join(value_words[:2]) if value_words else ""
    
    elif field_type == "amount":
        combined_text = " ".join(value_words)
        
        file_type = "pdf"  # Default to pdf for this function
        
        currency_pattern = r'(?:INR|USD|EUR|GBP|\$|£|€)\s*[\d,]+(?:\.\d+)?'
        match = re.search(currency_pattern, combined_text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        
        amount_parts = []
        found_currency = False
        found_digits = False
        
        for j, word in enumerate(value_words):
            if any(curr in word.upper() for curr in ["INR", "USD", "EUR", "GBP", "$", "£", "€"]):
                amount_parts.append(word)
                found_currency = True
            elif found_currency:
                if word.replace(",", "").replace(".", "").isdigit():
                    amount_parts.append(word)
                    found_digits = True
                elif word in [",", "."] and found_digits:
                    amount_parts.append(word)
                elif found_digits and word not in [",", "."]:
                    break
        
        if amount_parts:
            return "".join(amount_parts).strip()
        
        for i, word in enumerate(value_words):
            if any(curr in word.upper() for curr in ["INR", "USD", "EUR", "GBP", "$", "£", "€"]):
                amount_words = [word]
                for j in range(i+1, min(i+10, len(value_words))):
                    next_word = value_words[j]
                    if next_word.replace(",", "").replace(".", "").isdigit() or next_word in [",", "."]:
                        amount_words.append(next_word)
                    else:
                        break
                return "".join(amount_words).strip()
        
        for i, word in enumerate(value_words):
            if any(curr in word.upper() for curr in ["INR", "USD", "EUR", "GBP"]):
                number_parts = []
                for j in range(i+1, min(i+10, len(value_words))):
                    if value_words[j].replace(",", "").replace(".", "").isdigit():
                        number_parts.append(value_words[j])
                    elif value_words[j] in [",", "."] and number_parts:
                        number_parts.append(value_words[j])
                    elif number_parts:
                        break
                
                if number_parts:
                    return f"{word}{''.join(number_parts)}"
        
        inr_pattern = r'INR\s*[\d,]+(?:\.\d+)?'
        match = re.search(inr_pattern, combined_text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        
        currency_pattern = r'[A-Z]{3}\s*[\d,]+(?:\.\d+)?'
        match = re.search(currency_pattern, combined_text)
        if match:
            return match.group(0).strip()
        
        number_pattern = r'[\$£€¥]\s*[\d,]+(?:\.\d+)?'
        match = re.search(number_pattern, combined_text)
        if match:
            return match.group(0).strip()
        
        digit_pattern = r'[\d,]+(?:\.\d+)?'
        match = re.search(digit_pattern, combined_text)
        if match:
            amount_text = match.group(0).strip()
            if len(amount_text) > 3:
                return amount_text
        
        return " ".join(value_words[:5]).strip()
    
    elif field_type == "alphanumeric":
        for word in value_words:
            if len(word) > 6 and any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                return word
        
        for word in value_words:
            if len(word) > 6 and any(c.isdigit() for c in word):
                return word
        
        return " ".join(value_words[:2]).strip()
    
    elif field_type == "split_text" and label == "receipt_name":
        filtered_words = []
        skip_words = ["and", "contact", "information:", "information"]
        
        for word in value_words:
            if word.lower() not in skip_words:
                filtered_words.append(word)
        
        company_words = []
        email_found = False
        
        for word in filtered_words:
            if word.lower() in ["page:", "page", "1/1", "/"]:
                break
                
            if "@" in word or ".com" in word.lower() or ".co.in" in word.lower():
                company_words.append(word)
                email_found = True
                break
            else:
                company_words.append(word)
        
        result = " ".join(company_words).strip()
        result = result.replace(":", "").strip()
        
        return result
    
    else:
        clean_words = []
        for word in value_words:
            if word not in [":", "-", "–", "—", "|", "•"] and len(word) > 0:
                clean_words.append(word)
            if len(clean_words) >= 5:
                break
        return " ".join(clean_words).strip()

def merge_bounding_boxes(boxes):
    """Merge multiple bounding boxes into one"""
    if not boxes:
        return None
    
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    
    return (x0, y0, x1, y1)

def extract_entity_values(indexed_words, key_matches, field_types):
    """Extract values for each entity using index boundaries with improved logic"""
    words_with_info = [w_info for _, w_info in indexed_words]
    words = [w_info["word"] for w_info in words_with_info]
    results = {}
    
    key_matches_sorted = sorted(key_matches, key=lambda x: x["start_index"])
    
    for i, match in enumerate(key_matches_sorted):
        start_idx = match["end_index"]
        label = match["label"]
        
        field_type = field_types.get(label, "text")
        
        while start_idx < len(words) and words[start_idx] in [":", "-", "–", "—", "|", "•", ".", "and", "contact", "information"]:
            start_idx += 1
        
        if field_type == "reference":
            end_idx = start_idx + 5
            next_boundary = find_next_key_boundary(start_idx, key_matches_sorted, i)
            end_idx = min(end_idx, next_boundary)
            
            for j in range(start_idx + 1, min(start_idx + 5, len(words_with_info))):
                if is_line_break_between(words_with_info, start_idx, j):
                    end_idx = min(end_idx, j)
                    break
        
        elif field_type == "date":
            end_idx = start_idx + 4
            next_boundary = find_next_key_boundary(start_idx, key_matches_sorted, i)
            end_idx = min(end_idx, next_boundary)
        
        elif field_type == "amount":
            end_idx = start_idx + 100
            next_boundary = find_next_key_boundary(start_idx, key_matches_sorted, i)
            end_idx = min(end_idx, next_boundary)
            
            file_type = "pdf"  # Default to pdf for this function
            if file_type == 'image':
                found_currency = False
                for j in range(start_idx, min(end_idx, len(words))):
                    word = words[j]
                    if any(curr in word.upper() for curr in ["INR", "USD", "EUR", "GBP"]):
                        found_currency = True
                    elif found_currency and any(c.isdigit() for c in word):
                        continue
                    elif found_currency and word.strip() in [",", ".", ""]:
                        continue
                    elif found_currency:
                        end_idx = min(end_idx, j)
                        break
        
        elif field_type == "split_text":
            end_idx = start_idx + 20
            next_boundary = find_next_key_boundary(start_idx, key_matches_sorted, i)
            end_idx = min(end_idx, next_boundary)
            
            for j in range(start_idx, min(end_idx, len(words))):
                if (words[j].lower() in ["page:", "page", "1/1", "/"] or 
                    ".com" in words[j].lower() or 
                    ".co.in" in words[j].lower()):
                    end_idx = j + 1
                    break
        
        else:
            end_idx = start_idx + DEFAULT_MAX_LEN
            next_boundary = find_next_key_boundary(start_idx, key_matches_sorted, i)
            end_idx = min(end_idx, next_boundary)
        
        value_words = []
        value_indexes = []
        
        for j in range(start_idx, min(end_idx, len(words))):
            value_words.append(words[j])
            value_indexes.append(j)
        
        cleaned_value = clean_field_value(label, value_words, words, start_idx, end_idx, key_matches_sorted, field_type)
        
        actual_key_length = len(match["raw_key"].split())
        key_word_infos = words_with_info[match["start_index"]:match["start_index"] + actual_key_length]
        key_boxes = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in key_word_infos]
        key_merged_box = merge_bounding_boxes(key_boxes)
        key_page = key_word_infos[0]["page"] if key_word_infos else 0
        
        cleaned_value_words = cleaned_value.split() if cleaned_value else []
        actual_value_indexes = []
        
        if field_type == "split_text" and label == "receipt_name":
            value_parts = []
            
            email_match = re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', cleaned_value)
            if email_match:
                email = email_match.group(0)
                company_name = cleaned_value[:email_match.start()].strip()
                
                company_words = company_name.split()
                company_start_idx = None
                company_end_idx = None
                
                for i in range(len(value_words) - len(company_words) + 1):
                    if value_words[i:i+len(company_words)] == company_words:
                        company_start_idx = value_indexes[i]
                        company_end_idx = value_indexes[i+len(company_words)-1] + 1
                        break
                
                email_start_idx = None
                email_end_idx = None
                for i in range(len(value_words)):
                    if email in value_words[i]:
                        email_start_idx = value_indexes[i]
                        email_end_idx = value_indexes[i] + 1
                        break
                
                if company_start_idx is not None and company_end_idx is not None:
                    company_word_infos = [words_with_info[idx] for idx in range(company_start_idx, company_end_idx) if idx < len(words_with_info)]
                    company_boxes = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in company_word_infos]
                    company_merged_box = merge_bounding_boxes(company_boxes)
                    company_page = company_word_infos[0]["page"] if company_word_infos else 0
                    
                    value_parts.append({
                        "text": company_name,
                        "coordinates": {
                            "x": company_merged_box[0] if company_merged_box else 0,
                            "y": company_merged_box[1] if company_merged_box else 0,
                            "width": company_merged_box[2] - company_merged_box[0] if company_merged_box else 0,
                            "height": company_merged_box[3] - company_merged_box[1] if company_merged_box else 0,
                            "page": company_page + 1
                        }
                    })
                
                if email_start_idx is not None and email_end_idx is not None:
                    email_word_infos = [words_with_info[idx] for idx in range(email_start_idx, email_end_idx) if idx < len(words_with_info)]
                    email_boxes = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in email_word_infos]
                    email_merged_box = merge_bounding_boxes(email_boxes)
                    email_page = email_word_infos[0]["page"] if email_word_infos else 0
                    
                    value_parts.append({
                        "text": email,
                        "coordinates": {
                            "x": email_merged_box[0] if email_merged_box else 0,
                            "y": email_merged_box[1] if email_merged_box else 0,
                            "width": email_merged_box[2] - email_merged_box[0] if email_merged_box else 0,
                            "height": email_merged_box[3] - email_merged_box[1] if email_merged_box else 0,
                            "page": email_page + 1
                        }
                    })
                
                if company_start_idx is not None and email_end_idx is not None:
                    actual_value_indexes = list(range(company_start_idx, email_end_idx))
            else:
                if cleaned_value_words and value_indexes:
                    cleaned_text = cleaned_value.lower().replace(",", "").replace(".", "")
                    
                    for j, idx in enumerate(value_indexes):
                        if j < len(value_words):
                            word = value_words[j].lower().replace(",", "").replace(".", "")
                            if word in cleaned_text:
                                actual_value_indexes.append(idx)
                    
                    if not actual_value_indexes and value_indexes:
                        word_count = min(len(cleaned_value_words), len(value_indexes))
                        actual_value_indexes = value_indexes[:word_count]
        else:
            if cleaned_value_words and value_indexes:
                cleaned_text = cleaned_value.lower().replace(",", "").replace(".", "")
                
                for j, idx in enumerate(value_indexes):
                    if j < len(value_words):
                        word = value_words[j].lower().replace(",", "").replace(".", "")
                        if word in cleaned_text:
                            actual_value_indexes.append(idx)
                
                if not actual_value_indexes and value_indexes:
                    word_count = min(len(cleaned_value_words), len(value_indexes))
                    actual_value_indexes = value_indexes[:word_count]
        
        value_word_infos = [words_with_info[idx] for idx in actual_value_indexes if idx < len(words_with_info)]
        value_boxes = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in value_word_infos]
        value_merged_box = merge_bounding_boxes(value_boxes)
        value_page = value_word_infos[0]["page"] if value_word_infos else 0
        
        if label not in results:
            results[label] = {
                "key_start_index": match["start_index"],
                "key_end_index": match["start_index"] + actual_key_length,
                "key_text": match["raw_key"],
                "value_start_index": start_idx,
                "value_end_index": min(start_idx + len(actual_value_indexes), end_idx),
                "value_indexes": actual_value_indexes,
                "raw_value": " ".join(value_words),
                "cleaned_value": cleaned_value,
                "key_coordinates": {
                    "x": key_merged_box[0] if key_merged_box else 0,
                    "y": key_merged_box[1] if key_merged_box else 0,
                    "width": key_merged_box[2] - key_merged_box[0] if key_merged_box else 0,
                    "height": key_merged_box[3] - key_merged_box[1] if key_merged_box else 0,
                    "page": key_page + 1
                },
                "value_coordinates": {
                    "x": value_merged_box[0] if value_merged_box else 0,
                    "y": value_merged_box[1] if value_merged_box else 0,
                    "width": value_merged_box[2] - value_merged_box[0] if value_merged_box else 0,
                    "height": value_merged_box[3] - value_merged_box[1] if value_merged_box else 0,
                    "page": value_page + 1
                }
            }
            
            if field_type == "split_text" and label == "receipt_name" and 'value_parts' in locals():
                results[label]["value_parts"] = value_parts
    
    return results

# Bounding boxes functions
def draw_bounding_boxes_on_document(file_path, entity_details, file_type="pdf", field_colors=None):
    """Draw bounding boxes for all header fields on the document pages"""
    annotated_images = []
    
    if field_colors is None:
        field_colors = DEFAULT_FIELD_COLORS
    
    if file_type == "pdf":
        doc = fitz.open(file_path)
        
        entities_by_page = {}
        for label, info in entity_details.items():
            key_coords = info.get("key_coordinates", {})
            
            if key_coords.get("page") is not None:
                page_num = key_coords["page"] - 1
                if page_num not in entities_by_page:
                    entities_by_page[page_num] = []
                entities_by_page[page_num].append({
                    "label": label,
                    "key_coords": key_coords,
                    "value_coords": info.get("value_coordinates", {}),
                    "value_parts": info.get("value_parts", [])
                })
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                label_font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
                label_font = ImageFont.load_default()
            
            if page_num in entities_by_page:
                for entity in entities_by_page[page_num]:
                    label = entity["label"]
                    key_coords = entity["key_coords"]
                    value_coords = entity["value_coords"]
                    value_parts = entity["value_parts"]
                    
                    color = field_colors.get(label, (255, 0, 0))
                    
                    if key_coords and key_coords.get("width", 0) > 0 and key_coords.get("height", 0) > 0:
                        x0 = key_coords["x"] * 2
                        y0 = key_coords["y"] * 2
                        width = key_coords["width"] * 2
                        height = key_coords["height"] * 2
                        
                        draw.rectangle([x0, y0, x0 + width, y0 + height], outline=color, width=3)
                        
                        label_text = label.replace("_", " ").title()
                        text_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        draw.rectangle(
                            [x0, y0 - text_height - 4, x0 + text_width + 8, y0 - 4],
                            fill=color
                        )
                        
                        draw.text((x0 + 4, y0 - text_height - 4), label_text, fill="white", font=label_font)
                    
                    if value_parts:
                        for part in value_parts:
                            part_coords = part["coordinates"]
                            if part_coords and part_coords.get("width", 0) > 0 and part_coords.get("height", 0) > 0:
                                vx0 = part_coords["x"] * 2
                                vy0 = part_coords["y"] * 2
                                vwidth = part_coords["width"] * 2
                                vheight = part_coords["height"] * 2
                                
                                draw.rectangle([vx0, vy0, vx0 + vwidth, vy0 + vheight], outline=color, width=2)
                                
                                part_text = part["text"]
                                if part_text:
                                    part_bbox = draw.textbbox((0, 0), part_text[:50], font=font)
                                    part_text_width = part_bbox[2] - part_bbox[0]
                                    part_text_height = part_bbox[3] - part_bbox[1]
                                    
                                    draw.rectangle(
                                        [vx0, vy0 + vheight, vx0 + part_text_width + 8, vy0 + vheight + part_text_height + 4],
                                        fill=(200, 200, 200)
                                    )
                                    
                                    draw.text((vx0 + 4, vy0 + vheight + 2), part_text[:50], fill="black", font=font)
                    elif value_coords and value_coords.get("width", 0) > 0 and value_coords.get("height", 0) > 0:
                        vx0 = value_coords["x"] * 2
                        vy0 = value_coords["y"] * 2
                        vwidth = value_coords["width"] * 2
                        vheight = value_coords["height"] * 2
                        
                        draw.rectangle([vx0, vy0, vx0 + vwidth, vy0 + vheight], outline=color, width=2)
                        
                        value_text = entity_details[label]["cleaned_value"]
                        if value_text:
                            value_bbox = draw.textbbox((0, 0), value_text[:50], font=font)
                            value_text_width = value_bbox[2] - value_bbox[0]
                            value_text_height = value_bbox[3] - value_bbox[1]
                            
                            draw.rectangle(
                                [vx0, vy0 + vheight, vx0 + value_text_width + 8, vy0 + vheight + value_text_height + 4],
                                fill=(200, 200, 200)
                            )
                            
                            draw.text((vx0 + 4, vy0 + vheight + 2), value_text[:50], fill="black", font=font)
            
            annotated_images.append(img)
        
        doc.close()
    
    elif file_type == "image":
        img = Image.open(file_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            label_font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            label_font = ImageFont.load_default()
        
        for label, info in entity_details.items():
            key_coords = info.get("key_coordinates", {})
            value_coords = info.get("value_coordinates", {})
            value_parts = info.get("value_parts", [])
            
            color = field_colors.get(label, (255, 0, 0))
            
            if key_coords and key_coords.get("width", 0) > 0 and key_coords.get("height", 0) > 0:
                x0 = key_coords["x"]
                y0 = key_coords["y"]
                width = key_coords["width"]
                height = key_coords["height"]
                
                draw.rectangle([x0, y0, x0 + width, y0 + height], outline=color, width=3)
                
                label_text = label.replace("_", " ").title()
                text_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                draw.rectangle(
                    [x0, y0 - text_height - 4, x0 + text_width + 8, y0 - 4],
                    fill=color
                )
                
                draw.text((x0 + 4, y0 - text_height - 4), label_text, fill="white", font=label_font)
            
            if value_parts:
                for part in value_parts:
                    part_coords = part["coordinates"]
                    if part_coords and part_coords.get("width", 0) > 0 and part_coords.get("height", 0) > 0:
                        vx0 = part_coords["x"]
                        vy0 = part_coords["y"]
                        vwidth = part_coords["width"]
                        vheight = part_coords["height"]
                        
                        draw.rectangle([vx0, vy0, vx0 + vwidth, vy0 + vheight], outline=color, width=2)
                        
                        part_text = part["text"]
                        if part_text:
                            part_bbox = draw.textbbox((0, 0), part_text[:50], font=font)
                            part_text_width = part_bbox[2] - part_bbox[0]
                            part_text_height = part_bbox[3] - part_bbox[1]
                            
                            draw.rectangle(
                                [vx0, vy0 + vheight, vx0 + part_text_width + 8, vy0 + vheight + part_text_height + 4],
                                fill=(200, 200, 200)
                            )
                            
                            draw.text((vx0 + 4, vy0 + vheight + 2), part_text[:50], fill="black", font=font)
            elif value_coords and value_coords.get("width", 0) > 0 and value_coords.get("height", 0) > 0:
                vx0 = value_coords["x"]
                vy0 = value_coords["y"]
                vwidth = value_coords["width"]
                vheight = value_coords["height"]
                
                draw.rectangle([vx0, vy0, vx0 + vwidth, vy0 + vheight], outline=color, width=2)
                
                value_text = info["cleaned_value"]
                if value_text:
                    value_bbox = draw.textbbox((0, 0), value_text[:50], font=font)
                    value_text_width = value_bbox[2] - value_bbox[0]
                    value_text_height = value_bbox[3] - value_bbox[1]
                    
                    draw.rectangle(
                        [vx0, vy0 + vheight, vx0 + value_text_width + 8, vy0 + vheight + value_text_height + 4],
                        fill=(200, 200, 200)
                    )
                    
                    draw.text((vx0 + 4, vy0 + vheight + 2), value_text[:50], fill="black", font=font)
        
        annotated_images.append(img)
    
    return annotated_images

# Statistics functions
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
        
        covered_indexes.update(range(entity_info["key_start_index"], entity_info["key_end_index"]))
        covered_indexes.update(entity_info["value_indexes"])
    
    stats["word_coverage"]["covered_words"] = len(covered_indexes)
    stats["word_coverage"]["uncovered_words"] = total_words - len(covered_indexes)
    stats["word_coverage"]["coverage_percentage"] = (len(covered_indexes) / total_words) * 100 if total_words > 0 else 0
    
    return stats

# Main processing function
def process_document(file_path, file_type="pdf", head_keys=None, field_types=None):
    """Main function to process document and extract all entities"""
    if head_keys is None:
        head_keys = DEFAULT_HEAD_KEYS
    
    if field_types is None:
        field_types = DEFAULT_FIELD_TYPES
    
    words_with_coords = extract_text_with_coordinates(file_path, file_type)
    indexed_words = tokenize_with_index(words_with_coords)
    
    key_matches = find_entity_boundaries(indexed_words, head_keys)
    
    entities = extract_entity_values(indexed_words, key_matches, field_types)
    
    stats = generate_entity_statistics(entities, indexed_words)
    
    result = {
        "extracted_entities": {label: info["cleaned_value"] for label, info in entities.items()},
        "entity_details": entities,
        "statistics": stats,
        "raw_text_preview": " ".join([w_info["word"] for _, w_info in indexed_words[:50]]) + "..." if len(indexed_words) > 50 else " ".join([w_info["word"] for _, w_info in indexed_words])
    }
    
    return result