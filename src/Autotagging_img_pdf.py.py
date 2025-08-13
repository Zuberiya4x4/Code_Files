#AUTOTAGGING_CODE_FOR_IMG_AND_PDF
import streamlit as st
import json
import fitz  # PyMuPDF
import pdfplumber
from io import BytesIO
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import tempfile
import pytesseract
from pytesseract import Output

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

# Color mapping for different field types
FIELD_COLORS = {
    "advice_date": (255, 0, 0),        # Red
    "advice_ref": (0, 255, 0),         # Green
    "receipt_name": (0, 0, 255),       # Blue
    "transaction_type": (255, 165, 0), # Orange
    "sub_payment_type": (255, 0, 255),   # Magenta
    "beneficiary_name": (0, 255, 255),   # Cyan
    "beneficiary_bank": (128, 0, 128), # Purple
    "account_number": (255, 192, 203), # Pink
    "customer_reference": (0, 128, 0), # Light Green
    "debit_amount": (255, 215, 0),   # Dark Yellow
    "remittance_amount": (230, 230, 250), # Lavender
    "handling_fee_of_remitting_bank": (255, 69, 0), # Red Orange
    "value_date": (135, 206, 235),   # Sky Blue
    "remitter_name": (238, 130, 238), # Violet
    "remitting_bank": (34, 139, 34), # Forest Green
    "instruction_reference": (255, 218, 185), # Peach
    "other_reference": (75, 0, 130), # Indigo
    "remitter_to_beneficiary_info": (154, 205, 50), # Lime
}

# ------------------ Helpers ------------------
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
        # Open image using PIL
        img = Image.open(file_path)
        
        # Use pytesseract to extract text with bounding boxes
        ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT)
        
        # Process OCR data
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
                    "page": 0,  # Images have only one page
                    "block": ocr_data['block_num'][i],
                    "line": ocr_data['line_num'][i],
                    "word_no": ocr_data['word_num'][i]
                })
    
    return words_with_coords

def tokenize_with_index(words_with_coords):
    """Tokenize text and return list of (index, word_info) tuples"""
    return [(i, word_info) for i, word_info in enumerate(words_with_coords)]

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

def merge_bounding_boxes(boxes):
    """Merge multiple bounding boxes into one"""
    if not boxes:
        return None
    
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    
    return (x0, y0, x1, y1)

def draw_bounding_boxes_on_document(file_path, entity_details, file_type="pdf"):
    """Draw bounding boxes for all header fields on the document pages"""
    annotated_images = []
    
    if file_type == "pdf":
        doc = fitz.open(file_path)
        
        # Group entities by page
        entities_by_page = {}
        for label, info in entity_details.items():
            key_coords = info.get("key_coordinates", {})
            value_coords = info.get("value_coordinates", {})
            
            # If key_coords has a page, use that
            if key_coords.get("page") is not None:
                page_num = key_coords["page"] - 1  # Convert to 0-based index
                if page_num not in entities_by_page:
                    entities_by_page[page_num] = []
                entities_by_page[page_num].append({
                    "label": label,
                    "key_coords": key_coords,
                    "value_coords": value_coords
                })
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Convert page to image with 2x scaling for better quality
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(img)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                label_font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
                label_font = ImageFont.load_default()
            
            # Only draw if there are entities on this page
            if page_num in entities_by_page:
                for entity in entities_by_page[page_num]:
                    label = entity["label"]
                    key_coords = entity["key_coords"]
                    value_coords = entity["value_coords"]
                    
                    # Get color for this field
                    color = FIELD_COLORS.get(label, (255, 0, 0))
                    
                    # Draw key bounding box
                    if key_coords:
                        x0 = key_coords["x"] * 2  # Scale up because we scaled the image 2x
                        y0 = key_coords["y"] * 2
                        width = key_coords["width"] * 2
                        height = key_coords["height"] * 2
                        
                        draw.rectangle([x0, y0, x0 + width, y0 + height], outline=color, width=3)
                        
                        # Draw label above the key box
                        label_text = label.replace("_", " ").title()
                        text_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        # Draw background for label
                        draw.rectangle(
                            [x0, y0 - text_height - 4, x0 + text_width + 8, y0 - 4],
                            fill=color
                        )
                        
                        # Draw label text
                        draw.text((x0 + 4, y0 - text_height - 4), label_text, fill="white", font=label_font)
                    
                    # Draw value bounding box
                    if value_coords:
                        vx0 = value_coords["x"] * 2
                        vy0 = value_coords["y"] * 2
                        vwidth = value_coords["width"] * 2
                        vheight = value_coords["height"] * 2
                        
                        draw.rectangle([vx0, vy0, vx0 + vwidth, vy0 + vheight], outline=color, width=2)
                        
                        # Draw value text
                        value_text = entity_details[label]["cleaned_value"]
                        if value_text:
                            # Draw background for value text
                            value_bbox = draw.textbbox((0, 0), value_text, font=font)
                            value_text_width = value_bbox[2] - value_bbox[0]
                            value_text_height = value_bbox[3] - value_bbox[1]
                            
                            # Position value text below the value box
                            draw.rectangle(
                                [vx0, vy0 + vheight, vx0 + value_text_width + 8, vy0 + vheight + value_text_height + 4],
                                fill=(200, 200, 200)
                            )
                            
                            draw.text((vx0 + 4, vy0 + vheight + 2), value_text, fill="black", font=font)
            
            annotated_images.append(img)
        
        doc.close()
    
    elif file_type == "image":
        # Open the image directly
        img = Image.open(file_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            label_font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            label_font = ImageFont.load_default()
        
        # Draw all entities
        for label, info in entity_details.items():
            key_coords = info.get("key_coordinates", {})
            value_coords = info.get("value_coordinates", {})
            
            # Get color for this field
            color = FIELD_COLORS.get(label, (255, 0, 0))
            
            # Draw key bounding box
            if key_coords:
                x0 = key_coords["x"]
                y0 = key_coords["y"]
                width = key_coords["width"]
                height = key_coords["height"]
                
                draw.rectangle([x0, y0, x0 + width, y0 + height], outline=color, width=3)
                
                # Draw label above the key box
                label_text = label.replace("_", " ").title()
                text_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Draw background for label
                draw.rectangle(
                    [x0, y0 - text_height - 4, x0 + text_width + 8, y0 - 4],
                    fill=color
                )
                
                # Draw label text
                draw.text((x0 + 4, y0 - text_height - 4), label_text, fill="white", font=label_font)
            
            # Draw value bounding box
            if value_coords:
                vx0 = value_coords["x"]
                vy0 = value_coords["y"]
                vwidth = value_coords["width"]
                vheight = value_coords["height"]
                
                draw.rectangle([vx0, vy0, vx0 + vwidth, vy0 + vheight], outline=color, width=2)
                
                # Draw value text
                value_text = info["cleaned_value"]
                if value_text:
                    # Draw background for value text
                    value_bbox = draw.textbbox((0, 0), value_text, font=font)
                    value_text_width = value_bbox[2] - value_bbox[0]
                    value_text_height = value_bbox[3] - value_bbox[1]
                    
                    # Position value text below the value box
                    draw.rectangle(
                        [vx0, vy0 + vheight, vx0 + value_text_width + 8, vy0 + vheight + value_text_height + 4],
                        fill=(200, 200, 200)
                    )
                    
                    draw.text((vx0 + 4, vy0 + vheight + 2), value_text, fill="black", font=font)
        
        annotated_images.append(img)
    
    return annotated_images

# ------------------ Enhanced Entity Detection ------------------
def find_entity_boundaries(indexed_words, head_keys):
    """Find all entity positions and their boundaries using index-based approach"""
    words = [w_info["word"] for _, w_info in indexed_words]
    normalized_key_map = create_normalized_key_map(head_keys)
    
    # Find all potential key matches
    key_matches = []
    
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
                break  # Found a match, don't look for shorter matches at this position
    
    # Sort by start index
    key_matches.sort(key=lambda x: x["start_index"])
    
    return key_matches

def extract_entity_values(indexed_words, key_matches):
    """Extract values for each entity using index boundaries"""
    words_with_info = [w_info for _, w_info in indexed_words]
    words = [w_info["word"] for w_info in words_with_info]
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
                    
        elif label == "debit_amount":
            # For debit amount, look for the complete amount including multi-line values
            # Start by looking at the current line and the next line
            end_idx = start_idx + 20  # Look ahead up to 20 words
            
            # Find next key as boundary, but skip remittance_amount as it might be close
            for j in range(i + 1, len(key_matches_sorted)):
                next_match = key_matches_sorted[j]
                if next_match["start_index"] > start_idx and next_match["label"] != "remittance_amount":
                    end_idx = min(end_idx, next_match["start_index"])
                    break
            
            # Limit maximum extraction length
            end_idx = min(end_idx, start_idx + 20)
            
        elif label == "remittance_amount":
            # For remittance amount, look for the complete amount including multi-line values
            # This might span multiple lines, so we need to look further
            end_idx = start_idx + 25  # Look ahead up to 25 words
            
            # Find next key as boundary
            for j in range(i + 1, len(key_matches_sorted)):
                next_match = key_matches_sorted[j]
                if next_match["start_index"] > start_idx:
                    end_idx = min(end_idx, next_match["start_index"])
                    break
            
            # Limit maximum extraction length
            end_idx = min(end_idx, start_idx + 25)
                    
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
        
        # Get coordinates for key and value
        # Key coordinates
        key_word_infos = words_with_info[match["start_index"]:match["end_index"]]
        key_boxes = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in key_word_infos]
        key_merged_box = merge_bounding_boxes(key_boxes)
        key_page = key_word_infos[0]["page"] if key_word_infos else 0
        
        # Value coordinates
        value_word_infos = [words_with_info[idx] for idx in value_indexes if idx < len(words_with_info)]
        value_boxes = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in value_word_infos]
        value_merged_box = merge_bounding_boxes(value_boxes)
        value_page = value_word_infos[0]["page"] if value_word_infos else 0
        
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
                "cleaned_value": cleaned_value,
                "key_coordinates": {
                    "x": key_merged_box[0] if key_merged_box else 0,
                    "y": key_merged_box[1] if key_merged_box else 0,
                    "width": key_merged_box[2] - key_merged_box[0] if key_merged_box else 0,
                    "height": key_merged_box[3] - key_merged_box[1] if key_merged_box else 0,
                    "page": key_page + 1  # Convert to 1-based page numbering
                },
                "value_coordinates": {
                    "x": value_merged_box[0] if value_merged_box else 0,
                    "y": value_merged_box[1] if value_merged_box else 0,
                    "width": value_merged_box[2] - value_merged_box[0] if value_merged_box else 0,
                    "height": value_merged_box[3] - value_merged_box[1] if value_merged_box else 0,
                    "page": value_page + 1  # Convert to 1-based page numbering
                }
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
        # Special handling for debit amount - look for complete amount
        # Join all words and look for INR pattern
        combined_text = " ".join(value_words)
        
        # Look for INR amounts in the combined text
        # Pattern to match INR followed by digits, possibly with commas and decimal points
        inr_pattern = r'INR\s*[\d,]+\.?\d*'
        match = re.search(inr_pattern, combined_text)
        if match:
            return match.group(0).strip()
        
        # If no INR pattern found, look for any currency pattern
        currency_pattern = r'[A-Z]{3}\s*[\d,]+\.?\d*'
        match = re.search(currency_pattern, combined_text)
        if match:
            return match.group(0).strip()
        
        # If no currency pattern, look for numbers with possible currency symbols
        number_pattern = r'[\$£€¥]\s*[\d,]+\.?\d*'
        match = re.search(number_pattern, combined_text)
        if match:
            return match.group(0).strip()
        
        # If still nothing, look for any sequence of digits that could be an amount
        # This handles cases where the currency symbol is on a different line
        digit_pattern = r'[\d,]+\.?\d*'
        match = re.search(digit_pattern, combined_text)
        if match:
            # Check if there's a currency symbol nearby (within 3 words)
            match_start = match.start()
            match_end = match.end()
            
            # Look for currency symbol before the amount
            for i in range(max(0, match_start - 30), match_start):
                if i < len(combined_text) and combined_text[i] in ['$', '£', '€', '¥']:
                    # Found currency symbol, return the amount
                    return match.group(0).strip()
            
            # Look for currency symbol after the amount
            for i in range(match_end, min(len(combined_text), match_end + 30)):
                if i < len(combined_text) and combined_text[i] in ['$', '£', '€', '¥']:
                    # Found currency symbol, return the amount
                    return match.group(0).strip()
            
            # If no currency symbol found nearby, still return the digits
            return match.group(0).strip()
        
        # If still nothing, return the raw value
        return full_value.strip()
    
    elif label == "remittance_amount":
        # Special handling for remittance amount - look for complete amount
        # Join all words and look for INR pattern
        combined_text = " ".join(value_words)
        
        # Look for INR amounts in the combined text
        # Pattern to match INR followed by digits, possibly with commas and decimal points
        inr_pattern = r'INR\s*[\d,]+\.?\d*'
        match = re.search(inr_pattern, combined_text)
        if match:
            return match.group(0).strip()
        
        # If no INR pattern found, look for any currency pattern
        currency_pattern = r'[A-Z]{3}\s*[\d,]+\.?\d*'
        match = re.search(currency_pattern, combined_text)
        if match:
            return match.group(0).strip()
        
        # If no currency pattern, look for numbers with possible currency symbols
        number_pattern = r'[\$£€¥]\s*[\d,]+\.?\d*'
        match = re.search(number_pattern, combined_text)
        if match:
            return match.group(0).strip()
        
        # If still nothing, look for any sequence of digits that could be an amount
        # This handles cases where the currency symbol is on a different line
        digit_pattern = r'[\d,]+\.?\d*'
        match = re.search(digit_pattern, combined_text)
        if match:
            # Check if there's a currency symbol nearby (within 3 words)
            match_start = match.start()
            match_end = match.end()
            
            # Look for currency symbol before the amount
            for i in range(max(0, match_start - 30), match_start):
                if i < len(combined_text) and combined_text[i] in ['$', '£', '€', '¥']:
                    # Found currency symbol, return the amount
                    return match.group(0).strip()
            
            # Look for currency symbol after the amount
            for i in range(match_end, min(len(combined_text), match_end + 30)):
                if i < len(combined_text) and combined_text[i] in ['$', '£', '€', '¥']:
                    # Found currency symbol, return the amount
                    return match.group(0).strip()
            
            # If no currency symbol found nearby, still return the digits
            return match.group(0).strip()
        
        # If still nothing, return the raw value
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
    words = [word_info["word"] for _, word_info in indexed_words]
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
def process_document(file_path, file_type="pdf"):
    """Main function to process document and extract all entities"""
    
    # Extract text and tokenize with coordinates
    words_with_coords = extract_text_with_coordinates(file_path, file_type)
    indexed_words = tokenize_with_index(words_with_coords)
    
    # Find entity boundaries
    key_matches = find_entity_boundaries(indexed_words, HEAD_KEYS)
    
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
        "raw_text_preview": " ".join([w_info["word"] for _, w_info in indexed_words[:50]]) + "..." if len(indexed_words) > 50 else " ".join([w_info["word"] for _, w_info in indexed_words])
    }
    
    return result

# ------------------ Streamlit UI ------------------
st.set_page_config(
    page_title="Document Auto-Tagger",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Auto-Tagger with Bounding Boxes")
st.markdown("Upload a PDF or Image document to automatically extract and tag fields with bounding boxes")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'json_data' not in st.session_state:
    st.session_state.json_data = None
if 'annotated_images' not in st.session_state:
    st.session_state.annotated_images = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None

# File upload
uploaded_file = st.file_uploader("Upload PDF or Image Document", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Determine file type
    file_ext = uploaded_file.name.split(".")[-1].lower()
    file_type = "pdf" if file_ext == "pdf" else "image"
    st.session_state.file_type = file_type
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name
    
    # Process the document
    with st.spinner(f"Processing {file_type.upper()}..."):
        try:
            result = process_document(temp_file_path, file_type)
            st.session_state.results = result
            
            # Generate JSON data in the required format
            fields = []
            for label, entity_info in result["entity_details"].items():
                field = {
                    "field_name": label,
                    "trigger_word": entity_info["key_text"],
                    "coordinates": entity_info["key_coordinates"],
                    "expected_value": entity_info["cleaned_value"],
                    "value_coordinates": entity_info["value_coordinates"]
                }
                fields.append(field)
            
            st.session_state.json_data = {
                "template_name": "payment_advice",
                "fields": fields
            }
            
            # Draw bounding boxes on the document
            st.session_state.annotated_images = draw_bounding_boxes_on_document(temp_file_path, result["entity_details"], file_type)
            
            st.success(f"{file_type.upper()} processed successfully!")
        except Exception as e:
            st.error(f"Error processing {file_type}: {str(e)}")
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

# Display results if available
if st.session_state.results:
    st.header("🔍 Extracted Fields")
    
    # Display fields in a table
    fields_data = []
    for label, entity_info in st.session_state.results["entity_details"].items():
        fields_data.append({
            "Field Name": label,
            "Trigger Word": entity_info["key_text"],
            "Value": entity_info["cleaned_value"],
            "Key Position": f"Page {entity_info['key_coordinates']['page']}: ({entity_info['key_coordinates']['x']}, {entity_info['key_coordinates']['y']})",
            "Value Position": f"Page {entity_info['value_coordinates']['page']}: ({entity_info['value_coordinates']['x']}, {entity_info['value_coordinates']['y']})"
        })
    
    st.dataframe(fields_data)
    
    # Display annotated document pages
    st.header("🖼️ Annotated Document with Bounding Boxes")
    
    if st.session_state.annotated_images:
        # Create tabs for each page/image
        if st.session_state.file_type == "pdf":
            tabs = st.tabs([f"Page {i+1}" for i in range(len(st.session_state.annotated_images))])
        else:
            tabs = st.tabs(["Document"])
        
        for i, tab in enumerate(tabs):
            with tab:
                st.image(st.session_state.annotated_images[i], caption=f"{'Page ' + str(i+1) if st.session_state.file_type == 'pdf' else 'Document'} with Auto-Tagged Fields", use_column_width=True)
                
                # Download button for this page/image
                img_buffer = BytesIO()
                st.session_state.annotated_images[i].save(img_buffer, format="PNG")
                img_buffer.seek(0)
                
                if st.session_state.file_type == "pdf":
                    label = f"📥 Download Page {i+1}"
                    file_name = f"annotated_page_{i+1}.png"
                else:
                    label = "📥 Download Image"
                    file_name = "annotated_image.png"
                
                st.download_button(
                    label=label,
                    data=img_buffer,
                    file_name=file_name,
                    mime="image/png"
                )
    
    # Display JSON data
    st.header("📋 Generated JSON")
    st.json(st.session_state.json_data)
    
    # Download buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download JSON
        json_str = json.dumps(st.session_state.json_data, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="extracted_data.json",
            mime="application/json"
        )
    
    with col2:
        # Download detailed results
        detailed_json_str = json.dumps(st.session_state.results, indent=2)
        st.download_button(
            label="📥 Download Detailed Results",
            data=detailed_json_str,
            file_name="detailed_results.json",
            mime="application/json"
        )
    
    with col3:
        # Download all annotated pages/images as a ZIP file
        if st.session_state.annotated_images:
            import zipfile
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "a") as zip_file:
                for i, img in enumerate(st.session_state.annotated_images):
                    img_buffer = BytesIO()
                    img.save(img_buffer, format="PNG")
                    img_buffer.seek(0)
                    if st.session_state.file_type == "pdf":
                        zip_file.writestr(f"annotated_page_{i+1}.png", img_buffer.getvalue())
                    else:
                        zip_file.writestr("annotated_image.png", img_buffer.getvalue())
            
            zip_buffer.seek(0)
            st.download_button(
                label="📥 Download All Annotated Pages",
                data=zip_buffer,
                file_name="annotated_pages.zip",
                mime="application/zip"
            )

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a PDF or Image document using the file uploader
2. The system will automatically process the document and extract fields
3. View the extracted fields in the table below
4. See the annotated document pages with bounding boxes for all header fields
5. Download the JSON file with the extracted data
""")

# Information about the JSON format
st.sidebar.header("JSON Format")
st.sidebar.markdown("""
The generated JSON file contains:
- `template_name`: Name of the template (default: "payment_advice")
- `fields`: Array of extracted fields with:
  - `field_name`: Normalized field name
  - `trigger_word`: Original trigger word
  - `coordinates`: Position of the trigger word (x, y, width, height, page)
  - `expected_value`: Extracted value
  - `value_coordinates`: Position of the value (x, y, width, height, page)
""")

# Information about the bounding boxes
st.sidebar.header("Bounding Boxes")
st.sidebar.markdown("""
The system automatically draws bounding boxes for:
- All header fields (trigger words) in their respective colors
- All corresponding values in the same color as the trigger word
- Field names are displayed above the trigger word boxes
- Extracted values are displayed below the value boxes
Note: Only header fields and their values are annotated, not tables or important notes.
""")

