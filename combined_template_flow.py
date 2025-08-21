# template_executor.py

import fitz  # PyMuPDF
import pdfplumber
import re
import os
import json
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Configuration
DEFAULT_MAX_LEN = 20
FIELD_COLORS = {
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
    "table": (255, 128, 0)
}

# ------------------ Header Processing Functions ------------------
def extract_text_with_coordinates(file_path, file_type="pdf"):
    words_with_coords = []
    
    if file_type == "pdf":
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            words = page.get_text("words")
            for word_info in words:
                x0, y0, x1, y1, word, block_no, line_no, word_no = word_info
                words_with_coords.append({
                    "word": word, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "page": page_num, "block": block_no, "line": line_no, "word_no": word_no
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
                    "word": word, "x0": x, "y0": y, "x1": x + w, "y1": y + h,
                    "page": 0, "block": ocr_data['block_num'][i],
                    "line": ocr_data['line_num'][i], "word_no": ocr_data['word_num'][i]
                })
    
    return words_with_coords

def tokenize_with_index(words_with_coords):
    return [(i, word_info) for i, word_info in enumerate(words_with_coords)]

def normalize(word):
    return "".join(c for c in word if c.isalnum()).lower()

def create_normalized_key_map(head_keys):
    normalized_map = {}
    for raw_key, label in head_keys.items():
        normalized_words = [normalize(w) for w in raw_key.split()]
        normalized_key = " ".join(normalized_words)
        normalized_map[normalized_key] = {
            "raw_key": raw_key, "label": label, "word_count": len(normalized_words)
        }
    return normalized_map

def merge_bounding_boxes(boxes):
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return (x0, y0, x1, y1)

def find_entity_boundaries(indexed_words, head_keys):
    words = [w_info["word"] for _, w_info in indexed_words]
    normalized_key_map = create_normalized_key_map(head_keys)
    key_matches = []
    
    # Special handling for recipient field
    recipient_phrase = "Recipient's name and contact information"
    recipient_words = recipient_phrase.split()
    recipient_word_count = len(recipient_words)
    
    for i in range(len(words) - recipient_word_count + 1):
        if words[i].lower() in ["recipient's", "recipient"]:
            match = True
            for j in range(1, recipient_word_count):
                if i + j >= len(words) or normalize(words[i + j]) != normalize(recipient_words[j]):
                    match = False
                    break
            
            if match:
                key_matches.append({
                    "start_index": i, "end_index": i + recipient_word_count,
                    "raw_key": recipient_phrase, "label": "receipt_name",
                    "normalized_key": "recipient s name and contact information",
                    "actual_words": " ".join(words[i:i + recipient_word_count])
                })
                i += recipient_word_count - 1
    
    # Standard processing for other keys
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
                        matched_key = {"raw_key": raw_key, "label": label, "word_count": len(raw_key.split())}
                        break
            
            if matched_key:
                key_info = matched_key
                
                # Skip if this is a recipient phrase we already matched
                if key_info["label"] == "receipt_name" and any(
                    m["label"] == "receipt_name" and m["start_index"] <= i < m["end_index"]
                    for m in key_matches
                ):
                    continue
                
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
                        "start_index": i, "end_index": i + min(key_length, key_info["word_count"]),
                        "raw_key": key_info["raw_key"], "label": key_info["label"],
                        "normalized_key": normalized_phrase,
                        "actual_words": " ".join(words[i:i + key_length])
                    })
                break
    
    key_matches.sort(key=lambda x: x["start_index"])
    return key_matches

def extract_entity_values(indexed_words, key_matches):
    words_with_info = [w_info for _, w_info in indexed_words]
    words = [w_info["word"] for w_info in words_with_info]
    results = {}
    
    key_matches_sorted = sorted(key_matches, key=lambda x: x["start_index"])
    
    for i, match in enumerate(key_matches_sorted):
        start_idx = match["end_index"]
        label = match["label"]
        
        # Skip punctuation and connector words
        while start_idx < len(words) and words[start_idx] in [":", "-", "–", "—", "|", "•", ".", "and", "contact", "information"]:
            start_idx += 1
        
        end_idx = start_idx + DEFAULT_MAX_LEN
        
        # Find next key to limit value extraction
        for j in range(i + 1, len(key_matches_sorted)):
            next_match = key_matches_sorted[j]
            if next_match["start_index"] > start_idx:
                end_idx = min(end_idx, next_match["start_index"])
                break
        
        value_words = []
        value_indexes = []
        
        # Special handling for recipient field
        if label == "receipt_name":
            for j in range(start_idx, min(end_idx, len(words))):
                if (j > start_idx + 2 and 
                    any(header in words[j].lower() for header in ["date", "ref", "amount", "bank", "account"])):
                    break
                value_words.append(words[j])
                value_indexes.append(j)
        else:
            for j in range(start_idx, min(end_idx, len(words))):
                value_words.append(words[j])
                value_indexes.append(j)
        
        cleaned_value = clean_field_value(label, value_words)
        
        # Get key coordinates
        actual_key_length = len(match["raw_key"].split())
        key_word_infos = words_with_info[match["start_index"]:match["start_index"] + actual_key_length]
        key_boxes = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in key_word_infos]
        key_merged_box = merge_bounding_boxes(key_boxes)
        key_page = key_word_infos[0]["page"] if key_word_infos else 0
        
        # Get value coordinates
        cleaned_value_words = cleaned_value.split() if cleaned_value else []
        actual_value_indexes = []
        
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
    
    return results

def clean_field_value(label, value_words):
    if not value_words:
        return ""
    
    if label == "advice_ref":
        for word in value_words:
            if word.startswith("A2") and ("-IN" in word or len(word) > 5):
                return word
        for word in value_words:
            if len(word) > 5 and any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                return word
        for word in value_words:
            if len(word) > 2 and word not in [":", "-", "–", "—", "|", "•"]:
                return word
        return value_words[0] if value_words else ""
    
    elif label == "advice_date":
        date_words = []
        for word in value_words:
            if any(c.isdigit() for c in word) or word in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
                date_words.append(word)
            elif len(date_words) > 0:
                break
        return " ".join(date_words[:3]) if date_words else " ".join(value_words[:2]) if value_words else ""
    
    elif label in ["debit_amount", "remittance_amount"]:
        combined_text = " ".join(value_words)
        inr_pattern = r'INR\s*[\d,]+\.?\d*'
        match = re.search(inr_pattern, combined_text)
        if match:
            return match.group(0).strip()
        
        currency_pattern = r'[A-Z]{3}\s*[\d,]+\.?\d*'
        match = re.search(currency_pattern, combined_text)
        if match:
            return match.group(0).strip()
        
        number_pattern = r'[\$£€¥]\s*[\d,]+\.?\d*'
        match = re.search(number_pattern, combined_text)
        if match:
            return match.group(0).strip()
        
        digit_pattern = r'[\d,]+\.?\d*'
        match = re.search(digit_pattern, combined_text)
        if match:
            amount_text = match.group(0).strip()
            return amount_text if len(amount_text) > 3 else ""
        
        return " ".join(value_words[:5]).strip()
    
    elif label == "receipt_name":
        clean_words = [word for word in value_words if word not in [":", "-", "–", "—", "|", "•"] and len(word) > 0]
        return " ".join(clean_words).strip()
    
    else:
        clean_words = []
        for word in value_words:
            if word not in [":", "-", "–", "—", "|", "•"] and len(word) > 0:
                clean_words.append(word)
            if len(clean_words) >= 5:
                break
        return " ".join(clean_words).strip()

# ------------------ Table Processing Functions ------------------
def extract_tables_to_json(file_path):
    pdf_data = {"filename": os.path.basename(file_path), "pages": []}
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_data = {"page_number": page_num + 1, "tables": []}
            tables = page.find_tables()
            
            if tables:
                for table_idx, table in enumerate(tables):
                    table_data = table.extract()
                    table_entry = {
                        "table_index": table_idx + 1,
                        "bbox": {
                            "x0": round(table.bbox[0], 2),
                            "top": round(table.bbox[1], 2),
                            "x1": round(table.bbox[2], 2),
                            "bottom": round(table.bbox[3], 2)
                        },
                        "data": table_data
                    }
                    page_data["tables"].append(table_entry)
            
            pdf_data["pages"].append(page_data)
    
    return pdf_data

# ------------------ Regex-based Header Extraction ------------------
def extract_text_from_pdf(file_path):
    full_text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if text:
                full_text += "\n" + text
    return full_text

def extract_data_using_regex(pdf_text, head_keys):
    patterns = {}
    for key_text, label in head_keys.items():
        escaped_key = re.escape(key_text)
        flexible_key = escaped_key.replace(r'\ ', r'\s+').replace(r"\'", r"'?")
        pattern = rf"{flexible_key}[:\s]*([^\n]+)"
        patterns[label] = pattern
    
    extracted_data = {}
    for label, pattern in patterns.items():
        match = re.search(pattern, pdf_text, re.MULTILINE | re.IGNORECASE)
        extracted_data[label] = match.group(1).strip() if match else ""
    
    # Clean up extracted data
    for key, value in extracted_data.items():
        extracted_data[key] = " ".join(value.split()) if value else "Not Found"
    
    return extracted_data

# ------------------ Drawing Functions ------------------
def draw_bounding_boxes_on_document(file_path, entity_details, table_data, file_type="pdf"):
    annotated_images = []
    
    if file_type == "pdf":
        doc = fitz.open(file_path)
        
        # Group entities by page
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
                    "value_coords": info.get("value_coordinates", {})
                })
        
        # Group tables by page
        tables_by_page = {}
        for page_data in table_data["pages"]:
            page_num = page_data["page_number"] - 1
            tables_by_page[page_num] = page_data["tables"]
        
        # Process each page
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
            
            # Draw header fields
            if page_num in entities_by_page:
                for entity in entities_by_page[page_num]:
                    label = entity["label"]
                    key_coords = entity["key_coords"]
                    value_coords = entity["value_coords"]
                    color = FIELD_COLORS.get(label, (255, 0, 0))
                    
                    # Draw key bounding box
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
                    
                    # Draw value bounding box
                    if value_coords and value_coords.get("width", 0) > 0 and value_coords.get("height", 0) > 0:
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
            
            # Draw tables
            if page_num in tables_by_page:
                for table in tables_by_page[page_num]:
                    bbox = table["bbox"]
                    color = FIELD_COLORS["table"]
                    
                    # Scale coordinates for 2x image
                    x0 = bbox["x0"] * 2
                    y0 = bbox["top"] * 2
                    x1 = bbox["x1"] * 2
                    y1 = bbox["bottom"] * 2
                    
                    # Draw table bounding box
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                    
                    # Draw table label
                    table_label = f"Table {table['table_index']}"
                    text_bbox = draw.textbbox((0, 0), table_label, font=label_font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    draw.rectangle(
                        [x0, y0 - text_height - 4, x0 + text_width + 8, y0 - 4],
                        fill=color
                    )
                    draw.text((x0 + 4, y0 - text_height - 4), table_label, fill="white", font=label_font)
            
            annotated_images.append(img)
        
        doc.close()
    
    return annotated_images

# ------------------ JSON Key Extraction Function ------------------
def extract_head_keys_from_json(json_data):
    head_keys = {}
    
    # Check if JSON has head_keys directly
    if "head_keys" in json_data:
        head_keys = json_data["head_keys"]
    
    # Check for specific JSON structure
    elif "fields" in json_data and isinstance(json_data["fields"], list):
        for field in json_data["fields"]:
            if "trigger_word" in field and "field_name" in field:
                head_keys[field["trigger_word"]] = field["field_name"]
    
    # Check if JSON has a different structure (key and label)
    elif "fields" in json_data:
        for field in json_data["fields"]:
            if "key" in field and "label" in field:
                head_keys[field["key"]] = field["label"]
    
    # Check if JSON has key-value pairs directly
    elif isinstance(json_data, dict):
        head_keys = json_data
    
    # If still empty, try to infer from any list of objects
    if not head_keys and isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict) and "trigger" in item and "field" in item:
                head_keys[item["trigger"]] = item["field"]
    
    return head_keys

# ------------------ Main Processing Function ------------------
def process_complete_document(file_path, json_coordinates=None, file_type="pdf"):
    # Extract head keys from JSON
    head_keys = extract_head_keys_from_json(json_coordinates) if json_coordinates else {}
    
    if not head_keys:
        return None
    
    # Extract headers using coordinate-based approach
    words_with_coords = extract_text_with_coordinates(file_path, file_type)
    indexed_words = tokenize_with_index(words_with_coords)
    
    key_matches = find_entity_boundaries(indexed_words, head_keys)
    entities = extract_entity_values(indexed_words, key_matches)
    
    # Extract headers using regex approach
    pdf_text = extract_text_from_pdf(file_path)
    regex_entities = extract_data_using_regex(pdf_text, head_keys)
    
    # Extract tables
    table_data = extract_tables_to_json(file_path)
    
    # Prepare final result
    result = {
        "head_keys_used": head_keys,
        "coordinate_based_entities": {label: info["cleaned_value"] for label, info in entities.items()},
        "regex_based_entities": regex_entities,
        "entity_details": entities,
        "table_data": table_data,
        "raw_text_preview": " ".join([w_info["word"] for _, w_info in indexed_words[:50]]) + "..." if len(indexed_words) > 50 else " ".join([w_info["word"] for _, w_info in indexed_words])
    }
    
    return result

# ------------------ Output Generation Function ------------------
def create_complete_output(result, base_filename):
    # Prepare header fields with coordinates
    header_fields = {
        "coordinate_based": result["coordinate_based_entities"],
        "regex_based": result["regex_based_entities"],
        "detailed_coordinates": {
            label: {
                "value": info["cleaned_value"],
                "key_coordinates": info["key_coordinates"],
                "value_coordinates": info["value_coordinates"]
            }
            for label, info in result["entity_details"].items()
        }
    }
    
    # Prepare table data
    table_data = result["table_data"]
    
    # Create complete output structure
    complete_output = {
        "document_name": base_filename,
        "header_fields": header_fields,
        "tables": table_data,
        "processing_summary": {
            "total_header_fields_found": len(result["entity_details"]),
            "total_tables_found": sum(len(page["tables"]) for page in table_data["pages"]),
            "total_pages": len(table_data["pages"])
        }
    }
    
    return complete_output