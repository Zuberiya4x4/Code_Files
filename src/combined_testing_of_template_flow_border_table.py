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
import base64

# ------------------ Configuration ------------------
DEFAULT_MAX_LEN = 20
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
    "table": (255, 128, 0)  # Orange for tables
}

# ------------------ Header Processing Functions ------------------
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

def extract_entity_values(indexed_words, key_matches):
    """Extract values for each entity using index boundaries with improved logic"""
    words_with_info = [w_info for _, w_info in indexed_words]
    words = [w_info["word"] for w_info in words_with_info]
    results = {}
    
    key_matches_sorted = sorted(key_matches, key=lambda x: x["start_index"])
    
    for i, match in enumerate(key_matches_sorted):
        start_idx = match["end_index"]
        label = match["label"]
        
        while start_idx < len(words) and words[start_idx] in [":", "-", "–", "—", "|", "•", ".", "and", "contact", "information"]:
            start_idx += 1
        
        end_idx = start_idx + DEFAULT_MAX_LEN
        
        for j in range(i + 1, len(key_matches_sorted)):
            next_match = key_matches_sorted[j]
            if next_match["start_index"] > start_idx:
                end_idx = min(end_idx, next_match["start_index"])
                break
        
        value_words = []
        value_indexes = []
        
        for j in range(start_idx, min(end_idx, len(words))):
            value_words.append(words[j])
            value_indexes.append(j)
        
        cleaned_value = clean_field_value(label, value_words)
        
        actual_key_length = len(match["raw_key"].split())
        key_word_infos = words_with_info[match["start_index"]:match["start_index"] + actual_key_length]
        key_boxes = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in key_word_infos]
        key_merged_box = merge_bounding_boxes(key_boxes)
        key_page = key_word_infos[0]["page"] if key_word_infos else 0
        
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
    """Apply field-specific cleaning rules"""
    if not value_words:
        return ""
    
    full_value = " ".join(value_words)
    
    if label == "advice_ref":
        for word in value_words:
            if word.startswith("A2") and "-IN" in word:
                return word
            elif word.startswith("A2") and len(word) > 5:
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
        
        if date_words:
            return " ".join(date_words[:3])
        
        return " ".join(value_words[:2]) if value_words else ""
    
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
            if len(amount_text) > 3:
                return amount_text
        
        return " ".join(value_words[:5]).strip()
    
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
    """Extract tables from PDF and return JSON data with bounding box coordinates"""
    pdf_data = {
        "filename": os.path.basename(file_path),
        "pages": []
    }
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_data = {
                "page_number": page_num + 1,
                "tables": []
            }
            
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
    """Extract full text from PDF using PyMuPDF"""
    full_text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if text:
                full_text += "\n" + text
    return full_text

def extract_data_using_regex(pdf_text, head_keys):
    """Extract specific details using regex patterns based on head_keys"""
    # Create dynamic patterns based on head_keys
    patterns = {}
    
    for key_text, label in head_keys.items():
        # Create a regex pattern for each key
        escaped_key = re.escape(key_text)
        # Make it more flexible by allowing variations in spacing and punctuation
        flexible_key = escaped_key.replace(r'\ ', r'\s+').replace(r"\'", r"'?")
        pattern = rf"{flexible_key}[:\s]*([^\n]+)"
        patterns[label] = pattern
    
    extracted_data = {}
    for label, pattern in patterns.items():
        match = re.search(pattern, pdf_text, re.MULTILINE | re.IGNORECASE)
        if match:
            extracted_data[label] = match.group(1).strip()
        else:
            extracted_data[label] = ""
    
    # Clean up extracted data
    for key, value in extracted_data.items():
        if value:
            extracted_data[key] = " ".join(value.split())
        else:
            extracted_data[key] = "Not Found"
    
    return extracted_data

# ------------------ Drawing Functions ------------------
def draw_bounding_boxes_on_document(file_path, entity_details, table_data, file_type="pdf"):
    """Draw bounding boxes for headers and tables on document pages"""
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
    """Extract head keys from JSON file"""
    head_keys = {}
    
    # Check if JSON has head_keys directly
    if "head_keys" in json_data:
        head_keys = json_data["head_keys"]
    
    # Check for your specific JSON structure
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
        # If it's a simple key-value mapping
        head_keys = json_data
    
    # If still empty, try to infer from any list of objects
    if not head_keys and isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict) and "trigger" in item and "field" in item:
                head_keys[item["trigger"]] = item["field"]
    
    return head_keys

# ------------------ Main Processing Function ------------------
def process_complete_document(file_path, json_coordinates=None, file_type="pdf"):
    """Main function to process document and extract all entities and tables"""
    
    # Extract head keys from JSON
    head_keys = {}
    if json_coordinates:
        head_keys = extract_head_keys_from_json(json_coordinates)
    
    # Debug: Show extracted head keys
    st.write("### Extracted Head Keys")
    st.json(head_keys)
    
    if not head_keys:
        st.error("No head keys found in JSON file. Please check the JSON structure.")
        st.write("### JSON Structure Received:")
        st.json(json_coordinates)
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

# ------------------ Streamlit UI ------------------
st.set_page_config(
    page_title="Complete Document Auto-Tagger",
    page_icon="📄",
    layout="wide"
)
st.title("📄 Complete Document Auto-Tagger")
st.markdown("Upload a JSON coordinates file and PDF document to automatically extract headers and tables with bounding boxes")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'annotated_images' not in st.session_state:
    st.session_state.annotated_images = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None

# Create two columns for file upload
col1, col2 = st.columns(2)
with col1:
    json_file = st.file_uploader(
        "Upload JSON Coordinates File", 
        type="json",
        help="Upload JSON file containing coordinate data for headers"
    )
with col2:
    pdf_file = st.file_uploader(
        "Upload PDF Document", 
        type=["pdf"],
        help="Upload PDF document to process"
    )

# Process files when both are uploaded
if json_file and pdf_file:
    st.subheader(f"Processing: {pdf_file.name}")
    
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_file.read())
        temp_pdf_path = tmp_pdf.name
    
    # Read JSON coordinates
    try:
        coordinates = json.load(json_file)
        st.write("### Loaded JSON Configuration")
        st.json(coordinates)
        st.session_state.file_type = "pdf"
        
        # Process the document
        with st.spinner("Processing PDF..."):
            try:
                result = process_complete_document(temp_pdf_path, coordinates, "pdf")
                if result is None:
                    st.stop()  # Stop processing if result is None
                    
                st.session_state.results = result
                
                # Draw bounding boxes on the document
                st.session_state.annotated_images = draw_bounding_boxes_on_document(
                    temp_pdf_path, 
                    result["entity_details"], 
                    result["table_data"], 
                    "pdf"
                )
                
                st.success("PDF processed successfully!")
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.exception(e)
        
    except Exception as e:
        st.error(f"Error reading JSON file: {str(e)}")
        st.exception(e)
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)

# Display results if available
if st.session_state.results:
    result = st.session_state.results
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Header Fields", "📊 Tables", "🖼️ Annotated Document", "📥 Downloads"])
    
    with tab1:
        st.header("🔍 Extracted Header Fields")
        
        # Create sub-tabs for coordinate-based and regex-based extraction
        coord_tab, regex_tab = st.tabs(["Coordinate-based Extraction", "Regex-based Extraction"])
        
        with coord_tab:
            st.subheader("Coordinate-based Header Extraction")
            
            # Display fields in a table
            fields_data = []
            for label, entity_info in result["entity_details"].items():
                fields_data.append({
                    "Field Name": label,
                    "Trigger Word": entity_info["key_text"],
                    "Value": entity_info["cleaned_value"],
                    "Key Position": f"Page {entity_info['key_coordinates']['page']}: ({entity_info['key_coordinates']['x']:.1f}, {entity_info['key_coordinates']['y']:.1f})",
                    "Value Position": f"Page {entity_info['value_coordinates']['page']}: ({entity_info['value_coordinates']['x']:.1f}, {entity_info['value_coordinates']['y']:.1f})"
                })
            
            if fields_data:
                st.dataframe(fields_data, use_container_width=True)
            else:
                st.info("No header fields found using coordinate-based extraction")
        
        with regex_tab:
            st.subheader("Regex-based Header Extraction")
            
            # Display regex results
            regex_data = []
            for key, value in result["regex_based_entities"].items():
                regex_data.append({
                    "Field Name": key,
                    "Extracted Value": value if value else "Not Found"
                })
            
            if regex_data:
                st.dataframe(regex_data, use_container_width=True)
            else:
                st.info("No header fields found using regex-based extraction")
    
    with tab2:
        st.header("📊 Extracted Tables")
        
        table_data = result["table_data"]
        total_tables = sum(len(page["tables"]) for page in table_data["pages"])
        
        if total_tables > 0:
            # Display summary
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Pages", len(table_data["pages"]))
            col2.metric("Total Tables", total_tables)
            col3.metric("Pages with Tables", sum(1 for page in table_data["pages"] if page["tables"]))
            
            # Display tables by page
            for page in table_data["pages"]:
                if page["tables"]:
                    st.subheader(f"Page {page['page_number']}")
                    
                    for table in page["tables"]:
                        st.markdown(f"**Table {table['table_index']}**")
                        
                        # Display table coordinates
                        bbox = table["bbox"]
                        st.markdown(f"Coordinates: ({bbox['x0']}, {bbox['top']}) to ({bbox['x1']}, {bbox['bottom']})")
                        
                        # Display table data
                        if table["data"]:
                            try:
                                # Convert table data to DataFrame for better display
                                import pandas as pd
                                df = pd.DataFrame(table["data"][1:], columns=table["data"][0] if table["data"] else [])
                                st.dataframe(df, use_container_width=True)
                            except Exception as e:
                                st.json(table["data"])
                        else:
                            st.info("No data extracted from this table")
                        
                        st.markdown("---")
        else:
            st.info("No tables found in the document")
    
    with tab3:
        st.header("🖼️ Annotated Document with Bounding Boxes")
        
        if st.session_state.annotated_images:
            # Create legend
            st.subheader("Color Legend")
            legend_cols = st.columns(4)
            
            # Header fields legend
            with legend_cols[0]:
                st.markdown("**Header Fields:**")
                for field, color in list(FIELD_COLORS.items())[:9]:
                    if field != "table":
                        st.markdown(f'<div style="display: flex; align-items: center;"><div style="width: 20px; height: 15px; background-color: rgb{color}; margin-right: 10px;"></div>{field.replace("_", " ").title()}</div>', unsafe_allow_html=True)
            
            with legend_cols[1]:
                st.markdown("**More Fields:**")
                for field, color in list(FIELD_COLORS.items())[9:18]:
                    if field != "table":
                        st.markdown(f'<div style="display: flex; align-items: center;"><div style="width: 20px; height: 15px; background-color: rgb{color}; margin-right: 10px;"></div>{field.replace("_", " ").title()}</div>', unsafe_allow_html=True)
            
            with legend_cols[2]:
                st.markdown("**Tables:**")
                table_color = FIELD_COLORS["table"]
                st.markdown(f'<div style="display: flex; align-items: center;"><div style="width: 20px; height: 15px; background-color: rgb{table_color}; margin-right: 10px;"></div>Table</div>', unsafe_allow_html=True)
            
            with legend_cols[3]:
                st.markdown("**Legend:**")
                st.markdown("• **Thick border**: Header field triggers")
                st.markdown("• **Thin border**: Header field values")
                st.markdown("• **Table border**: Complete tables")
            
            # Display annotated pages
            tabs = st.tabs([f"Page {i+1}" for i in range(len(st.session_state.annotated_images))])
            
            for i, tab in enumerate(tabs):
                with tab:
                    st.image(
                        st.session_state.annotated_images[i], 
                        caption=f"Page {i+1} with Auto-Tagged Fields and Tables", 
                        use_column_width=True
                    )
                    
                    # Download button for this page
                    img_buffer = BytesIO()
                    st.session_state.annotated_images[i].save(img_buffer, format="PNG")
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label=f"📥 Download Page {i+1}",
                        data=img_buffer,
                        file_name=f"annotated_page_{i+1}.png",
                        mime="image/png"
                    )
        else:
            st.info("No annotated images available")
    
    with tab4:
        st.header("📥 Download Results")
        
        # Prepare different JSON outputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Header Fields JSON")
            
            # Combined header results
            combined_headers = {
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
            
            header_json_str = json.dumps(combined_headers, indent=2)
            st.download_button(
                label="📥 Download Header Fields",
                data=header_json_str,
                file_name=f"header_fields_{pdf_file.name.split('.')[0]}.json",
                mime="application/json"
            )
        
        with col2:
            st.subheader("Tables JSON")
            
            # Clean table data (remove any extra fields)
            clean_table_data = {
                "filename": result["table_data"]["filename"],
                "pages": []
            }
            
            for page in result["table_data"]["pages"]:
                clean_page = {
                    "page_number": page["page_number"],
                    "tables": page["tables"]
                }
                clean_table_data["pages"].append(clean_page)
            
            table_json_str = json.dumps(clean_table_data, indent=2)
            st.download_button(
                label="📥 Download Tables",
                data=table_json_str,
                file_name=f"tables_{pdf_file.name.split('.')[0]}.json",
                mime="application/json"
            )
        
        with col3:
            st.subheader("Complete Results")
            
            # Complete results without image data
            complete_results = {
                "document_name": pdf_file.name,
                "header_fields": combined_headers,
                "tables": clean_table_data,
                "processing_summary": {
                    "total_header_fields_found": len(result["entity_details"]),
                    "total_tables_found": sum(len(page["tables"]) for page in result["table_data"]["pages"]),
                    "total_pages": len(result["table_data"]["pages"])
                }
            }
            
            complete_json_str = json.dumps(complete_results, indent=2)
            st.download_button(
                label="📥 Download Complete Results",
                data=complete_json_str,
                file_name=f"complete_results_{pdf_file.name.split('.')[0]}.json",
                mime="application/json"
            )
        
        # Download all annotated pages as ZIP
        if st.session_state.annotated_images:
            st.subheader("Annotated Images")
            
            import zipfile
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "a") as zip_file:
                for i, img in enumerate(st.session_state.annotated_images):
                    img_buffer = BytesIO()
                    img.save(img_buffer, format="PNG")
                    img_buffer.seek(0)
                    zip_file.writestr(f"annotated_page_{i+1}.png", img_buffer.getvalue())
            
            zip_buffer.seek(0)
            st.download_button(
                label="📥 Download All Annotated Pages (ZIP)",
                data=zip_buffer,
                file_name=f"annotated_pages_{pdf_file.name.split('.')[0]}.zip",
                mime="application/zip"
            )
else:
    st.info("Please upload both JSON coordinates file and PDF document to begin processing")

# Instructions sidebar
st.sidebar.header("📖 Instructions")
st.sidebar.markdown("""
**Step 1: Upload Files**
1. Upload a JSON file containing coordinate data for header fields
2. Upload a PDF document to process
**Step 2: Processing**
- The system will automatically extract header fields using both coordinate-based and regex-based methods
- Tables will be detected and extracted with their coordinates
- Bounding boxes will be drawn for all detected elements
**Step 3: Review Results**
- View extracted header fields in the "Header Fields" tab
- Review detected tables in the "Tables" tab
- See annotated document with bounding boxes in the "Annotated Document" tab
- Download results in various formats from the "Downloads" tab
**Features:**
- Dual extraction methods for headers (coordinate + regex)
- Automatic table detection and extraction
- Visual bounding boxes for all elements
- Multiple download formats
- Color-coded field types
""")

st.sidebar.header("🎨 Bounding Box Colors")
st.sidebar.markdown("""
**Header Fields:**
- Each field type has a unique color
- Thick borders: Field labels/triggers
- Thin borders: Field values
**Tables:**
- Orange borders for table boundaries
- Automatic table numbering
**Legend:**
- Field names shown above trigger words
- Extracted values shown below value areas
""")

st.sidebar.header("📊 JSON Output Formats")
st.sidebar.markdown("""
**Header Fields JSON:**
- Coordinate-based extraction results
- Regex-based extraction results
- Detailed coordinate information
**Tables JSON:**
- Table data with coordinates
- Structured table content
- Page-wise organization
**Complete Results:**
- Combined header and table data
- Processing summary
- Document metadata
""")