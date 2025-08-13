# code works for both pdf and images with auto field separation
import streamlit as st
import pandas as pd
import pytesseract
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json
import os
import re
from streamlit_drawable_canvas import st_canvas
import numpy as np
# Tesseract Path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
st.set_page_config(page_title="OCR Tagger", layout="wide")
# ---------------- Predefined Trigger Words ----------------
TRIGGER_WORDS = [
    "Advice sending date:",
    "Advice reference no:",
    "Recipient's name and contact information:",
    "Transaction type:",
    "Sub payment type:",
    "Beneficiary's name:",
    "Beneficiary's bank:",
    "Beneficiary's account:",
    "Customer reference:",
    "Debit amount:",
    "Remittance amount:",
    "Handling fee of remitting bank:",
    "Value date:",
    "Remitter's name:",
    "Remitting bank:",
    "Instruction reference:",
    "Other reference:",
    "Remitter to beneficiary information:"
]
# ---------------- State Initialization ----------------
def init_session_state():
    """Initialize all session state variables"""
    if 'trigger_fields' not in st.session_state:
        st.session_state.trigger_fields = []
    if 'selected_page_index' not in st.session_state:
        st.session_state.selected_page_index = 0
    if 'highlight_box_idx' not in st.session_state:
        st.session_state.highlight_box_idx = 0
    if 'update_flag' not in st.session_state:
        st.session_state.update_flag = False
    if 'zoom_level' not in st.session_state:
        st.session_state.zoom_level = 100
    if 'image_hash' not in st.session_state:
        st.session_state.image_hash = None
    if 'template_name' not in st.session_state:
        st.session_state.template_name = "payment_advice"
    if 'is_pdf' not in st.session_state:
        st.session_state.is_pdf = False

# Initialize session state
init_session_state()

# ---------------- Helpers ----------------
def load_font(size=18):
    try:
        return ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", size)
    except:
        return ImageFont.load_default()

def normalize_field_name(trigger_word):
    """Convert trigger word to a normalized field name"""
    # Remove special characters and colons
    normalized = re.sub(r'[^\w\s]', '', trigger_word)
    # Convert to lowercase
    normalized = normalized.lower()
    # Replace spaces with underscores
    normalized = normalized.replace(' ', '_')
    # Remove trailing underscores if any
    normalized = normalized.rstrip('_')
    return normalized

def draw_text(draw, position, text, font, text_color="white", bg_color="black"):
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=bg_color)
    draw.text(position, text, fill=text_color, font=font)

def separate_trigger_and_value(image, x, y, w, h, trigger_words):
    """
    Automatically separate trigger word and value from a bounding box
    Returns: (trigger_word, trigger_coords, value, value_coords, field_name)
    """
    # Extract the full text from the bounding box
    cropped = image.crop((x, y, x + w, y + h)).convert("L")
    full_text = pytesseract.image_to_string(cropped, config="--psm 6").strip()
    
    # Try to find the trigger word in the text
    best_trigger = None
    best_match_pos = -1
    best_match_length = 0
    
    # Create a list to store all potential matches
    matches = []
    
    for trigger in trigger_words:
        # Clean trigger word for comparison
        clean_trigger = re.sub(r'[^\w\s]', '', trigger).lower()
        clean_text = re.sub(r'[^\w\s]', '', full_text).lower()
        
        # Use regex to match the entire trigger phrase as a sequence of words
        # This ensures we match the full phrase, not just a substring
        pattern = r'\b' + re.sub(r'\s+', r'\\s+', re.escape(clean_trigger)) + r'\b'
        if re.search(pattern, clean_text):
            pos = clean_text.find(clean_trigger)
            # Store the match with its position and length
            matches.append({
                'trigger': trigger,
                'position': pos,
                'length': len(clean_trigger)
            })
    
    # Sort matches by position (earlier first) and then by length (longer first)
    if matches:
        matches.sort(key=lambda m: (m['position'], -m['length']))
        best_trigger = matches[0]['trigger']
        best_match_pos = matches[0]['position']
    
    if not best_trigger:
        # If no predefined trigger found, try to split by common separators
        separators = [':', '-', '=', '\t']
        for sep in separators:
            if sep in full_text:
                parts = full_text.split(sep, 1)
                if len(parts) == 2:
                    best_trigger = parts[0].strip()
                    break
        
        if not best_trigger:
            # Use the first few words as trigger
            words = full_text.split()
            if len(words) > 1:
                best_trigger = ' '.join(words[:min(3, len(words)//2)])
            else:
                best_trigger = full_text
    
    # Now try to separate the trigger and value spatially
    # Get word-level OCR data
    ocr_data = pytesseract.image_to_data(cropped, output_type=pytesseract.Output.DICT)
    
    # Filter out empty text
    word_data = []
    for i in range(len(ocr_data['text'])):
        if int(ocr_data['conf'][i]) > 30 and ocr_data['text'][i].strip():
            word_data.append({
                'text': ocr_data['text'][i],
                'x': ocr_data['left'][i],
                'y': ocr_data['top'][i],
                'w': ocr_data['width'][i],
                'h': ocr_data['height'][i]
            })
    
    if not word_data:
        # Fallback: split the box in half
        trigger_coords = {"x": x, "y": y, "width": w//2, "height": h}
        value_coords = {"x": x + w//2, "y": y, "width": w//2, "height": h}
        
        # Extract text from each half
        trigger_crop = image.crop((x, y, x + w//2, y + h)).convert("L")
        value_crop = image.crop((x + w//2, y, x + w, y + h)).convert("L")
        
        trigger_text = pytesseract.image_to_string(trigger_crop, config="--psm 6").strip()
        value_text = pytesseract.image_to_string(value_crop, config="--psm 6").strip()
        
        if not trigger_text:
            trigger_text = best_trigger
        
        return (trigger_text, trigger_coords, value_text, value_coords, normalize_field_name(trigger_text))
    
    # Find trigger words in the OCR data
    trigger_words_in_text = []
    value_words = []
    
    # Clean trigger for comparison
    clean_trigger = re.sub(r'[^\w\s]', '', best_trigger).lower().split()
    
    # Try to match the entire trigger phrase consecutively
    matched_indices = set()
    if len(clean_trigger) > 0:
        # We'll look for consecutive words that match the trigger phrase
        for i in range(len(word_data) - len(clean_trigger) + 1):
            # Check if the next len(clean_trigger) words match the trigger phrase
            match = True
            for j in range(len(clean_trigger)):
                idx = i + j
                clean_word = re.sub(r'[^\w\s]', '', word_data[idx]['text']).lower()
                # Check if the trigger word is contained in the OCR word or vice versa
                if not (clean_trigger[j] in clean_word or clean_word in clean_trigger[j]):
                    match = False
                    break
            if match:
                # Mark these indices as trigger
                for j in range(len(clean_trigger)):
                    matched_indices.add(i + j)
                # Also, add these words to trigger_words_in_text
                for j in range(len(clean_trigger)):
                    trigger_words_in_text.append(word_data[i+j])
                break   # Only take the first match
    # If we didn't find a consecutive match, fall back to individual word matching
    if not matched_indices:
        for i, word in enumerate(word_data):
            clean_word = re.sub(r'[^\w\s]', '', word['text']).lower()
            if any(trigger_word in clean_word or clean_word in trigger_word for trigger_word in clean_trigger):
                matched_indices.add(i)
                trigger_words_in_text.append(word)
    
    # Remaining words are likely values
    for i, word in enumerate(word_data):
        if i not in matched_indices:
            value_words.append(word)
    
    # Calculate bounding boxes
    if trigger_words_in_text:
        # Trigger coordinates
        trigger_x_min = min(word['x'] for word in trigger_words_in_text)
        trigger_y_min = min(word['y'] for word in trigger_words_in_text)
        trigger_x_max = max(word['x'] + word['w'] for word in trigger_words_in_text)
        trigger_y_max = max(word['y'] + word['h'] for word in trigger_words_in_text)
        
        trigger_coords = {
            "x": x + trigger_x_min,
            "y": y + trigger_y_min,
            "width": trigger_x_max - trigger_x_min,
            "height": trigger_y_max - trigger_y_min
        }
        
        trigger_text = best_trigger
    else:
        # Use the original trigger
        trigger_coords = {"x": x, "y": y, "width": w//3, "height": h}
        trigger_text = best_trigger
    
    if value_words:
        # Value coordinates
        value_x_min = min(word['x'] for word in value_words)
        value_y_min = min(word['y'] for word in value_words)
        value_x_max = max(word['x'] + word['w'] for word in value_words)
        value_y_max = max(word['y'] + word['h'] for word in value_words)
        
        value_coords = {
            "x": x + value_x_min,
            "y": y + value_y_min,
            "width": value_x_max - value_x_min,
            "height": value_y_max - value_y_min
        }
        
        # Extract value text - join all value words with spaces
        value_text = ' '.join(word['text'] for word in value_words)
    else:
        # Fallback: use remaining space
        trigger_width = trigger_coords["width"]
        value_coords = {
            "x": x + trigger_width + 5,
            "y": y,
            "width": w - trigger_width - 5,
            "height": h
        }
        
        # Extract remaining text as value
        # Remove the first occurrence of the trigger text (case-insensitive)
        pattern = re.compile(re.escape(trigger_text), re.IGNORECASE)
        remaining_text = pattern.sub('', full_text, count=1).strip()
        # Remove common separators from the beginning
        remaining_text = re.sub(r'^[:\-=\s]+', '', remaining_text)
        value_text = remaining_text
    
    field_name = normalize_field_name(trigger_text)
    
    return (trigger_text, trigger_coords, value_text, value_coords, field_name)

def update_annotated_image(image, trigger_fields, stroke_color, stroke_width, font, highlight_idx=-1):
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # Draw trigger fields
    for i, field in enumerate(trigger_fields):
        if field["trigger_coordinates"]:
            coords = field["trigger_coordinates"]
            x, y, w, h = coords["x"], coords["y"], coords["width"], coords["height"]
            color = "#00FF00" if i == highlight_idx else stroke_color
            draw.rectangle([x, y, x + w, y + h], outline=color, width=stroke_width)
            draw_text(draw, (x, y - 40), f"trigger: {field['trigger_word']}", font, bg_color="#6a1b9a")
        
        if field["value_coordinates"]:
            coords = field["value_coordinates"]
            x, y, w, h = coords["x"], coords["y"], coords["width"], coords["height"]
            color = "#FF9900"  # Orange color for value boxes
            draw.rectangle([x, y, x + w, y + h], outline=color, width=stroke_width)
            draw_text(draw, (x, y - 25), f"value: {field['value']}", font, bg_color="#1976d2")
    
    return annotated_img

def find_existing_field(trigger_fields, trigger_word):
    """Find an existing field by trigger word"""
    for i, field in enumerate(trigger_fields):
        if field["trigger_word"] == trigger_word:
            return i
    return -1

# ---------------- Main Title ----------------
st.title("🖼️ OCR Annotator with Auto Field Separation")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
if uploaded_file:
    # Ensure session state is initialized
    init_session_state()
    
    file_ext = uploaded_file.name.split(".")[-1].lower()
    is_pdf = file_ext == "pdf"
    st.session_state.is_pdf = is_pdf
    images = []
    
    if is_pdf:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        selected_page = st.selectbox("Select Page", [f"Page {i+1}" for i in range(len(images))])
        st.session_state.selected_page_index = int(selected_page.split(" ")[1]) - 1
    else:
        image = Image.open(uploaded_file).convert("RGB")
        images.append(image)
        st.session_state.selected_page_index = 0
    
    # Use get with default to avoid KeyError
    image = images[st.session_state.get('selected_page_index', 0)]
    font = load_font(16)
    
    # Calculate initial zoom to fit image in container
    current_image_hash = hash(uploaded_file.getvalue())
    if st.session_state.image_hash != current_image_hash:
        st.session_state.image_hash = current_image_hash
        # Calculate zoom to fit image in 800px width container
        st.session_state.zoom_level = min(100, int(800 / image.width * 100))
    
    # ---------------- Workflow Instructions ----------------
    st.info("""
    **Enhanced Workflow Instructions:**
    1. Set a template name (e.g., "payment_advice")
    2. Draw bounding boxes around complete fields (trigger word + value together)
    3. The system will automatically separate trigger words and values
    4. Review and edit the extracted fields if needed
    5. Export the JSON when complete
    
    **New Feature:** Draw one box around the entire field (label + value) and the system will automatically separate them!
    """)
    
    # ---------------- 4-Column Layout ----------------
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1.5])
    
    # Column 1: Annotation Controls
    with col1:
        st.subheader("🎛️ Controls")
        
        # Template Name
        st.session_state.template_name = st.text_input("Template Name", value=st.session_state.template_name)
        
        st.info("🎯 **Smart Mode**: Draw boxes around complete fields (label + value). The system will automatically separate them!")
        
        stroke_width = st.slider("Stroke Width", 1, 10, 2)
        stroke_color = st.color_picker("Box Color", "#FF0000")
        
        # Zoom slider with session state
        zoom_level = st.slider("Zoom Level (%)", 50, 200, st.session_state.zoom_level)
        st.session_state.zoom_level = zoom_level
        
        if st.button("🧹 Clear All", key="clear_all"):
            st.session_state.trigger_fields = []
            st.experimental_rerun()
    
    # Column 2: Draw & Annotate
    with col2:
        st.subheader("✏️ Draw Fields")
        zoom = zoom_level / 100.0
        zoomed_size = (int(image.width * zoom), int(image.height * zoom))
        zoomed_image = image.resize(zoomed_size)
        
        # Create a container with scrollable area for large images
        canvas_container = st.container()
        with canvas_container:
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=zoomed_image,
                update_streamlit=True,
                height=zoomed_size[1],
                width=zoomed_size[0],
                drawing_mode="rect",
                key="canvas",
                display_toolbar=True
            )
        
        # Add scroll instructions if image is large
        if zoomed_size[0] > 800 or zoomed_size[1] > 600:
            st.caption("📜 Scroll to see the full image")
        
        if canvas_result.json_data and canvas_result.json_data["objects"]:
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    x = int(obj["left"] / zoom)
                    y = int(obj["top"] / zoom)
                    w = int(obj["width"] / zoom)
                    h = int(obj["height"] / zoom)
                    
                    # Use the auto-separation function for regular fields
                    trigger_word, trigger_coords, value, value_coords, field_name = separate_trigger_and_value(
                        image, x, y, w, h, TRIGGER_WORDS
                    )
                    
                    # Add page information
                    trigger_coords["page"] = st.session_state.get('selected_page_index', 0) + 1
                    value_coords["page"] = st.session_state.get('selected_page_index', 0) + 1
                    
                    # Check if this field already exists
                    existing_index = find_existing_field(st.session_state.trigger_fields, trigger_word)
                    
                    if existing_index >= 0:
                        # Update existing field
                        st.session_state.trigger_fields[existing_index].update({
                            "field_name": field_name,
                            "trigger_word": trigger_word,
                            "trigger_coordinates": trigger_coords,
                            "value": value,
                            "value_coordinates": value_coords
                        })
                        st.success(f"Updated field: {trigger_word}")
                    else:
                        # Create new field
                        st.session_state.trigger_fields.append({
                            "field_name": field_name,
                            "trigger_word": trigger_word,
                            "trigger_coordinates": trigger_coords,
                            "value": value,
                            "value_coordinates": value_coords
                        })
                        st.success(f"Created new field: {trigger_word}")
    
    # Column 3: Annotated Image
    with col3:
        st.subheader("🖼️ Annotated Image")
        annotated_img = update_annotated_image(
            image, 
            st.session_state.trigger_fields, 
            stroke_color, 
            stroke_width, 
            font, 
            highlight_idx=st.session_state.highlight_box_idx
        )
        
        # Display image with proper sizing
        st.image(annotated_img, caption="Annotated Image with Auto-Separated Fields", width=min(800, annotated_img.width))
        
        buffer = BytesIO()
        annotated_img.save(buffer, format="PNG")
        st.download_button("📥 Download Annotated Image", buffer.getvalue(), "annotated_output.png", "image/png")
    
    # Column 4: Fields Extracted
    with col4:
        st.subheader("📝 Auto-Extracted Fields")
        
        # Show trigger fields
        if st.session_state.trigger_fields:
            for i, field in enumerate(st.session_state.trigger_fields):
                # Status indicators
                has_trigger = field.get("trigger_coordinates") is not None
                has_value = field.get("value_coordinates") is not None
                
                if has_trigger and has_value:
                    status_icon = "✅"
                elif has_trigger or has_value:
                    status_icon = "⚠️"
                else:
                    status_icon = "❌"
                
                expander_title = f"{status_icon} {field['trigger_word']}"
                
                with st.expander(expander_title):
                    # Edit trigger word
                    trigger_options = TRIGGER_WORDS + ["Custom"]
                    current_trigger = field["trigger_word"]
                    
                    # Find current trigger in options
                    if current_trigger in trigger_options:
                        trigger_index = trigger_options.index(current_trigger)
                    else:
                        trigger_index = len(trigger_options) - 1  # Custom
                    
                    selected_trigger = st.selectbox(
                        "Trigger Word", 
                        options=trigger_options,
                        index=trigger_index,
                        key=f"trigger_{i}"
                    )
                    
                    if selected_trigger == "Custom":
                        trigger_word = st.text_input("Custom Trigger", value=current_trigger, key=f"custom_trigger_{i}")
                    else:
                        trigger_word = selected_trigger
                    
                    # Show auto-generated field name
                    field_name = normalize_field_name(trigger_word)
                    st.text_input("Field Name", value=field_name, key=f"field_name_{i}", disabled=True)
                    
                    # Edit value
                    value = st.text_input("Extracted Value", value=field.get("value", ""), key=f"value_{i}")
                    
                    # Update field if changes were made
                    if trigger_word != field["trigger_word"] or value != field.get("value", ""):
                        field["trigger_word"] = trigger_word
                        field["field_name"] = field_name
                        field["value"] = value
                        st.session_state.update_flag = True
                    
                    # Show coordinates information
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if field.get("trigger_coordinates"):
                            st.caption("**Trigger Coords:**")
                            coords = field["trigger_coordinates"]
                            st.caption(f"({coords['x']:.1f}, {coords['y']:.1f}) {coords['width']:.1f}×{coords['height']:.1f}")
                    
                    with col_b:
                        if field.get("value_coordinates"):
                            st.caption("**Value Coords:**")
                            coords = field["value_coordinates"]
                            st.caption(f"({coords['x']:.1f}, {coords['y']:.1f}) {coords['width']:.1f}×{coords['height']:.1f}")
                    
                    # Action buttons
                    col_c, col_d = st.columns(2)
                    with col_c:
                        if st.button("✏️ Edit", key=f"edit_{i}"):
                            st.session_state.highlight_box_idx = i
                            st.success(f"Field {i+1} highlighted for editing.")
                    with col_d:
                        if st.button("🗑️ Delete", key=f"delete_{i}"):
                            st.session_state.trigger_fields.pop(i)
                            st.experimental_rerun()
        
        # Export JSON Section
        st.subheader("📦 Export JSON")
        
        # Prepare the data in the required format
        fields_for_export = []
        for field in st.session_state.trigger_fields:
            export_field = {
                "field_name": field["field_name"],
                "trigger_word": field["trigger_word"],
                "coordinates": field.get("trigger_coordinates"),
                "expected_value": field.get("value", ""),
                "value_coordinates": field.get("value_coordinates")
            }
            fields_for_export.append(export_field)
        
        full_data = {
            "template_name": st.session_state.template_name,
            "fields": fields_for_export
        }
        
        with st.expander("View JSON Data"):
            st.json(full_data)
        
        json_bytes = BytesIO(json.dumps(full_data, indent=2).encode("utf-8"))
        st.download_button("📄 Download JSON", data=json_bytes, file_name="extracted_data.json", mime="application/json")
else:
    st.info("Upload an image or PDF to begin auto field extraction.")
    
    
    
    
    
    
    
#FLOW DIAGRAM OF THIS CODE
# Start
#   ↓
# Initialize session state variables (trigger_fields, selected_page_index, etc.)
#   ↓
# Display file upload interface
#   ↓
# User uploads file? → No → Show info message → End
#   ↓ Yes
# Check file type:
#   - If PDF: Convert each page to image using PyMuPDF
#   - If image: Directly load the image
#   ↓
# Display page selector (for PDF) or single image view
#   ↓
# Calculate appropriate zoom level to fit container
#   ↓
# Display 4-column layout:
#   Column 1: Controls (template name, stroke settings, zoom)
#   Column 2: Drawing canvas with zoom controls
#   Column 3: Annotated image preview
#   Column 4: Extracted fields list
#   ↓
# User draws bounding box around field? → No → Wait for user input
#   ↓ Yes
# Extract text from bounding box using Tesseract OCR
#   ↓
# Try to match extracted text with predefined trigger words:
#   - Use regex patterns with word boundaries for accurate matching
#   - Prioritize longer matches to avoid confusion (e.g., "Handling fee of remitting bank" over "Remitting bank")
#   ↓
# Separate trigger word and value:
#   - Get word-level OCR data
#   - Match trigger phrase consecutively in OCR data
#   - Calculate precise bounding boxes for both trigger and value
#   - If spatial separation fails, split box in half
#   ↓
# Check if field already exists? → Yes → Update existing field
#   ↓ No
# Create new field entry with:
#   - Field name (normalized trigger)
#   - Trigger word
#   - Trigger coordinates
#   - Value
#   - Value coordinates
#   - Page number (for PDFs)
#   ↓
# Display extracted fields in sidebar with edit options
#   ↓
# User edits field? → Yes → Update field data
#   ↓ No
# Generate annotated image with:
#   - Green boxes for trigger regions
#   - Orange boxes for value regions
#   - Labels showing trigger word and value
#   ↓
# User exports JSON? → No → Wait for more actions
#   ↓ Yes
# Prepare JSON structure with:
#   - Template name
#   - Field names
#   - Trigger words
#   - Trigger coordinates
#   - Expected values
#   - Value coordinates
#   ↓
# Generate download link for JSON file
#   ↓
# End