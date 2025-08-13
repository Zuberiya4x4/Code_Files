# Enhanced OCR Tagger with Advanced Features
import streamlit as st
import pandas as pd
import pytesseract
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from io import BytesIO
import json
import os
import re
from streamlit_drawable_canvas import st_canvas
import numpy as np
from datetime import datetime
import zipfile
import base64

# Tesseract Path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="OCR Tagger Pro", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    .success-alert {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #c3e6cb;
    }
    .warning-alert {
        background: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #ffeaa7;
    }
    .info-alert {
        background: #d1ecf1;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Enhanced Helpers ----------------
def load_font(size=18):
    try:
        return ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", size)
    except:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Arial.ttf", size)  # macOS
        except:
            try:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)  # Linux
            except:
                return ImageFont.load_default()

def draw_text(draw, position, text, font, text_color="white", bg_color="black"):
    bbox = draw.textbbox(position, text, font=font)
    # Add padding to background
    padding = 2
    draw.rectangle([bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding], fill=bg_color)
    draw.text(position, text, fill=text_color, font=font)

def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """Enhance image quality for better OCR"""
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    return image

def preprocess_image_for_ocr(image_crop, preprocessing_type="default"):
    """Enhanced preprocessing for better OCR accuracy"""
    if preprocessing_type == "grayscale":
        return image_crop.convert("L")
    elif preprocessing_type == "threshold":
        gray = image_crop.convert("L")
        return gray.point(lambda x: 0 if x < 128 else 255, '1')
    elif preprocessing_type == "blur":
        return image_crop.filter(ImageFilter.GaussianBlur(radius=1))
    elif preprocessing_type == "sharpen":
        return image_crop.filter(ImageFilter.SHARPEN)
    else:
        return image_crop.convert("L")

def advanced_ocr_with_confidence(image_crop, preprocessing_type="default", psm_mode=6):
    """Perform OCR with confidence scores and multiple preprocessing options"""
    processed_image = preprocess_image_for_ocr(image_crop, preprocessing_type)
    
    # Get OCR data with confidence
    ocr_data = pytesseract.image_to_data(processed_image, config=f"--psm {psm_mode}", output_type=pytesseract.Output.DICT)
    
    # Extract text with confidence
    text_parts = []
    confidences = []
    
    for i in range(len(ocr_data['text'])):
        if int(ocr_data['conf'][i]) > 0:  # Only consider text with confidence > 0
            text_parts.append(ocr_data['text'][i])
            confidences.append(int(ocr_data['conf'][i]))
    
    raw_text = ' '.join(text_parts)
    avg_confidence = np.mean(confidences) if confidences else 0
    
    return raw_text, avg_confidence

def update_annotated_image(image, boxes, stroke_color, stroke_width, font, highlight_idx=-1, show_confidence=True):
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    for i, box in enumerate(boxes):
        coords = box["coordinates"]
        x, y, w, h = coords["x"], coords["y"], coords["width"], coords["height"]
        
        # Highlight selected box
        color = "#00FF00" if i == highlight_idx else stroke_color
        line_width = stroke_width + 2 if i == highlight_idx else stroke_width
        
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)
        
        # Enhanced text display
        y_offset = -60 if show_confidence else -45
        draw_text(draw, (x, y + y_offset), f"📋 {box['type']}", font, bg_color="#6a1b9a")
        draw_text(draw, (x, y + y_offset + 15), f"🏷️ {box['title']}", font, bg_color="#d32f2f")
        draw_text(draw, (x, y + y_offset + 30), f"💡 {box['value'][:30]}{'...' if len(box['value']) > 30 else ''}", font, bg_color="#1976d2")
        
        if show_confidence and 'confidence' in box:
            confidence_color = "#4caf50" if box['confidence'] > 70 else "#ff9800" if box['confidence'] > 40 else "#f44336"
            draw_text(draw, (x, y + y_offset + 45), f"🎯 {box['confidence']:.1f}%", font, bg_color=confidence_color)
    
    return annotated_img

def clean_and_extract_text(raw_text):
    """Enhanced text cleaning with better pattern recognition"""
    # Enhanced cleaning patterns
    cleaned = raw_text.replace("TINR", "INR").replace("INRSO", "INR 50").replace("INR5O", "INR 50")
    cleaned = re.sub(r'\bS0\b', "50", cleaned)
    cleaned = re.sub(r'\bO\b', "0", cleaned)  # Common OCR mistake
    cleaned = re.sub(r'\s+', ' ', cleaned.strip())

    # Enhanced pattern matching
    patterns = [
        (r'INR\s*[\d\s,\.]+', "Amount"),
        (r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', "Date"),
        (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', "Email"),
        (r'\+?[\d\s\-\(\)]{10,}', "Phone"),
        (r'[A-Z]{2}\d{2}\s?\d{4}\s?\d{10}', "Account Number"),
        (r'GST\s*:?\s*[A-Z0-9]{15}', "GST Number"),
        (r'PAN\s*:?\s*[A-Z]{5}\d{4}[A-Z]', "PAN Number"),
    ]
    
    for pattern, field_type in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            value = match.group().strip()
            if field_type == "Amount":
                numeric_part = re.sub(r'[^\d.]', '', value)
                try:
                    formatted_amount = f"INR {float(numeric_part):,.2f}"
                    return field_type, formatted_amount
                except ValueError:
                    return field_type, value
            return field_type, value
    
    # Default extraction
    match = re.match(r"(.*?):\s*(.*)", cleaned)
    if match:
        title, value = match.groups()
        return title.strip(), value.strip()
    else:
        return "Field", cleaned

def validate_field_data(field_type, value):
    """Validate extracted field data"""
    if field_type == "email":
        return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value) is not None
    elif field_type == "date":
        try:
            # Try common date formats
            for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y']:
                try:
                    datetime.strptime(value, fmt)
                    return True
                except ValueError:
                    continue
            return False
        except:
            return False
    elif field_type == "currency":
        return re.match(r'.*\d+.*', value) is not None
    elif field_type == "number":
        return re.match(r'^\d+(\.\d+)?$', value) is not None
    return True

def export_to_excel(data):
    """Export data to Excel format"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Fields sheet
        if data['fields']:
            fields_df = pd.DataFrame(data['fields'])
            fields_df.to_excel(writer, sheet_name='Fields', index=False)
        
        # Table sheet
        if data['table'] and 'invoice_details' in data['table']:
            table_data = data['table']['invoice_details']
            if isinstance(table_data, list) and table_data:
                table_df = pd.DataFrame(table_data)
                table_df.to_excel(writer, sheet_name='Table', index=False)
    
    output.seek(0)
    return output.getvalue()

def create_backup_zip(data, annotated_image):
    """Create a backup ZIP file with all data"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add JSON data
        zip_file.writestr('extracted_data.json', json.dumps(data, indent=2))
        
        # Add annotated image
        img_buffer = BytesIO()
        annotated_image.save(img_buffer, format='PNG')
        zip_file.writestr('annotated_image.png', img_buffer.getvalue())
        
        # Add Excel file
        excel_data = export_to_excel(data)
        zip_file.writestr('extracted_data.xlsx', excel_data)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# ---------------- Enhanced State Management ----------------
state_keys = [
    "boxes", "json_table", "selected_page_index", "highlight_box_idx", 
    "table_view_mode", "update_flag", "processing_history", "image_enhancements",
    "ocr_settings", "field_templates", "auto_save_enabled", "export_format"
]

for key in state_keys:
    if key not in st.session_state:
        if key == "boxes":
            st.session_state[key] = []
        elif key == "json_table":
            st.session_state[key] = {}
        elif key in ["selected_page_index", "highlight_box_idx"]:
            st.session_state[key] = 0
        elif key == "table_view_mode":
            st.session_state[key] = "row-wise"
        elif key == "processing_history":
            st.session_state[key] = []
        elif key == "image_enhancements":
            st.session_state[key] = {"brightness": 1.0, "contrast": 1.0, "sharpness": 1.0}
        elif key == "ocr_settings":
            st.session_state[key] = {"preprocessing": "default", "psm_mode": 6}
        elif key == "field_templates":
            st.session_state[key] = {}
        elif key == "export_format":
            st.session_state[key] = "json"
        else:
            st.session_state[key] = False

# ---------------- Enhanced Main Title ----------------
st.markdown("""
<div class="main-header">
    <h1>🖼️ Field Tagging System</h1>
    <p>Enhanced OCR with AI-powered text extraction, validation, and export capabilities</p>
</div>
""", unsafe_allow_html=True)

# ---------------- File Upload with Enhanced Features ----------------
col_upload, col_settings = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "📁 Upload PDF or Image", 
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        help="Supports PDF, PNG, JPG, JPEG, TIFF, and BMP formats"
    )

with col_settings:
    st.markdown("### ⚙️ Global Settings")
    st.session_state.auto_save_enabled = st.checkbox("🔄 Auto-save progress", value=st.session_state.auto_save_enabled)
    st.session_state.export_format = st.selectbox("📤 Default export format", ["json", "excel", "csv"], index=["json", "excel", "csv"].index(st.session_state.export_format))

if uploaded_file:
    # Enhanced file processing
    file_ext = uploaded_file.name.split(".")[-1].lower()
    is_pdf = file_ext == "pdf"
    images = []
    
    # File info display
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
    st.markdown(f"""
    <div class="info-alert">
        📄 <strong>File:</strong> {uploaded_file.name} | 
        📊 <strong>Size:</strong> {file_size:.2f} MB | 
        🎯 <strong>Type:</strong> {file_ext.upper()}
    </div>
    """, unsafe_allow_html=True)

    if is_pdf:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        total_pages = len(doc)
        
        for page in doc:
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        # Enhanced page selection
        col_page, col_info = st.columns([1, 1])
        with col_page:
            selected_page = st.selectbox(
                "📄 Select Page", 
                [f"Page {i+1}" for i in range(len(images))],
                help=f"Document has {total_pages} pages"
            )
            st.session_state.selected_page_index = int(selected_page.split(" ")[1]) - 1
        
        with col_info:
            st.markdown(f"""
            <div class="metric-card">
                <strong>📊 Document Info</strong><br>
                📄 Total Pages: {total_pages}<br>
                🎯 Current Page: {st.session_state.selected_page_index + 1}<br>
                📏 Page Size: {images[st.session_state.selected_page_index].size}
            </div>
            """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded_file).convert("RGB")
        images.append(image)
        st.session_state.selected_page_index = 0
        
        # Image info display
        st.markdown(f"""
        <div class="metric-card">
            <strong>🖼️ Image Info</strong><br>
            📏 Dimensions: {image.size[0]} x {image.size[1]}<br>
            🎨 Mode: {image.mode}<br>
            📊 Format: {image.format}
        </div>
        """, unsafe_allow_html=True)

    image = images[st.session_state.selected_page_index]
    font = load_font(16)

    # ---------------- Enhanced 4-Column Layout ----------------
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1.5])

    # Column 1: Enhanced Annotation Controls
    with col1:
        st.markdown("### 🎛️ Advanced Controls")
        
        # Drawing controls
        with st.expander("🎨 Drawing Settings", expanded=True):
            stroke_width = st.slider("Stroke Width", 1, 10, 2)
            stroke_color = st.color_picker("Box Color", "#FF0000")
            field_type = st.selectbox("Field Type", ["text", "date", "email", "currency", "number", "label", "phone", "gst", "pan"])
            zoom_level = st.slider("Zoom Level (%)", 50, 200, 100)
        
        # Image enhancement controls
        with st.expander("🖼️ Image Enhancement"):
            brightness = st.slider("Brightness", 0.5, 2.0, st.session_state.image_enhancements["brightness"], 0.1)
            contrast = st.slider("Contrast", 0.5, 2.0, st.session_state.image_enhancements["contrast"], 0.1)
            sharpness = st.slider("Sharpness", 0.5, 2.0, st.session_state.image_enhancements["sharpness"], 0.1)
            
            st.session_state.image_enhancements = {
                "brightness": brightness,
                "contrast": contrast,
                "sharpness": sharpness
            }
        
        # OCR settings
        with st.expander("🔍 OCR Settings"):
            preprocessing = st.selectbox("Preprocessing", ["default", "grayscale", "threshold", "blur", "sharpen"])
            psm_mode = st.selectbox("PSM Mode", [3, 6, 7, 8, 13], index=1, help="Page Segmentation Mode")
            
            st.session_state.ocr_settings = {
                "preprocessing": preprocessing,
                "psm_mode": psm_mode
            }
        
        # Action buttons
        st.markdown("### 🚀 Quick Actions")
        col_clear, col_export = st.columns(2)
        
        with col_clear:
            if st.button("🧹 Clear All", key="clear_all", use_container_width=True):
                st.session_state.boxes = []
                st.rerun()
        
        with col_export:
            if st.button("📤 Quick Export", key="quick_export", use_container_width=True):
                if st.session_state.boxes:
                    st.success("✅ Data ready for export!")
                else:
                    st.warning("⚠️ No data to export")
        
        # Processing statistics
        if st.session_state.boxes:
            st.markdown("### 📊 Statistics")
            total_fields = len(st.session_state.boxes)
            avg_confidence = np.mean([box.get('confidence', 0) for box in st.session_state.boxes])
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>📈 Processing Stats</strong><br>
                📋 Total Fields: {total_fields}<br>
                🎯 Avg Confidence: {avg_confidence:.1f}%<br>
                ✅ Valid Fields: {sum(1 for box in st.session_state.boxes if validate_field_data(box['type'], box['value']))}
            </div>
            """, unsafe_allow_html=True)

    # Column 2: Enhanced Draw & Annotate
    with col2:
        st.markdown("### ✏️ Smart Draw & Annotate")
        
        # Apply image enhancements
        enhanced_image = enhance_image(
            image, 
            brightness=st.session_state.image_enhancements["brightness"],
            contrast=st.session_state.image_enhancements["contrast"],
            sharpness=st.session_state.image_enhancements["sharpness"]
        )
        
        zoom = zoom_level / 100.0
        zoomed_size = (int(enhanced_image.width * zoom), int(enhanced_image.height * zoom))
        zoomed_image = enhanced_image.resize(zoomed_size, Image.Resampling.LANCZOS)
        
        # Enhanced canvas with better tools
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

        # Enhanced processing with confidence scores
        if canvas_result.json_data and canvas_result.json_data["objects"]:
            new_boxes = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, obj in enumerate(canvas_result.json_data["objects"]):
                if obj["type"] == "rect":
                    progress_bar.progress((idx + 1) / len(canvas_result.json_data["objects"]))
                    status_text.text(f"Processing field {idx + 1}/{len(canvas_result.json_data['objects'])}")
                    
                    x = int(obj["left"] / zoom)
                    y = int(obj["top"] / zoom)
                    w = int(obj["width"] / zoom)
                    h = int(obj["height"] / zoom)
                    
                    cropped = enhanced_image.crop((x, y, x + w, y + h))
                    
                    # Advanced OCR with confidence
                    raw_text, confidence = advanced_ocr_with_confidence(
                        cropped, 
                        st.session_state.ocr_settings["preprocessing"],
                        st.session_state.ocr_settings["psm_mode"]
                    )
                    
                    title, value = clean_and_extract_text(raw_text)
                    
                    # Field validation
                    is_valid = validate_field_data(field_type, value)
                    
                    new_boxes.append({
                        "title": title,
                        "value": value,
                        "type": field_type,
                        "confidence": confidence,
                        "is_valid": is_valid,
                        "raw_text": raw_text,
                        "coordinates": {
                            "x": x, "y": y, "width": w, "height": h,
                            "page": st.session_state.selected_page_index + 1
                        },
                        "timestamp": datetime.now().isoformat()
                    })
            
            st.session_state.boxes = new_boxes
            progress_bar.empty()
            status_text.empty()
            
            if new_boxes:
                st.markdown(f"""
                <div class="success-alert">
                    ✅ Successfully processed {len(new_boxes)} field(s) with average confidence of {np.mean([box['confidence'] for box in new_boxes]):.1f}%
                </div>
                """, unsafe_allow_html=True)

    # Column 3: Enhanced Annotated Image
    with col3:
        st.markdown("### 🖼️ Smart Annotated View")
        
        # Display controls
        col_show, col_highlight = st.columns(2)
        with col_show:
            show_confidence = st.checkbox("🎯 Show Confidence", value=True)
        with col_highlight:
            if st.session_state.boxes:
                highlight_idx = st.selectbox("🔍 Highlight Field", 
                                           ["None"] + [f"Field {i+1}" for i in range(len(st.session_state.boxes))])
                st.session_state.highlight_box_idx = -1 if highlight_idx == "None" else int(highlight_idx.split(" ")[1]) - 1
        
        # Generate annotated image
        annotated_img = update_annotated_image(
            enhanced_image, 
            st.session_state.boxes, 
            stroke_color, 
            stroke_width, 
            font, 
            highlight_idx=st.session_state.highlight_box_idx,
            show_confidence=show_confidence
        )
        
        st.image(annotated_img, caption="Smart Annotated Image", use_column_width=True)
        
        # Enhanced download options
        st.markdown("### 📥 Download Options")
        col_png, col_zip = st.columns(2)
        
        with col_png:
            buffer = BytesIO()
            annotated_img.save(buffer, format="PNG")
            st.download_button(
                "🖼️ Download PNG", 
                buffer.getvalue(), 
                f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                "image/png",
                use_container_width=True
            )
        
        with col_zip:
            if st.session_state.boxes:
                full_data = {
                    "fields": st.session_state.boxes,
                    "table": st.session_state.json_table,
                    "metadata": {
                        "processed_at": datetime.now().isoformat(),
                        "file_name": uploaded_file.name,
                        "total_fields": len(st.session_state.boxes),
                        "average_confidence": np.mean([box.get('confidence', 0) for box in st.session_state.boxes])
                    }
                }
                
                zip_data = create_backup_zip(full_data, annotated_img)
                st.download_button(
                    "📦 Download All", 
                    zip_data,
                    f"ocr_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    "application/zip",
                    use_container_width=True
                )

    # Column 4: Enhanced Fields Management
    with col4:
        st.markdown("### 📝 Smart Fields Manager")
        
        # Field filtering and sorting
        if st.session_state.boxes:
            col_filter, col_sort = st.columns(2)
            with col_filter:
                filter_type = st.selectbox("🔍 Filter by Type", ["All"] + list(set(box['type'] for box in st.session_state.boxes)))
            with col_sort:
                sort_by = st.selectbox("🔄 Sort by", ["Index", "Confidence", "Type", "Title"])
            
            # Apply filters and sorting
            filtered_boxes = st.session_state.boxes
            if filter_type != "All":
                filtered_boxes = [box for box in filtered_boxes if box['type'] == filter_type]
            
            if sort_by == "Confidence":
                filtered_boxes.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            elif sort_by == "Type":
                filtered_boxes.sort(key=lambda x: x['type'])
            elif sort_by == "Title":
                filtered_boxes.sort(key=lambda x: x['title'])
            
            # Display filtered fields
            for i, box in enumerate(filtered_boxes):
                original_idx = st.session_state.boxes.index(box)
                
                # Enhanced field display with validation status
                validation_icon = "✅" if box.get('is_valid', True) else "❌"
                confidence_color = "🟢" if box.get('confidence', 0) > 70 else "🟡" if box.get('confidence', 0) > 40 else "🔴"
                
                with st.expander(f"{validation_icon} {confidence_color} Field {original_idx+1} ({box['type']}) - {box.get('confidence', 0):.1f}%"):
                    # Enhanced field editing
                    new_type = st.selectbox(
                        "Type", 
                        ["text", "date", "email", "currency", "number", "label", "phone", "gst", "pan"], 
                        index=["text", "date", "email", "currency", "number", "label", "phone", "gst", "pan"].index(box["type"]), 
                        key=f"type_{original_idx}"
                    )
                    
                    new_title = st.text_input("Title", value=box["title"], key=f"title_{original_idx}")
                    new_value = st.text_area("Value", value=box["value"], key=f"value_{original_idx}", height=60)
                    
                    # Show confidence and validation
                    col_conf, col_valid = st.columns(2)
                    with col_conf:
                        st.metric("Confidence", f"{box.get('confidence', 0):.1f}%")
                    with col_valid:
                        is_valid = validate_field_data(new_type, new_value)
                        st.metric("Valid", "✅" if is_valid else "❌")
                    
                    # Show raw OCR text if different from cleaned value
                    if box.get('raw_text') and box['raw_text'] != box['value']:
                        st.text_area("Raw OCR Text", value=box['raw_text'], key=f"raw_{original_idx}", height=40, disabled=True)
                    
                    # Update field data
                    if new_type != box["type"] or new_title != box["title"] or new_value != box["value"]:
                        st.session_state.boxes[original_idx]["type"] = new_type
                        st.session_state.boxes[original_idx]["title"] = new_title
                        st.session_state.boxes[original_idx]["value"] = new_value
                        st.session_state.boxes[original_idx]["is_valid"] = is_valid
                        st.session_state.update_flag = True
                    
                    # Enhanced coordinates display
                    st.markdown("**📍 Coordinates:**")
                    coord_data = box["coordinates"]
                    st.json(coord_data)
                    
                    # Action buttons
                    col_update, col_delete, col_highlight = st.columns(3)
                    with col_update:
                        if st.button("💾 Update", key=f"update_{original_idx}", use_container_width=True):
                            st.session_state.boxes[original_idx]["timestamp"] = datetime.now().isoformat()
                            st.success("✅ Updated!")
                    
                    with col_delete:
                        if st.button("🗑️ Delete", key=f"delete_{original_idx}", use_container_width=True):
                            st.session_state.boxes.pop(original_idx)
                            st.rerun()
                    
                    with col_highlight:
                        if st.button("🔍 Focus", key=f"highlight_{original_idx}", use_container_width=True):
                            st.session_state.highlight_box_idx = original_idx
                            st.rerun()

        # Enhanced Extract Table Section
        st.markdown("### 📊 Smart Table Extraction")
        
        # Table extraction settings
        with st.expander("⚙️ Table Settings"):
            st.session_state.table_view_mode = st.selectbox(
                "Table JSON Format", 
                ["row-wise", "column-wise", "table"],
                help="Choose how to structure the extracted table data"
            )
            
            table_detection_mode = st.selectbox(
                "Detection Mode",
                ["auto", "lattice", "stream"],
                help="Table detection algorithm"
            )

        col_extract, col_preview = st.columns(2)
        with col_extract:
            if st.button("🧾 Extract Table", key="extract_table", use_container_width=True):
                if is_pdf:
                    uploaded_file.seek(0)
                    with pdfplumber.open(uploaded_file) as pdf:
                        page_index = st.session_state.selected_page_index
                        page = pdf.pages[page_index]
                        
                        # Enhanced table extraction
                        tables = page.extract_tables()
                        
                        if tables:
                            table = tables[0]  # Get first table
                            headers = [str(h).strip() if h else f"Column_{i+1}" for i, h in enumerate(table[0])]
                            rows = table[1:]
                            
                            # Clean and process table data
                            cleaned_rows = []
                            for row in rows:
                                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                                cleaned_rows.append(cleaned_row)
                            
                            # Structure data based on selected format
                            if st.session_state.table_view_mode == "row-wise":
                                table_data = []
                                for row in cleaned_rows:
                                    row_dict = {}
                                    for i, header in enumerate(headers):
                                        row_dict[header] = row[i] if i < len(row) else ""
                                    table_data.append(row_dict)
                                
                                st.session_state.json_table = {
                                    "page": page_index + 1,
                                    "extraction_mode": st.session_state.table_view_mode,
                                    "headers": headers,
                                    "total_rows": len(cleaned_rows),
                                    "extracted_at": datetime.now().isoformat(),
                                    "invoice_details": table_data
                                }
                                
                            elif st.session_state.table_view_mode == "column-wise":
                                col_dict = {header: [] for header in headers}
                                for row in cleaned_rows:
                                    for i, header in enumerate(headers):
                                        col_dict[header].append(row[i] if i < len(row) else "")
                                
                                st.session_state.json_table = {
                                    "page": page_index + 1,
                                    "extraction_mode": st.session_state.table_view_mode,
                                    "headers": headers,
                                    "total_rows": len(cleaned_rows),
                                    "extracted_at": datetime.now().isoformat(),
                                    "invoice_details": col_dict
                                }
                                
                            elif st.session_state.table_view_mode == "table":
                                st.session_state.json_table = {
                                    "page": page_index + 1,
                                    "extraction_mode": st.session_state.table_view_mode,
                                    "headers": headers,
                                    "total_rows": len(cleaned_rows),
                                    "extracted_at": datetime.now().isoformat(),
                                    "invoice_details": [headers] + cleaned_rows
                                }
                            
                            st.markdown(f"""
                            <div class="success-alert">
                                ✅ Table extracted successfully! Found {len(cleaned_rows)} rows with {len(headers)} columns.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="warning-alert">
                                ⚠️ No table found on this page. Try adjusting the detection mode or check if the page contains tabular data.
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-alert">
                        ⚠️ Table extraction is only supported for PDF files.
                    </div>
                    """, unsafe_allow_html=True)

        with col_preview:
            if st.button("👁️ Preview Table", key="preview_table", use_container_width=True):
                if st.session_state.json_table:
                    st.success("✅ Table preview available below!")
                else:
                    st.info("ℹ️ No table data to preview")

        # Display extracted table with enhanced formatting
        if st.session_state.json_table:
            st.markdown("### 📋 Extracted Table Preview")
            
            # Table metadata
            table_info = st.session_state.json_table
            st.markdown(f"""
            <div class="metric-card">
                <strong>📊 Table Information</strong><br>
                📄 Page: {table_info.get('page', 'N/A')}<br>
                🔧 Mode: {table_info.get('extraction_mode', 'N/A')}<br>
                📊 Rows: {table_info.get('total_rows', 'N/A')}<br>
                📋 Columns: {len(table_info.get('headers', []))}<br>
                🕐 Extracted: {table_info.get('extracted_at', 'N/A')[:19] if table_info.get('extracted_at') else 'N/A'}
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced table display
            with st.expander("📊 Table Data", expanded=True):
                if st.session_state.table_view_mode in ["row-wise", "column-wise"]:
                    table_data = st.session_state.json_table.get("invoice_details", {})
                    if isinstance(table_data, list) and table_data:
                        # Convert to DataFrame for better display
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True, height=300)
                    elif isinstance(table_data, dict):
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True, height=300)
                    else:
                        st.json(table_data)
                else:
                    st.json(st.session_state.json_table)

        # Enhanced Export JSON Section
        st.markdown("### 📦 Smart Export Options")
        
        # Export format selection
        export_format = st.selectbox(
            "📤 Export Format",
            ["json", "excel", "csv", "all"],
            index=["json", "excel", "csv", "all"].index(st.session_state.export_format)
        )
        
        # Prepare comprehensive export data
        full_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "file_name": uploaded_file.name,
                "total_fields": len(st.session_state.boxes),
                "total_pages": len(images) if is_pdf else 1,
                "current_page": st.session_state.selected_page_index + 1,
                "average_confidence": np.mean([box.get('confidence', 0) for box in st.session_state.boxes]) if st.session_state.boxes else 0,
                "valid_fields": sum(1 for box in st.session_state.boxes if box.get('is_valid', True)),
                "ocr_settings": st.session_state.ocr_settings,
                "image_enhancements": st.session_state.image_enhancements
            },
            "fields": st.session_state.boxes,
            "table": st.session_state.json_table,
            "processing_summary": {
                "field_types": {field_type: len([box for box in st.session_state.boxes if box['type'] == field_type]) 
                              for field_type in set(box['type'] for box in st.session_state.boxes)} if st.session_state.boxes else {},
                "confidence_distribution": {
                    "high_confidence": len([box for box in st.session_state.boxes if box.get('confidence', 0) > 70]),
                    "medium_confidence": len([box for box in st.session_state.boxes if 40 <= box.get('confidence', 0) <= 70]),
                    "low_confidence": len([box for box in st.session_state.boxes if box.get('confidence', 0) < 40])
                }
            }
        }

        # Export preview
        with st.expander("👁️ Export Preview", expanded=False):
            st.json(full_data)
        
        # Export buttons
        col_json, col_excel, col_all = st.columns(3)
        
        with col_json:
            json_bytes = BytesIO(json.dumps(full_data, indent=2).encode("utf-8"))
            st.download_button(
                "📄 JSON",
                data=json_bytes.getvalue(),
                file_name=f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_excel:
            if st.session_state.boxes or st.session_state.json_table:
                excel_data = export_to_excel(full_data)
                st.download_button(
                    "📊 Excel",
                    data=excel_data,
                    file_name=f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.button("📊 Excel", disabled=True, use_container_width=True, help="No data to export")
        
        with col_all:
            if st.session_state.boxes:
                zip_data = create_backup_zip(full_data, annotated_img)
                st.download_button(
                    "📦 All Formats",
                    data=zip_data,
                    file_name=f"ocr_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            else:
                st.button("📦 All Formats", disabled=True, use_container_width=True, help="No data to export")

        # Processing history and templates
        if st.session_state.boxes:
            st.markdown("### 📈 Processing Insights")
            
            # Field type distribution
            field_counts = {}
            for box in st.session_state.boxes:
                field_type = box['type']
                field_counts[field_type] = field_counts.get(field_type, 0) + 1
            
            # Confidence distribution
            high_conf = len([box for box in st.session_state.boxes if box.get('confidence', 0) > 70])
            med_conf = len([box for box in st.session_state.boxes if 40 <= box.get('confidence', 0) <= 70])
            low_conf = len([box for box in st.session_state.boxes if box.get('confidence', 0) < 40])
            
            # Display insights
            st.markdown(f"""
            <div class="metric-card">
                <strong>🎯 Processing Insights</strong><br>
                📊 Field Distribution: {', '.join([f'{k}: {v}' for k, v in field_counts.items()])}<br>
                🟢 High Confidence: {high_conf}<br>
                🟡 Medium Confidence: {med_conf}<br>
                🔴 Low Confidence: {low_conf}<br>
                ✅ Validation Rate: {(sum(1 for box in st.session_state.boxes if box.get('is_valid', True)) / len(st.session_state.boxes) * 100):.1f}%
            </div>
            """, unsafe_allow_html=True)

        # Save template feature
        if st.session_state.boxes:
            st.markdown("### 💾 Save as Template")
            template_name = st.text_input("Template Name", placeholder="e.g., Invoice Template")
            if st.button("💾 Save Template", use_container_width=True):
                if template_name:
                    template_data = {
                        "name": template_name,
                        "fields": [{"type": box["type"], "title": box["title"]} for box in st.session_state.boxes],
                        "created_at": datetime.now().isoformat()
                    }
                    st.session_state.field_templates[template_name] = template_data
                    st.success(f"✅ Template '{template_name}' saved successfully!")
                else:
                    st.warning("⚠️ Please enter a template name")

else:
    # Enhanced welcome screen
    st.markdown("""
    <div class="feature-box">
        <h3>🚀 Welcome to OCR Tagger Pro</h3>
        <p>Upload an image or PDF to begin advanced document processing with:</p>
        <ul>
            <li>🎯 <strong>Smart OCR</strong> - Advanced text extraction with confidence scores</li>
            <li>🖼️ <strong>Image Enhancement</strong> - Brightness, contrast, and sharpness controls</li>
            <li>📊 <strong>Table Extraction</strong> - Intelligent table detection and parsing</li>
            <li>✅ <strong>Data Validation</strong> - Automatic field validation and error detection</li>
            <li>📤 <strong>Multi-format Export</strong> - JSON, Excel, CSV, and complete backup options</li>
            <li>🔍 <strong>Advanced Analysis</strong> - Processing insights and statistics</li>
        </ul>
        <p><strong>Supported formats:</strong> PDF, PNG, JPG, JPEG, TIFF, BMP</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Smart OCR</h4>
            <p>Advanced text extraction with confidence scoring and multiple preprocessing options for optimal accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_feat2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Table Extraction</h4>
            <p>Intelligent table detection and parsing with multiple output formats for structured data extraction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_feat3:
        st.markdown("""
        <div class="metric-card">
            <h4>📤 Smart Export</h4>
            <p>Multiple export formats including JSON, Excel, CSV, and complete backup packages with metadata.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🖼️ <strong>OCR Tagger Pro</strong> - Advanced Document Processing Solution</p>
    <p>Enhanced with AI-powered text extraction, validation, and export capabilities</p>
</div>
""", unsafe_allow_html=True)