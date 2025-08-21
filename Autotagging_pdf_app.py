# app.py

import streamlit as st
import json
import os
import tempfile
import zipfile
from io import BytesIO
from PIL import Image
from document_processor import (
    process_document, 
    draw_bounding_boxes_on_document,
    DEFAULT_HEAD_KEYS, 
    DEFAULT_FIELD_TYPES,
    DEFAULT_FIELD_COLORS
)

# Page configuration
st.set_page_config(
    page_title="Document Auto-Tagger",
    page_icon="📄",
    layout="wide"
)

# Title and description
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
if 'head_keys' not in st.session_state:
    st.session_state.head_keys = DEFAULT_HEAD_KEYS.copy()
if 'field_types' not in st.session_state:
    st.session_state.field_types = DEFAULT_FIELD_TYPES.copy()

# Sidebar for configuration
st.sidebar.header("Field Configuration")
st.sidebar.markdown("Configure the fields you want to extract from the document:")

# Text area for head keys JSON
head_keys_json = st.sidebar.text_area(
    "HEAD_KEYS Configuration (JSON format)",
    value=json.dumps(st.session_state.head_keys, indent=2),
    height=200
)

# Text area for field types JSON
field_types_json = st.sidebar.text_area(
    "FIELD_TYPES Configuration (JSON format)",
    value=json.dumps(st.session_state.field_types, indent=2),
    height=200
)

# Button to update configurations
if st.sidebar.button("Update Configurations"):
    try:
        new_head_keys = json.loads(head_keys_json)
        if isinstance(new_head_keys, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in new_head_keys.items()):
            st.session_state.head_keys = new_head_keys
            st.sidebar.success("HEAD_KEYS updated successfully!")
        else:
            st.sidebar.error("Invalid HEAD_KEYS format. Please provide a dictionary with string keys and values.")
    except json.JSONDecodeError:
        st.sidebar.error("Invalid HEAD_KEYS JSON format. Please check your syntax.")
    
    try:
        new_field_types = json.loads(field_types_json)
        if isinstance(new_field_types, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in new_field_types.items()):
            st.session_state.field_types = new_field_types
            st.sidebar.success("FIELD_TYPES updated successfully!")
        else:
            st.sidebar.error("Invalid FIELD_TYPES format. Please provide a dictionary with string keys and values.")
    except json.JSONDecodeError:
        st.sidebar.error("Invalid FIELD_TYPES JSON format. Please check your syntax.")

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
            result = process_document(
                temp_file_path, 
                file_type, 
                st.session_state.head_keys,
                st.session_state.field_types
            )
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
            
            # Create field colors dynamically
            field_colors = {}
            predefined_colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 255),
                (0, 255, 255), (128, 0, 128), (255, 192, 203), (0, 128, 0), (255, 215, 0),
                (230, 230, 250), (255, 69, 0), (135, 206, 235), (238, 130, 238), (34, 139, 34),
                (255, 218, 185), (75, 0, 130), (154, 205, 50)
            ]
            
            for i, label in enumerate(st.session_state.head_keys.values()):
                color = predefined_colors[i % len(predefined_colors)]
                field_colors[label] = color
            
            # Draw bounding boxes on the document
            st.session_state.annotated_images = draw_bounding_boxes_on_document(
                temp_file_path, 
                result["entity_details"], 
                file_type, 
                field_colors
            )
            
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
            "Field Type": st.session_state.field_types.get(label, "text"),
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
1. Configure the fields you want to extract in the HEAD_KEYS section
2. Configure the field types in the FIELD_TYPES section (text, date, amount, reference, alphanumeric, split_text)
3. Upload a PDF or Image document using the file uploader
4. The system will automatically process the document and extract fields
5. View the extracted fields in the table below
6. See the annotated document pages with bounding boxes for all header fields
7. Download the JSON file with the extracted data
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

# Information about the field types
st.sidebar.header("Field Types")
st.sidebar.markdown("""
Supported field types:
- `text`: Generic text field (default)
- `date`: Date field with special date pattern recognition
- `amount`: Currency amount with special amount pattern recognition
- `reference`: Reference number with special pattern recognition
- `alphanumeric`: Alphanumeric code with special pattern recognition
- `split_text`: Text field that needs to be split into parts (like name and email)
""")

# Information about the bounding boxes
st.sidebar.header("Bounding Boxes")
st.sidebar.markdown("""
The system automatically draws bounding boxes for:
- All header fields (trigger words) in their respective colors
- All corresponding values in the same color as the trigger word
- For split_text fields, separate boxes are drawn for each part
- Field names are displayed above the trigger word boxes
- Extracted values are displayed below the value boxes
Note: Only header fields and their values are annotated, not tables or important notes.
""")