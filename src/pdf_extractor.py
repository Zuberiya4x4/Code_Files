import streamlit as st
import pandas as pd
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import numpy as np
from streamlit_drawable_canvas import st_canvas
import pdfplumber
import tempfile
import os
import re

# -------------------------------
# Page Configuration & Styles
# -------------------------------
st.set_page_config(page_title="📄 Complete PDF Data Extractor", layout="wide")
st.markdown("""
<style>
.main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
.section-header { font-size: 1.5rem; font-weight: bold; color: #333; margin-top: 2rem; margin-bottom: 1rem; }
.success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 5px; margin: 10px 0; }
.info-box { background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 10px; border-radius: 5px; margin: 10px 0; }
.annotation-box { background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; border-radius: 8px; margin: 10px 0; }
.format-preview { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header Extraction Functions
# -------------------------------
def extract_text_and_tables(pdf_file):
    """Extract full text and tables from PDF using pdfplumber."""
    full_text = ""
    tables_data = []
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += "\n" + text
                page_tables = page.extract_tables()
                if page_tables:
                    tables_data.extend(page_tables)
    finally:
        os.remove(tmp_path)
    
    return full_text, tables_data

def extract_data_using_regex(pdf_text):
    """Extract specific header fields using regex patterns."""
    patterns = {
        "advice_date": r"Advice sending date[:\s]*(\d{2} \w{3} \d{4})",
        "advice_ref": r"Advice reference no[:\s]*([A-Za-z0-9\-]+)",
        "receipt_name": r"Recipient's name and contact information:[\s]*([^\n]+)",
        "receipt_email": r"Recipient's name and contact information:[\s]*[^\n]+[\s]*([\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,})",
        "sub_payment_type": r"Sub payment type[:\s]*([A-Za-z\s]+)(?=\n|$)",
        "transaction_type": r"Transaction type[:\s]*([A-Za-z\s]+)(?= Sub payment type|$)",
        
        # Updated beneficiary fields regex patterns
        "beneficiary_name": r"Beneficiary's\s+name\s*[:\-]?\s*\n?([A-Za-z0-9() ,.\-&\n]+?(?:\s+(?:Pvt|Private))?\s+(?:Ltd|Limited|Limit))",
        "beneficiary_bank": r"Beneficiary's bank[:\-]?\s*\n?([A-Za-z0-9 ,.\-&]+BANK)",
        "beneficiary_account_no": r"Beneficiary's\s+account\s*[:\-]?\s*\n?([A-Za-z0-9*]+)",
        "customer_reference": r"Customer reference[:\s]*(\S+)",
        "debit_amount": r"Debit amount[:\s]*([A-Za-z]+[\d,]+.\d{2})",
        "remittance_amount": r"Remittance amount[:\s]*([A-Za-z]+[\d,]+.\d{2})",
        "handling_fee": r"(Collect from Remitter|Collect from Beneficiary)",
        "value_date": r"Value date[:\s]*(\d{2} \w{3} \d{4})",
        "remitter_name": r"Remitter's name[:\s]*(.*)",
        "remitting_bank": r"Remitting bank[:\s]*(.*)",
        "instruction_reference": r"Instruction reference[:\s]*(\S+)",
        "other_reference": r"Other reference[:\s]*(\S+)",
        "remitter_to_beneficiary_info": r"Remitter to beneficiary information[:\s]*(.*?)(?=\n|$)"
    }

    extracted_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, pdf_text)
        if match:
            try:
                extracted_data[key] = match.group(1).strip()
            except IndexError:
                extracted_data[key] = None
        else:
            extracted_data[key] = None

    # Handle missing fields and avoid KeyErrors
    extracted_data['receipt_name'] = extracted_data.get('receipt_name', "")
    extracted_data['receipt_email'] = extracted_data.get('receipt_email', "")
    extracted_data['handling_fee'] = extracted_data.get('handling_fee', "Not Provided")

    # For remitter_to_beneficiary_info, set to "Not Provided" if it's empty or contains unwanted header-like text
    if extracted_data['remitter_to_beneficiary_info'] in ["DESCRIPTION DATE DOCUMENT NUMBER AMOUNT", 
                                                          "Pay Date Beneficiary Name Gross Amt TDS Amt Net Amt Invoice Num"]:
        extracted_data['remitter_to_beneficiary_info'] = ""
    
    return extracted_data

def process_invoice_table(table):
    """Process the table and structure the invoice details."""
    invoice_details = []
    
    for page_table in table:
        if page_table:
            header = page_table[0]
            for row in page_table[1:]:
                row_data = {}
                for i, column in enumerate(row):
                    if i < len(header):
                        row_data[header[i]] = column.strip().replace("\n", " ") if column else "Not Provided"
                invoice_details.append(row_data)
    
    return invoice_details

# -------------------------------
# Interactive Table Extraction Functions
# -------------------------------
def convert_pdf_to_images(pdf_file):
    """Convert PDF pages to images for visualization."""
    images = []
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(image)
    return images

def extract_table_with_pdfplumber(pdf_file, page_num, coordinates):
    """Extract table from specific coordinates on a PDF page."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    x1, y1, x2, y2 = coordinates
    try:
        with pdfplumber.open(tmp_path) as pdf:
            page = pdf.pages[page_num]
            crop_box = (x1 / 2, y1 / 2, x2 / 2, y2 / 2)
            cropped_page = page.crop(crop_box)
            table = cropped_page.extract_table()
    finally:
        os.remove(tmp_path)

    if table:
        return pd.DataFrame(table[1:], columns=table[0])
    return None

def dataframe_to_json(df, format_type="row_wise"):
    """Convert DataFrame to JSON in different formats."""
    if df is None or df.empty:
        return {}
    df = df.dropna(how='all').replace('', np.nan)
    if format_type == "row_wise":
        return {"rows": df.to_dict(orient='records')}
    elif format_type == "column_wise":
        return df.to_dict(orient='list')
    elif format_type == "table":
        return df.to_dict(orient='records')
    return {}

def draw_annotation(image, boxes_with_labels):
    """Draw annotation boxes on the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    for i, box in enumerate(boxes_with_labels):
        x1, y1, x2, y2 = box['coords']
        label = box['label']
        color = colors[i % len(colors)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], 
                      fill='white', outline=color, width=1)
        
        # Draw label text
        draw.text((x1, y1 - 25), label, fill=color, font=font)
    
    return annotated

def create_annotation_legend(boxes_with_labels):
    """Create a legend for annotation boxes."""
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    legend_html = '<div class="annotation-box"><h4>📍 Annotation Legend</h4><ul>'
    
    for i, box in enumerate(boxes_with_labels):
        color = colors[i % len(colors)]
        legend_html += f'<li><span style="color: {color}; font-weight: bold;">■</span> {box["label"]} (Page {box.get("page", 1)})</li>'
    
    legend_html += '</ul></div>'
    return legend_html

# -------------------------------
# Format Preview Functions
# -------------------------------
def show_format_preview(df, format_type):
    """Show preview of JSON format."""
    if df is None or df.empty:
        return "No data available"
    
    # Limit preview to first 3 rows for display
    preview_df = df.head(3)
    json_data = dataframe_to_json(preview_df, format_type)
    
    if format_type == "row_wise":
        preview_text = "Row-wise format (array of objects):\n"
        preview_text += json.dumps(json_data, indent=2)[:500] + "..."
    elif format_type == "column_wise":
        preview_text = "Column-wise format (object with arrays):\n"
        preview_text += json.dumps(json_data, indent=2)[:500] + "..."
    elif format_type == "table":
        preview_text = "Table format (array of records):\n"
        preview_text += json.dumps(json_data, indent=2)[:500] + "..."
    
    return preview_text

# -------------------------------
# Main Application
# -------------------------------
def main():
    st.markdown('<div class="main-header">📄 Complete PDF Data Extractor</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if "extracted_headers" not in st.session_state:
        st.session_state.extracted_headers = {}
    if "extracted_tables" not in st.session_state:
        st.session_state.extracted_tables = []
    if "boxes" not in st.session_state:
        st.session_state.boxes = []
    if "images" not in st.session_state:
        st.session_state.images = []
    if "current_page" not in st.session_state:
        st.session_state.current_page = 0
    if "selected_format" not in st.session_state:
        st.session_state.selected_format = "row_wise"

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["📝 Header Extraction", "📊 Interactive Table Extraction", "📋 Combined Results"])
        
        with tab1:
            st.markdown('<div class="section-header">📝 Header Data Extraction</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔍 Extract Header Data", key="extract_headers"):
                    with st.spinner("⏳ Extracting header data..."):
                        try:
                            # Extract text and tables
                            pdf_text, tables_data = extract_text_and_tables(uploaded_file)
                            
                            # Extract header data using regex
                            header_data = extract_data_using_regex(pdf_text)
                            
                            # Process automatic table extraction
                            invoice_details = process_invoice_table(tables_data)
                            
                            # Combine data
                            complete_data = {**header_data, "invoice_details": invoice_details}
                            
                            st.session_state.extracted_headers = complete_data
                            st.success("✅ Header data extracted successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error extracting header data: {str(e)}")
            
            with col2:
                if st.session_state.extracted_headers:
                    st.markdown("### 📊 Extraction Summary")
                    header_count = len([v for v in st.session_state.extracted_headers.values() if v and v != "Not Provided"])
                    st.write(f"- **Header fields extracted:** {header_count}")
                    st.write(f"- **Invoice details:** {len(st.session_state.extracted_headers.get('invoice_details', []))}")
            
            # Display extracted header data
            if st.session_state.extracted_headers:
                st.markdown('<div class="section-header">📋 Extracted Header Data</div>', unsafe_allow_html=True)
                
                # Display in expandable sections
                with st.expander("📝 Payment Details", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Advice Date:** {st.session_state.extracted_headers.get('advice_date', 'Not Found')}")
                        st.write(f"**Advice Reference:** {st.session_state.extracted_headers.get('advice_ref', 'Not Found')}")
                        st.write(f"**Transaction Type:** {st.session_state.extracted_headers.get('transaction_type', 'Not Found')}")
                        st.write(f"**Sub Payment Type:** {st.session_state.extracted_headers.get('sub_payment_type', 'Not Found')}")
                    with col2:
                        st.write(f"**Debit Amount:** {st.session_state.extracted_headers.get('debit_amount', 'Not Found')}")
                        st.write(f"**Remittance Amount:** {st.session_state.extracted_headers.get('remittance_amount', 'Not Found')}")
                        st.write(f"**Value Date:** {st.session_state.extracted_headers.get('value_date', 'Not Found')}")
                        st.write(f"**Handling Fee:** {st.session_state.extracted_headers.get('handling_fee', 'Not Found')}")
                
                with st.expander("🏦 Beneficiary Information"):
                    st.write(f"**Name:** {st.session_state.extracted_headers.get('beneficiary_name', 'Not Found')}")
                    st.write(f"**Bank:** {st.session_state.extracted_headers.get('beneficiary_bank', 'Not Found')}")
                    st.write(f"**Account No:** {st.session_state.extracted_headers.get('beneficiary_account_no', 'Not Found')}")
                
                with st.expander("👤 Remitter Information"):
                    st.write(f"**Name:** {st.session_state.extracted_headers.get('remitter_name', 'Not Found')}")
                    st.write(f"**Bank:** {st.session_state.extracted_headers.get('remitting_bank', 'Not Found')}")
                
                with st.expander("📧 Recipient Information"):
                    st.write(f"**Name:** {st.session_state.extracted_headers.get('receipt_name', 'Not Found')}")
                    st.write(f"**Email:** {st.session_state.extracted_headers.get('receipt_email', 'Not Found')}")
                
                # Display invoice details table
                if st.session_state.extracted_headers.get('invoice_details'):
                    st.markdown('<div class="section-header">📊 Invoice Details Table</div>', unsafe_allow_html=True)
                    df_invoice = pd.DataFrame(st.session_state.extracted_headers['invoice_details'])
                    st.dataframe(df_invoice, use_container_width=True)
        
        with tab2:
            st.markdown('<div class="section-header">📊 Interactive Table Extraction</div>', unsafe_allow_html=True)
            
            # Generate images if not already done or if file changed
            if not st.session_state.images or st.session_state.get('current_file_name') != uploaded_file.name:
                with st.spinner("⏳ Converting PDF to images..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    st.session_state.images = convert_pdf_to_images(uploaded_file)
                    st.session_state.current_file_name = uploaded_file.name
                    # Reset boxes for new file
                    st.session_state.boxes = []
                    st.session_state.extracted_tables = []
            
            if st.session_state.images:
                # Main layout with annotation and preview
                main_col, preview_col = st.columns([2, 1])
                
                with main_col:
                    st.markdown("### 🖼️ PDF Page Viewer & Annotation")
                    
                    # Controls
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        page_num = st.selectbox("Select page", range(len(st.session_state.images)), 
                                              format_func=lambda x: f"Page {x + 1}")
                        st.session_state.current_page = page_num
                    
                    with col2:
                        zoom = st.slider("🔍 Zoom Level", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
                    
                    with col3:
                        label_text = st.text_input("✏️ Label for Table", f"table_{len(st.session_state.extracted_tables) + 1}")
                    
                    original_image = st.session_state.images[page_num]
                    width, height = original_image.size
                    zoomed_size = (int(width * zoom), int(height * zoom))
                    display_image = original_image.resize(zoomed_size)
                    
                    st.markdown("🎯 **Draw a box around the table to extract**")
                    st.info("📝 Instructions: Select 'rect' drawing mode and draw a rectangle around the table you want to extract")
                    
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",
                        stroke_width=2,
                        stroke_color="red",
                        background_image=display_image,
                        update_streamlit=True,
                        width=zoomed_size[0],
                        height=zoomed_size[1],
                        drawing_mode="rect",
                        key=f"canvas_{page_num}_{zoom}"
                    )
                    
                    # Extract button
                    if st.button("➕ Extract Table from Selection", type="primary"):
                        if canvas_result.json_data and canvas_result.json_data["objects"]:
                            rect = canvas_result.json_data["objects"][-1]
                            x1 = int(rect["left"] / zoom)
                            y1 = int(rect["top"] / zoom)
                            x2 = int(x1 + rect["width"] / zoom)
                            y2 = int(y1 + rect["height"] / zoom)
                            coords = (x1, y1, x2, y2)
                            
                            with st.spinner("⏳ Extracting table from selection..."):
                                extracted_df = extract_table_with_pdfplumber(uploaded_file, page_num, coords)
                            
                            if extracted_df is not None:
                                st.session_state.extracted_tables.append({
                                    "name": label_text,
                                    "data": extracted_df,
                                    "page": page_num + 1
                                })
                                st.session_state.boxes.append({
                                    "label": label_text, 
                                    "coords": coords,
                                    "page": page_num + 1
                                })
                                st.success(f"✅ Table '{label_text}' extracted successfully!")
                                st.rerun()
                            else:
                                st.error("❌ No table found in selection. Try adjusting your selection.")
                        else:
                            st.warning("⚠️ Please draw a bounding box first.")
                    
                    # Show annotated image if boxes exist
                    if st.session_state.boxes:
                        current_page_boxes = [box for box in st.session_state.boxes if box.get('page', 1) == page_num + 1]
                        if current_page_boxes:
                            st.markdown("### 🏷️ Annotated View")
                            annotated_img = draw_annotation(original_image, current_page_boxes)
                            st.image(annotated_img, caption=f"Annotated Page {page_num + 1}", use_column_width=True)
                
                with preview_col:
                    st.markdown("### 📋 Extracted Tables")
                    
                    if st.session_state.extracted_tables:
                        # Show annotation legend
                        if st.session_state.boxes:
                            legend_html = create_annotation_legend(st.session_state.boxes)
                            st.markdown(legend_html, unsafe_allow_html=True)
                        
                        # Table selector
                        selected_table_idx = st.selectbox(
                            "Select table to view",
                            range(len(st.session_state.extracted_tables)),
                            format_func=lambda x: f"{st.session_state.extracted_tables[x]['name']} (Page {st.session_state.extracted_tables[x]['page']})"
                        )
                        
                        if selected_table_idx is not None:
                            table_info = st.session_state.extracted_tables[selected_table_idx]
                            
                            # Show table data
                            st.markdown(f"#### 📊 {table_info['name']}")
                            st.dataframe(table_info['data'], use_container_width=True, height=200)
                            
                            # Format selection
                            st.markdown("#### 📝 JSON Format Selection")
                            format_options = {
                                "row_wise": "Row-wise (Array of Objects)",
                                "column_wise": "Column-wise (Object with Arrays)",
                                "table": "Table (Array of Records)"
                            }
                            
                            selected_format = st.radio(
                                "Select output format:",
                                options=list(format_options.keys()),
                                format_func=lambda x: format_options[x],
                                key=f"format_radio_{selected_table_idx}"
                            )
                            
                            # Show format preview
                            st.markdown("#### 👀 Format Preview")
                            preview_text = show_format_preview(table_info['data'], selected_format)
                            st.markdown(f'<div class="format-preview"><pre>{preview_text}</pre></div>', 
                                      unsafe_allow_html=True)
                            
                            # Generate and show complete JSON
                            json_output = dataframe_to_json(table_info['data'], selected_format)
                            json_str = json.dumps(json_output, indent=2, ensure_ascii=False)
                            
                            # Download button
                            st.download_button(
                                f"📥 Download {table_info['name']} JSON",
                                json_str,
                                file_name=f"{table_info['name']}_{selected_format}.json",
                                mime="application/json",
                                key=f"download_{selected_table_idx}",
                                use_container_width=True
                            )
                            
                            # Delete table button
                            if st.button(f"🗑️ Delete {table_info['name']}", key=f"delete_{selected_table_idx}"):
                                st.session_state.extracted_tables.pop(selected_table_idx)
                                # Remove corresponding box
                                st.session_state.boxes = [box for box in st.session_state.boxes 
                                                        if box['label'] != table_info['name']]
                                st.rerun()
                    else:
                        st.info("📝 No tables extracted yet. Draw a box around a table to extract it.")
            else:
                st.error("❌ Failed to convert PDF to images. Please try uploading the file again.")
        
        with tab3:
            st.markdown('<div class="section-header">📋 Combined Results & Export</div>', unsafe_allow_html=True)
            
            # Combine all extracted data
            combined_data = {}
            
            # Add header data
            if st.session_state.extracted_headers:
                combined_data.update(st.session_state.extracted_headers)
            
            # Add interactive table data
            if st.session_state.extracted_tables:
                combined_data["interactive_tables"] = []
                for table_info in st.session_state.extracted_tables:
                    combined_data["interactive_tables"].append({
                        "name": table_info["name"],
                        "page": table_info["page"],
                        "data": table_info["data"].to_dict(orient='records')
                    })
            
            if combined_data:
                st.markdown("### 📊 Complete Data Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    header_count = len([v for k, v in combined_data.items() 
                                      if k != "invoice_details" and k != "interactive_tables" 
                                      and v and v != "Not Provided"])
                    st.metric("Header Fields", header_count)
                
                with col2:
                    invoice_count = len(combined_data.get("invoice_details", []))
                    st.metric("Invoice Details", invoice_count)
                
                with col3:
                    interactive_count = len(combined_data.get("interactive_tables", []))
                    st.metric("Interactive Tables", interactive_count)
                
                # Show annotation legend if available
                if st.session_state.boxes:
                    legend_html = create_annotation_legend(st.session_state.boxes)
                    st.markdown(legend_html, unsafe_allow_html=True)
                
                # Display combined JSON
                st.markdown("### 📄 Complete JSON Output")
                complete_json = json.dumps(combined_data, indent=2, ensure_ascii=False, default=str)
                st.code(complete_json, language="json", line_numbers=True)
                
                # Download options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📥 Download Complete JSON",
                        complete_json,
                        file_name="complete_pdf_extraction.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2:
                    if st.session_state.boxes and st.session_state.images:
                        # Create a combined annotated image with all boxes
                        all_images_annotated = []
                        for page_idx, image in enumerate(st.session_state.images):
                            page_boxes = [box for box in st.session_state.boxes if box.get('page', 1) == page_idx + 1]
                            if page_boxes:
                                annotated_img = draw_annotation(image, page_boxes)
                                all_images_annotated.append(annotated_img)
                        
                        if all_images_annotated:
                            # For now, download the first annotated image
                            img_bytes = BytesIO()
                            all_images_annotated[0].save(img_bytes, format="PNG")
                            st.download_button(
                                "📥 Download Annotated Image",
                                img_bytes.getvalue(),
                                file_name="annotated_pdf_image.png",
                                mime="image/png",
                                use_container_width=True
                            )
                
                with col3:
                    # Export all table data as separate JSON files in a zip
                    if st.session_state.extracted_tables:
                        import zipfile
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                            for table_info in st.session_state.extracted_tables:
                                for format_type in ["row_wise", "column_wise", "table"]:
                                    json_data = dataframe_to_json(table_info['data'], format_type)
                                    json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
                                    zip_file.writestr(f"{table_info['name']}_{format_type}.json", json_str)
                        
                        st.download_button(
                            "📥 Download All Tables (ZIP)",
                            zip_buffer.getvalue(),
                            file_name="all_extracted_tables.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                
                # Display all annotated images
                if st.session_state.boxes and st.session_state.images:
                    st.markdown("### 🖼️ All Annotated Images")
                    for page_idx, image in enumerate(st.session_state.images):
                        page_boxes = [box for box in st.session_state.boxes if box.get('page', 1) == page_idx + 1]
                        if page_boxes:
                            annotated_img = draw_annotation(image, page_boxes)
                            st.image(annotated_img, caption=f"Annotated Page {page_idx + 1}", use_column_width=True)
            else:
                st.info("📝 No data extracted yet. Please use the Header Extraction or Interactive Table Extraction tabs first.")
    
    # Enhanced Sidebar with instructions
    with st.sidebar:
        st.markdown("## 🛠️ How to Use")
        st.markdown("""
        ### 📝 Header Extraction
        1. Upload a PDF file
        2. Click "Extract Header Data"
        3. Review extracted fields
        
        ### 📊 Interactive Table Extraction
        1. Select a page from the PDF
        2. Adjust zoom if needed
        3. Draw a box around the table
        4. Choose output format (row-wise, column-wise, table)
        5. Preview and download as JSON
        
        ### 📋 Combined Results
        1. View all extracted data
        2. Download complete JSON
        3. Download annotated images
        4. Export all tables as ZIP
        """)
        
        st.markdown("### 🔧 New Features")
        st.markdown("""
        - **🏷️ Enhanced Annotation**: Multiple colored boxes with legend
        - **👀 Format Preview**: Live preview of JSON formats
        - **📊 Format Selection**: Choose between row-wise, column-wise, or table format
        - **🎨 Visual Feedback**: Color-coded annotations with labels
        - **📦 Bulk Export**: Download all tables in ZIP format
        - **🗑️ Table Management**: Delete individual tables
        """)
        
        st.markdown("### 📋 JSON Format Types")
        st.markdown("""
        - **Row-wise**: `{"rows": [{"col1": "val1"}, ...]}`
        - **Column-wise**: `{"col1": ["val1", "val2"], ...}`
        - **Table**: `[{"col1": "val1"}, {"col1": "val2"}, ...]`
        """)
        
        # Statistics
        if st.session_state.extracted_headers or st.session_state.extracted_tables:
            st.markdown("### 📈 Current Session Stats")
            header_fields = len([v for v in st.session_state.extracted_headers.values() 
                               if v and v != "Not Provided"]) if st.session_state.extracted_headers else 0
            st.metric("Header Fields", header_fields)
            st.metric("Extracted Tables", len(st.session_state.extracted_tables))
            st.metric("Annotations", len(st.session_state.boxes))
        
        # Clear session state button
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.extracted_headers = {}
            st.session_state.extracted_tables = []
            st.session_state.boxes = []
            st.session_state.images = []
            st.session_state.current_page = 0
            st.session_state.selected_format = "row_wise"
            st.success("✅ All data cleared!")
            st.rerun()

if __name__ == "__main__":
    main()