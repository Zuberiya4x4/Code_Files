# 80% worked code
import streamlit as st
import fitz  # PyMuPDF
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Dict, List, Tuple, Any
import pandas as pd

# Configure page
st.set_page_config(
    page_title="PDF Header Field Extractor",
    page_icon="📄",
    layout="wide"
)

class PDFHeaderExtractor:
    def __init__(self):
        # Define expected header fields with more precise patterns
        self.header_patterns = {
            "document_title": [r"payment\s*advice", r"remittance\s*advice", r"transfer\s*advice"],
            "advice_date": [r"advice\s*sending\s*date", r"advice\s*date", r"date\s*of\s*advice", r"sending\s*date"],
            "advice_ref": [r"advice\s*reference\s*no", r"advice\s*ref\s*no", r"advice\s*reference", r"advice\s*ref"],
            "recipient_name": [r"recipient'?s?\s*name\s*and\s*contact\s*information", r"recipient'?s?\s*name", r"company\s*name"],
            "receipt_email": [r"receipt\s*email", r"email", r"e-mail"],
            "transaction_type": [r"transaction\s*type", r"payment\s*type"],
            "sub_payment_type": [r"sub\s*payment\s*type", r"payment\s*sub\s*type"],
            "beneficiary_name": [r"beneficiary'?s?\s*name", r"beneficiary"],
            "beneficiary_bank": [r"beneficiary'?s?\s*bank", r"bank\s*name"],
            "account_number": [r"beneficiary'?s?\s*account", r"account\s*number", r"account\s*no", r"a/c\s*no"],
            "customer_reference": [r"customer\s*reference", r"cust\s*ref", r"customer\s*ref"],
            "debit_amount": [r"debit\s*amount", r"amount\s*debited"],
            "remittance_amount": [r"remittance\s*amount", r"remit\s*amount"],
            "handling_fee": [r"handling\s*fee", r"charges", r"fees"],
            "value_date": [r"value\s*date", r"settlement\s*date"],
            "remitter_name": [r"remitter'?s?\s*name", r"sender\s*name"],
            "remitting_bank": [r"remitting\s*bank", r"sender\s*bank", r"of\s*remitting\s*bank"],
            "instruction_reference": [r"instruction\s*reference", r"instr\s*ref"],
            "other_reference": [r"other\s*reference", r"additional\s*ref"],
            "remitter_to_beneficiary_info": [r"remitter\s*to\s*beneficiary\s*information", r"remitter\s*to\s*beneficiary", r"additional\s*info", r"payment\s*info"]
        }
        
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (0, 128, 0), (128, 128, 0), (0, 0, 128),
            (128, 0, 0), (0, 128, 128), (192, 192, 192), (255, 20, 147),
            (32, 178, 170), (255, 69, 0), (154, 205, 50)
        ]

    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with coordinates from PDF"""
        doc = fitz.open(pdf_path)
        text_blocks = []
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text and len(text) > 1:  # Filter out single characters
                                text_blocks.append({
                                    "text": text,
                                    "bbox": span["bbox"],
                                    "page": page_num,
                                    "font_size": span["size"]
                                })
        
        doc.close()
        return text_blocks

    def find_header_field_matches(self, text_blocks: List[Dict]) -> Dict[str, List[Dict]]:
        """Find potential header field matches with improved accuracy"""
        matches = {}
        
        for field_name, patterns in self.header_patterns.items():
            matches[field_name] = []
            
            for block in text_blocks:
                text_lower = block["text"].lower()
                text_clean = re.sub(r'[^\w\s]', ' ', text_lower).strip()
                
                for pattern in patterns:
                    # Use word boundaries for more precise matching
                    if re.search(f"\\b{pattern}\\b", text_clean):
                        matches[field_name].append(block)
                        break
        
        return matches

    def find_values_for_headers(self, text_blocks: List[Dict], header_matches: Dict) -> Dict[str, Dict]:
        """Find values associated with header fields with improved logic"""
        field_data = {}
        
        for field_name, headers in header_matches.items():
            if not headers:
                continue
                
            best_match = None
            best_values = []
            best_score = float('inf')
            
            for header in headers:
                header_bbox = header["bbox"]
                header_y = header_bbox[1]
                header_x_end = header_bbox[2]
                
                # Look for colon patterns first (field: value)
                # Check if the header text contains a colon and value in the same block
                if ':' in header["text"]:
                    parts = header["text"].split(':', 1)
                    if len(parts) == 2:
                        header_part = parts[0].strip()
                        value_part = parts[1].strip()
                        
                        if value_part and len(value_part) > 1:
                            # Estimate value position within the same block
                            char_width = header["font_size"] * 0.6
                            value_start_x = header_bbox[0] + len(header_part + ': ') * char_width
                            
                            virtual_value = {
                                "text": value_part,
                                "bbox": [value_start_x, header_bbox[1], header_bbox[2], header_bbox[3]],
                                "page": header["page"],
                                "font_size": header["font_size"]
                            }
                            
                            # Update header to only contain the field name part
                            header_only = {
                                **header,
                                "text": header_part,
                                "bbox": [header_bbox[0], header_bbox[1], value_start_x, header_bbox[3]]
                            }
                            
                            best_match = header_only
                            best_values = [virtual_value]
                            best_score = 0
                            break
                
                # If no colon pattern found in header, look for adjacent values
                if best_score == float('inf'):
                    potential_values = []
                    
                    for block in text_blocks:
                        # Skip if it's the header itself
                        if block["text"].lower() == header["text"].lower():
                            continue
                        
                        # Skip if it contains the header text (partial match)
                        if header["text"].lower() in block["text"].lower() and len(header["text"]) > 5:
                            continue
                            
                        block_bbox = block["bbox"]
                        block_x = block_bbox[0]
                        block_y = block_bbox[1]
                        
                        # Calculate distances
                        horizontal_distance = block_x - header_x_end
                        vertical_distance = abs(block_y - header_y)
                        
                        # Check if it's to the right of the header (same line) - prefer this
                        if vertical_distance < 15 and horizontal_distance > -10 and horizontal_distance < 200:
                            score = abs(horizontal_distance) + vertical_distance * 2
                            potential_values.append({
                                "block": block,
                                "score": score,
                                "type": "right"
                            })
                        
                        # Check if it's below the header (next line or two)
                        elif block_y > header_y and vertical_distance < 60:
                            # Prefer values that are somewhat aligned horizontally
                            horizontal_alignment_penalty = abs(block_x - header_bbox[0]) * 0.3
                            score = vertical_distance * 3 + horizontal_alignment_penalty
                            potential_values.append({
                                "block": block,
                                "score": score,
                                "type": "below"
                            })
                    
                    # Sort by score and collect multiple values if they're close
                    if potential_values:
                        potential_values.sort(key=lambda x: x["score"])
                        
                        # Take the best value(s) - if multiple values are very close, take them all
                        closest_values = [potential_values[0]]
                        best_score_val = potential_values[0]["score"]
                        
                        # Look for additional values that might be part of the same field
                        for val in potential_values[1:]:
                            if val["score"] - best_score_val < 20:  # If scores are close
                                # Check if it's on the same line or immediately adjacent
                                prev_bbox = closest_values[-1]["block"]["bbox"]
                                curr_bbox = val["block"]["bbox"]
                                
                                # Same line continuation
                                if (abs(curr_bbox[1] - prev_bbox[1]) < 10 and 
                                    curr_bbox[0] - prev_bbox[2] < 50):
                                    closest_values.append(val)
                                # Next line continuation for multi-line values
                                elif (curr_bbox[1] - prev_bbox[3] < 25 and
                                      abs(curr_bbox[0] - prev_bbox[0]) < 30):
                                    closest_values.append(val)
                        
                        if best_score_val < best_score:
                            best_match = header
                            best_values = [v["block"] for v in closest_values]
                            best_score = best_score_val
            
            if best_match and best_values:
                # Create combined bounding box for header + all values
                all_bboxes = [best_match["bbox"]] + [v["bbox"] for v in best_values]
                
                combined_bbox = [
                    min(bbox[0] for bbox in all_bboxes),
                    min(bbox[1] for bbox in all_bboxes),
                    max(bbox[2] for bbox in all_bboxes),
                    max(bbox[3] for bbox in all_bboxes)
                ]
                
                # Combine all value texts
                value_texts = []
                for value in best_values:
                    clean_text = value["text"].strip()
                    # Remove any remaining header text from the value
                    if best_match["text"].lower() in clean_text.lower():
                        clean_text = re.sub(re.escape(best_match["text"]), "", clean_text, flags=re.IGNORECASE).strip()
                    clean_text = re.sub(r'^[:\s]+', '', clean_text).strip()
                    if clean_text:
                        value_texts.append(clean_text)
                
                combined_value_text = " ".join(value_texts)
                
                field_data[field_name] = {
                    "header": best_match,
                    "values": best_values,
                    "combined_bbox": combined_bbox,
                    "extracted_text": f"{best_match['text']}: {combined_value_text}",
                    "clean_value": combined_value_text
                }
        
        return field_data

    def draw_bounding_boxes(self, pdf_path: str, field_data: Dict) -> Image.Image:
        """Draw bounding boxes on PDF and return as image"""
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)  # First page
        
        # Convert PDF page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            label_font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            label_font = ImageFont.load_default()
        
        color_index = 0
        for field_name, data in field_data.items():
            if "combined_bbox" in data:
                bbox = data["combined_bbox"]
                # Scale coordinates (2x scaling was applied)
                scaled_bbox = [coord * 2 for coord in bbox]
                
                color = self.colors[color_index % len(self.colors)]
                
                # Draw bounding box with thicker lines
                draw.rectangle(scaled_bbox, outline=color, width=4)
                
                # Draw label with better positioning
                label = field_name.replace("_", " ").title()
                
                # Calculate text dimensions
                text_bbox = draw.textbbox((0, 0), label, font=label_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Position label above the bounding box
                label_x = scaled_bbox[0]
                label_y = max(5, scaled_bbox[1] - text_height - 8)
                
                # Draw background for label with padding
                padding = 4
                draw.rectangle(
                    [label_x - padding, label_y - padding, 
                     label_x + text_width + padding, label_y + text_height + padding],
                    fill=color
                )
                draw.text((label_x, label_y), label, fill="white", font=label_font)
                
                color_index += 1
        
        doc.close()
        return img

    def extract_json_data(self, field_data: Dict) -> Dict[str, str]:
        """Extract data as JSON format with cleaned values"""
        json_data = {}
        
        for field_name, data in field_data.items():
            if "clean_value" in data and data["clean_value"]:
                json_data[field_name] = data["clean_value"]
            elif "values" in data and data["values"]:
                # Handle multiple values case
                combined_text = " ".join([v["text"] for v in data["values"]]).strip()
                json_data[field_name] = combined_text
            elif "value" in data:
                json_data[field_name] = data["value"]["text"]
            else:
                json_data[field_name] = ""
        
        return json_data

def main():
    st.title("📄 PDF Header Field Extractor with Enhanced Bounding Boxes")
    st.markdown("Upload a PDF document to automatically detect and extract header fields with improved accuracy")
    
    # Initialize session state
    if 'field_data' not in st.session_state:
        st.session_state.field_data = {}
    if 'json_data' not in st.session_state:
        st.session_state.json_data = {}
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'field_status' not in st.session_state:
        st.session_state.field_status = {}
    
    extractor = PDFHeaderExtractor()
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_pdf.pdf", "wb") as f:
            f.write(uploaded_file.read())
        
        # Process PDF
        with st.spinner("Processing PDF and detecting header fields..."):
            # Extract text blocks
            text_blocks = extractor.extract_text_blocks("temp_pdf.pdf")
            
            # Find header matches
            header_matches = extractor.find_header_field_matches(text_blocks)
            
            # Find values for headers
            field_data = extractor.find_values_for_headers(text_blocks, header_matches)
            
            # Generate image with bounding boxes
            processed_image = extractor.draw_bounding_boxes("temp_pdf.pdf", field_data)
            
            # Extract JSON data
            json_data = extractor.extract_json_data(field_data)
            
            # Store in session state
            st.session_state.field_data = field_data
            st.session_state.json_data = json_data
            st.session_state.processed_image = processed_image
            
            # Initialize field status
            for field_name in field_data.keys():
                if field_name not in st.session_state.field_status:
                    st.session_state.field_status[field_name] = "pending"
        
        st.success(f"Detected {len(field_data)} header fields!")
        
        # Display processed image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Document with Enhanced Bounding Boxes")
            if st.session_state.processed_image:
                st.image(st.session_state.processed_image, caption="Detected Header Fields", use_column_width=True)
                
                # Download button for image
                img_buffer = io.BytesIO()
                st.session_state.processed_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Bounding Box Image",
                    data=img_buffer,
                    file_name="detected_fields.png",
                    mime="image/png"
                )
        
        with col2:
            st.subheader("🔍 Detected Fields")
            
            # Display extracted fields with Accept/Reject buttons
            for field_name, data in st.session_state.field_data.items():
                with st.container():
                    st.markdown(f"**{field_name.replace('_', ' ').title()}:**")
                    
                    if "extracted_text" in data:
                        st.text(data["extracted_text"])
                    
                    # Show clean value if available
                    if "clean_value" in data and data["clean_value"]:
                        st.code(f"Extracted Value: {data['clean_value']}")
                    
                    # Accept/Reject buttons
                    col_accept, col_reject = st.columns(2)
                    
                    with col_accept:
                        if st.button(f"✅ Accept", key=f"accept_{field_name}"):
                            st.session_state.field_status[field_name] = "accepted"
                            try:
                                st.rerun()
                            except AttributeError:
                                st.experimental_rerun()
                    
                    with col_reject:
                        if st.button(f"❌ Reject", key=f"reject_{field_name}"):
                            st.session_state.field_status[field_name] = "rejected"
                            try:
                                st.rerun()
                            except AttributeError:
                                st.experimental_rerun()
                    
                    # Status indicator
                    status = st.session_state.field_status.get(field_name, "pending")
                    if status == "accepted":
                        st.success("✅ Accepted")
                    elif status == "rejected":
                        st.error("❌ Rejected")
                    else:
                        st.warning("⏳ Pending")
                    
                    st.divider()
        
        # JSON Output Section
        st.subheader("📊 Extracted Data (JSON Format)")
        
        # Filter data based on accepted fields
        accepted_data = {
            field_name: value 
            for field_name, value in st.session_state.json_data.items()
            if st.session_state.field_status.get(field_name) == "accepted"
        }
        
        # Display JSON
        if accepted_data:
            st.json(accepted_data)
            
            # Submit button
            if st.button("🚀 Submit Final JSON", type="primary"):
                st.success("✅ Data submitted successfully!")
                st.balloons()
                
                # Download JSON
                json_str = json.dumps(accepted_data, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name="extracted_data.json",
                    mime="application/json"
                )
        else:
            st.info("Accept some fields to generate JSON output")
        
        # Summary statistics
        st.subheader("📈 Processing Summary")
        total_fields = len(st.session_state.field_data)
        accepted_fields = len([k for k, v in st.session_state.field_status.items() if v == "accepted"])
        rejected_fields = len([k for k, v in st.session_state.field_status.items() if v == "rejected"])
        pending_fields = total_fields - accepted_fields - rejected_fields
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Fields", total_fields)
        col2.metric("Accepted", accepted_fields)
        col3.metric("Rejected", rejected_fields)
        col4.metric("Pending", pending_fields)
        
        # Debug information (optional)
        if st.checkbox("Show Debug Information"):
            st.subheader("🔧 Debug Information")
            st.write("Raw field data:")
            st.json(st.session_state.field_data)

if __name__ == "__main__":
    main()