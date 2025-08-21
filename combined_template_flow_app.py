# app.py

import streamlit as st
import json
import os
import tempfile
import zipfile
from io import BytesIO
from combined_template_flow import (
    process_complete_document, 
    draw_bounding_boxes_on_document,
    create_complete_output,
    FIELD_COLORS
)

# Page configuration
st.set_page_config(page_title="Complete Document Auto-Tagger", page_icon="📄", layout="wide")
st.title("📄 Complete Document Auto-Tagger")
st.markdown("Upload a JSON coordinates file and PDF document to automatically extract headers and tables with bounding boxes")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'annotated_images' not in st.session_state:
    st.session_state.annotated_images = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'pdf_filename' not in st.session_state:
    st.session_state.pdf_filename = None
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None

# Create two columns for file upload
col1, col2 = st.columns(2)
with col1:
    json_file = st.file_uploader("Upload JSON Coordinates File", type="json", help="Upload JSON file containing coordinate data for headers")
with col2:
    pdf_file = st.file_uploader("Upload PDF Document", type=["pdf"], help="Upload PDF document to process")

# Process files when both are uploaded
if json_file and pdf_file:
    # Store PDF bytes and filename in session state
    st.session_state.pdf_bytes = pdf_file.read()
    st.session_state.pdf_filename = pdf_file.name
    st.subheader(f"Processing: {pdf_file.name}")
    
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(st.session_state.pdf_bytes)
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
                    st.error("No head keys found in JSON file. Please check the JSON structure.")
                    st.write("### JSON Structure Received:")
                    st.json(coordinates)
                    st.stop()
                    
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
        
        # Check if annotated images are available, if not try to regenerate them
        if not st.session_state.annotated_images and st.session_state.get('pdf_bytes'):
            if st.button("Regenerate Annotated Images"):
                with st.spinner("Regenerating annotated images..."):
                    # Write the PDF bytes to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                        tmp_pdf.write(st.session_state.pdf_bytes)
                        temp_pdf_path = tmp_pdf.name
                    
                    try:
                        st.session_state.annotated_images = draw_bounding_boxes_on_document(
                            temp_pdf_path,
                            st.session_state.results["entity_details"],
                            st.session_state.results["table_data"],
                            "pdf"
                        )
                        if not st.session_state.annotated_images:
                            st.error("Failed to generate annotated images.")
                        else:
                            st.success("Annotated images regenerated successfully!")
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error regenerating annotated images: {str(e)}")
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_pdf_path):
                            os.unlink(temp_pdf_path)
        
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
            st.info("No annotated images available. Click 'Regenerate Annotated Images' to try again.")
    
    with tab4:
        st.header("📥 Download Results")
        
        # Get base filename from session state
        base_filename = st.session_state.pdf_filename.split('.')[0] if st.session_state.pdf_filename else "document"
        
        # Create complete output
        complete_output = create_complete_output(result, base_filename)
        
        # Prepare different JSON outputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Complete Results (Headers + Tables)")
            
            # Convert to JSON
            complete_json_str = json.dumps(complete_output, indent=2)
            st.download_button(
                label="📥 Download Complete Results",
                data=complete_json_str,
                file_name=f"complete_results_{base_filename}.json",
                mime="application/json"
            )
        
        with col2:
            st.subheader("Header Fields Only")
            
            # Header fields only
            header_fields = complete_output["header_fields"]
            header_json_str = json.dumps(header_fields, indent=2)
            st.download_button(
                label="📥 Download Header Fields",
                data=header_json_str,
                file_name=f"header_fields_{base_filename}.json",
                mime="application/json"
            )
        
        with col3:
            st.subheader("Tables Only")
            
            # Tables only
            table_data = complete_output["tables"]
            table_json_str = json.dumps(table_data, indent=2)
            st.download_button(
                label="📥 Download Tables",
                data=table_json_str,
                file_name=f"tables_{base_filename}.json",
                mime="application/json"
            )
        
        # Download all annotated pages as ZIP
        if st.session_state.annotated_images:
            st.subheader("Annotated Images")
            
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
                file_name=f"annotated_pages_{base_filename}.zip",
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
**Complete Results JSON:**
- Combined header and table data
- Processing summary
- Document metadata
**Header Fields JSON:**
- Coordinate-based extraction results
- Regex-based extraction results
- Detailed coordinate information
**Tables JSON:**
- Table data with coordinates
- Structured table content
- Page-wise organization
""")