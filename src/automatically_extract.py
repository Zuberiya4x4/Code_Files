import streamlit as st
import pandas as pd
import re
from datetime import datetime
import io
from typing import Dict, List, Optional
import json

# For PDF processing - you'll need to install these packages
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# For image processing - optional
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

class PaymentAdviceExtractor:
    def __init__(self):
        # More precise regex patterns to extract only the required values
        self.header_patterns = {
            "advice_date": [
                r"Advice Date:\s*(\d{1,2}\s+\w+\s+\d{4})",  # From the processed output
                r"Advice sending date[:\s]*(\d{2} \w{3} \d{4})",
                r'advice\s+date\s*[:\s]+(\d{1,2}\s+\w+\s+\d{4})',
            ],
            "advice_ref": [
                r"Advice Reference:\s*([A-Za-z0-9\-]+)",  # From the processed output
                r"Advice reference no[:\s]*([A-Za-z0-9\-]+)",
            ],
            "receipt_name": [
                r"Receipt Name:\s*([^A-Z@]+?)(?=\s+[A-Z@])",  # Clean company name only
                r"Recipient's name and contact information:[\s]*([^\n@]+?)(?=\s+[\w\.-]+@)",
            ],
            "receipt_email": [
                r"Receipt Email:\s*([\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,})",  # From processed output
                r"([\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,})",  # Any email pattern
            ],
            "transaction_type": [
                r"Transaction Type:\s*([A-Za-z\s]+?)(?=\s+Sub payment type|\n|$)",  # From processed output
                r"Transaction type[:\s]*([A-Za-z\s]+?)(?=\s+Sub payment type|\n|$)",
            ],
            "sub_payment_type": [
                r"Sub Payment Type:\s*([A-Za-z\s]+?)(?=\s+Beneficiary|\n|$)",  # From processed output
                r"Sub payment type[:\s]*([A-Za-z\s]+?)(?=\s+Beneficiary|\n|$)",
            ],
            "beneficiary_name": [
                r"Beneficiary Name:\s*([^D]+?)(?=\s+Beneficiary Bank|\n|$)",  # From processed output
                r"Beneficiary's\s+name[:\s]*([^D]+?)(?=\s+Debit amount|\n|$)",
            ],
            "beneficiary_bank": [
                r"Beneficiary Bank:\s*([A-Za-z0-9\s&]+BANK[A-Za-z0-9\s]*?)(?=\s+Account Number|\n|$)",  # From processed output
                r"Beneficiary's\s+bank[:\s]*([A-Za-z0-9\s&,.-]+BANK[A-Za-z0-9\s]*?)(?=\s+Remittance|\n|$)",
            ],
            "account_number": [
                r"Account Number:\s*([A-Za-z0-9\*]+)",  # From processed output
                r"Beneficiary's account[:\s]*([A-Za-z0-9\*]+)",
            ],
            "customer_reference": [
                r"Customer Reference:\s*([A-Za-z0-9\-]+)",  # From processed output
                r"Customer reference[:\s]*([A-Za-z0-9\-]+)",
            ],
            "debit_amount": [
                r"Debit Amount:\s*(INR[\d,]+\.?\d*)",  # From processed output
                r"Debit amount[:\s]*(INR[\d,]+\.?\d*)",
            ],
            "remittance_amount": [
                r"Remittance Amount:\s*(INR[\d,]+\.?\d*)",  # From processed output
                r"Remittance amount[:\s]*(INR[\d,]+\.?\d*)",
            ],
            "handling_fee": [
                r"Handling Fee:\s*(Collect from (?:Remitter|Beneficiary))",  # From processed output
                r"(Collect from (?:Remitter|Beneficiary))",
            ],
            "value_date": [
                r"Value Date:\s*(\d{1,2}\s+\w+\s+\d{4})",  # From processed output
                r"Value date[:\s]*(\d{2} \w{3} \d{4})",
            ],
            "remitter_name": [
                r"Remitter Name:\s*([A-Za-z0-9\s&]+?)(?=\s+Remitting Bank|\n|$)",  # From processed output
                r"Remitter's name[:\s]*([A-Za-z0-9\s&]+?)(?=\s+Remitting bank|\n|$)",
            ],
            "remitting_bank": [
                r"Remitting Bank:\s*([A-Za-z0-9\s&]+?)(?=\s+Instruction|\n|$)",  # From processed output
                r"Remitting bank[:\s]*([A-Za-z0-9\s&]+?)(?=\s+Instruction|\n|$)",
            ],
            "instruction_reference": [
                r"Instruction Reference:\s*([A-Za-z0-9\-]+)",  # From processed output
                r"Instruction reference[:\s]*([A-Za-z0-9\-]+)",
            ],
            "other_reference": [
                r"Other Reference:\s*([A-Za-z0-9\-]+)",  # From processed output
                r"Other reference[:\s]*([A-Za-z0-9\-]+)",
            ],
            "remitter_to_beneficiary_info": [
                r"Remitter to Beneficiary Info:\s*([A-Za-z0-9\s]+?)(?=\s+Important notes|\n|$)",  # From processed output
                r"Remitter to beneficiary information[:\s]*([A-Za-z0-9\s]+?)(?=\s+Important notes|\n|$)",
            ]
        }
    
    def extract_from_text(self, text: str) -> Dict[str, str]:
        """Extract header values from text using regex patterns"""
        extracted_data = {}
        
        # Clean the text - remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Additional patterns for raw PDF text format
        raw_patterns = {
            "advice_date": r"Advice sending date[:\s]*(\d{1,2}\s+\w+\s+\d{4})",
            "advice_ref": r"Advice reference no[:\s]*([A-Za-z0-9\-]+)",
            "receipt_name": r"Recipient's name and contact information:[\s]*([A-Za-z0-9\s&]+?)(?=\s+[\w\.-]+@)",
            "receipt_email": r"([\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,})",
            "transaction_type": r"Transaction type[:\s]*([A-Za-z\s]+?)(?=\s+Sub payment type)",
            "sub_payment_type": r"Sub payment type[:\s]*([A-Za-z\s]+?)(?=\s+Beneficiary)",
            "beneficiary_name": r"Beneficiary's\s+name[:\s]*([A-Za-z0-9\s&]+?)(?=\s+Debit amount)",
            "beneficiary_bank": r"Beneficiary's\s+bank[:\s]*([A-Za-z0-9\s&]+?)(?=\s+Remittance)",
            "account_number": r"Beneficiary's account[:\s]*([A-Za-z0-9\*]+)",
            "customer_reference": r"Customer reference[:\s]*([A-Za-z0-9\-]+)",
            "debit_amount": r"Debit amount[:\s]*(INR[\d,]+\.?\d*)",
            "remittance_amount": r"Remittance amount[:\s]*(INR[\d,]+\.?\d*)",
            "handling_fee": r"(Collect from (?:Remitter|Beneficiary))",
            "value_date": r"Value date[:\s]*(\d{1,2}\s+\w+\s+\d{4})",
            "remitter_name": r"Remitter's name[:\s]*([A-Za-z0-9\s&]+?)(?=\s+Remitting bank)",
            "remitting_bank": r"Remitting bank[:\s]*([A-Za-z0-9\s&]+?)(?=\s+Instruction reference)",
            "instruction_reference": r"Instruction reference[:\s]*([A-Za-z0-9\-]+)",
            "other_reference": r"Other reference[:\s]*([A-Za-z0-9\-]+)",
            "remitter_to_beneficiary_info": r"Remitter to beneficiary information[:\s]*([A-Za-z0-9\s]+?)(?=\s+Important notes)",
        }
        
        # Try to extract each field
        for field, patterns in self.header_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    # Clean extracted value
                    value = re.sub(r'\s+', ' ', value)
                    value = value.strip()
                    if value and len(value) > 1:  # Ensure meaningful extraction
                        extracted_data[field] = value
                        break
            
            # If not found in processed format, try raw format
            if field not in extracted_data and field in raw_patterns:
                match = re.search(raw_patterns[field], text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    value = re.sub(r'\s+', ' ', value)
                    value = value.strip()
                    if value and len(value) > 1:
                        extracted_data[field] = value
        
        return extracted_data
    
    def extract_from_pdf(self, pdf_file) -> Dict[str, str]:
        """Extract text from PDF and then extract header values"""
        if not PDF_AVAILABLE:
            return {"error": "PDF processing libraries not available"}
        
        try:
            # Try pdfplumber first (better for structured documents)
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if not text.strip():
                # Fallback to PyPDF2
                pdf_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            result = self.extract_from_text(text)
            result['raw_text'] = text  # Store raw text for debugging
            return result
        
        except Exception as e:
            return {"error": f"PDF processing error: {str(e)}"}
    
    def extract_from_image(self, image_file) -> Dict[str, str]:
        """Extract text from image using OCR and then extract header values"""
        if not OCR_AVAILABLE:
            return {"error": "OCR libraries not available"}
        
        try:
            image = Image.open(image_file)
            text = pytesseract.image_to_string(image)
            result = self.extract_from_text(text)
            result['raw_text'] = text  # Store raw text for debugging
            return result
        
        except Exception as e:
            return {"error": f"OCR processing error: {str(e)}"}
    
    def extract_from_manual_text(self, text: str) -> Dict[str, str]:
        """Extract from manual text input"""
        result = self.extract_from_text(text)
        result['raw_text'] = text  # Store raw text for debugging
        return result

def main():
    st.set_page_config(
        page_title="Payment Advice Header Extractor",
        page_icon="💳",
        layout="wide"
    )
    
    st.title("💳 Payment Advice Header Extractor")
    st.markdown("Upload payment advice documents to automatically extract header information")
    
    # Initialize extractor
    extractor = PaymentAdviceExtractor()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # File upload section
    st.sidebar.subheader("Upload Files")
    upload_type = st.sidebar.selectbox(
        "Select input type:",
        ["PDF Files", "Image Files", "Text Input"]
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        extracted_data_list = []
        
        if upload_type == "PDF Files":
            if not PDF_AVAILABLE:
                st.error("PDF processing libraries not installed. Please install PyPDF2 and pdfplumber.")
                st.code("pip install PyPDF2 pdfplumber")
            else:
                uploaded_files = st.file_uploader(
                    "Upload PDF files",
                    type=['pdf'],
                    accept_multiple_files=True
                )
                
                if uploaded_files:
                    for i, uploaded_file in enumerate(uploaded_files):
                        st.subheader(f"Processing: {uploaded_file.name}")
                        
                        with st.spinner(f"Extracting from {uploaded_file.name}..."):
                            extracted_data = extractor.extract_from_pdf(uploaded_file)
                            extracted_data['filename'] = uploaded_file.name
                            extracted_data_list.append(extracted_data)
        
        elif upload_type == "Image Files":
            if not OCR_AVAILABLE:
                st.error("OCR libraries not installed. Please install pytesseract and Pillow.")
                st.code("pip install pytesseract Pillow")
            else:
                uploaded_files = st.file_uploader(
                    "Upload image files",
                    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
                    accept_multiple_files=True
                )
                
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        st.subheader(f"Processing: {uploaded_file.name}")
                        
                        with st.spinner(f"Extracting from {uploaded_file.name}..."):
                            extracted_data = extractor.extract_from_image(uploaded_file)
                            extracted_data['filename'] = uploaded_file.name
                            extracted_data_list.append(extracted_data)
        
        else:  # Text Input
            st.subheader("Enter Payment Advice Text")
            text_input = st.text_area(
                "Paste your payment advice text here:",
                height=300,
                placeholder="Paste the text content of your payment advice here..."
            )
            
            if st.button("Extract Headers") and text_input:
                with st.spinner("Extracting headers..."):
                    extracted_data = extractor.extract_from_manual_text(text_input)
                    extracted_data['filename'] = "Text Input"
                    extracted_data_list.append(extracted_data)
    
    with col2:
        st.header("Extracted Data")
        
        if extracted_data_list:
            # Display results
            for i, data in enumerate(extracted_data_list):
                st.subheader(f"Results for: {data.get('filename', f'File {i+1}')}")
                
                if 'error' in data:
                    st.error(data['error'])
                else:
                    # Create a clean display of extracted data
                    display_data = {}
                    field_labels = {
                        'advice_date': 'Advice Date',
                        'advice_ref': 'Advice Reference',
                        'receipt_name': 'Receipt Name',
                        'receipt_email': 'Receipt Email',
                        'sub_payment_type': 'Sub Payment Type',
                        'transaction_type': 'Transaction Type',
                        'beneficiary_name': 'Beneficiary Name',
                        'beneficiary_bank': 'Beneficiary Bank',
                        'account_number': 'Beneficiary Account No',  # Changed to match expected output
                        'customer_reference': 'Customer Reference',
                        'debit_amount': 'Debit Amount',
                        'remittance_amount': 'Remittance Amount',
                        'handling_fee': 'Handling Fee',
                        'value_date': 'Value Date',
                        'remitter_name': 'Remitter Name',
                        'remitting_bank': 'Remitting Bank',
                        'instruction_reference': 'Instruction Reference',
                        'other_reference': 'Other Reference',
                        'remitter_to_beneficiary_info': 'Remitter to Beneficiary Info'
                    }
                    
                    # Map fields to expected JSON output format
                    json_field_mapping = {
                        'advice_date': 'advice_date',
                        'advice_ref': 'advice_ref',
                        'receipt_name': 'receipt_name',
                        'receipt_email': 'receipt_email',
                        'sub_payment_type': 'sub_payment_type',
                        'transaction_type': 'transaction_type',
                        'beneficiary_name': 'beneficiary_name',
                        'beneficiary_bank': 'beneficiary_bank',
                        'account_number': 'beneficiary_account_no',  # Map to expected field name
                        'customer_reference': 'customer_reference',
                        'debit_amount': 'debit_amount',
                        'remittance_amount': 'remittance_amount',
                        'handling_fee': 'handling_fee',
                        'value_date': 'value_date',
                        'remitter_name': 'remitter_name',
                        'remitting_bank': 'remitting_bank',
                        'instruction_reference': 'instruction_reference',
                        'other_reference': 'other_reference',
                        'remitter_to_beneficiary_info': 'remitter_to_beneficiary_info'
                    }
                    
                    for field, label in field_labels.items():
                        if field in data and data[field]:
                            display_data[label] = data[field]
                    
                    if display_data:
                        for label, value in display_data.items():
                            st.write(f"**{label}:** {value}")
                    else:
                        st.warning("No header information could be extracted from this document.")
                
                st.divider()
            
            # Create downloadable results
            if len(extracted_data_list) > 0:
                st.subheader("Download Results")
                
                # Prepare data for export
                export_data = []
                for data in extracted_data_list:
                    if 'error' not in data:
                        # Create a clean export record with proper field names
                        export_record = {}
                        for internal_field, json_field in json_field_mapping.items():
                            if internal_field in data and data[internal_field]:
                                export_record[json_field] = data[internal_field]
                            else:
                                export_record[json_field] = ""  # Empty string for missing fields
                        
                        # Add filename for reference
                        export_record['filename'] = data.get('filename', '')
                        export_data.append(export_record)
                
                if export_data:
                    df = pd.DataFrame(export_data)
                    
                    # Display as table
                    st.dataframe(df, use_container_width=True)
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV Download
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        csv_string = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv_string,
                            file_name=f"payment_advice_headers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # JSON Download - FIXED: Use standard json.dumps instead of pd.io.json.dumps
                        json_data = df.to_dict('records')  # Convert to list of dictionaries
                        json_string = json.dumps(json_data, indent=2)
                        
                        st.download_button(
                            label="📋 Download JSON",
                            data=json_string,
                            file_name=f"payment_advice_headers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    # Show JSON preview
                    st.subheader("JSON Preview")
                    st.json(json_data)
    
    # Instructions and tips
    st.sidebar.markdown("---")
    st.sidebar.header("Tips for Better Extraction")
    st.sidebar.markdown("""
    **For better results:**
    - Ensure documents are clear and readable
    - Use high-quality scans for image files
    - Make sure text is not skewed or rotated
    - Check that important information is visible
    
    **Expected Fields:**
    - Advice Date & Reference
    - Receipt Name & Email
    - Payment & Transaction Type
    - Beneficiary Information
    - Amount & Fee Details
    - Remitter Information
    - Reference Numbers
    
    **Supported formats:**
    - PDF files (text-based preferred)
    - Image files (PNG, JPG, JPEG, TIFF, BMP)
    - Direct text input
    """)
    
    # Debug section
    st.sidebar.markdown("---")
    st.sidebar.header("Debug Options")
    show_raw_text = st.sidebar.checkbox("Show Raw Extracted Text")
    
    if show_raw_text and extracted_data_list:
        st.subheader("Raw Text Debug")
        for i, data in enumerate(extracted_data_list):
            if 'raw_text' in data:
                st.text_area(f"Raw text from {data.get('filename', f'File {i+1}')}", 
                           data['raw_text'], height=200)

if __name__ == "__main__":
    main()