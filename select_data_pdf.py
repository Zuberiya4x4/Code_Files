import streamlit as st
import fitz  # PyMuPDF
import re
import pdfplumber
import json
import tempfile
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to extract text and tables from PDF
def extract_text_and_tables(pdf_path):
    full_text = ""
    tables_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += "\n" + text
                page_tables = page.extract_tables()
                if page_tables:
                    tables_data.extend(page_tables)
    except Exception as e:
        st.error(f"Error extracting text and tables: {e}")
    return full_text, tables_data

# Function to extract fields using regex
def extract_data_using_regex(pdf_text, selected_fields):
    patterns = {
        "advice_date": r"Advice sending date[:\s]*(\d{2} \w{3} \d{4})",
        "advice_ref": r"Advice reference no[:\s]*([A-Za-z0-9\-]+)",
        "receipt_name": r"Recipient's name and contact information:[\s]*([^\n]+)",
        "receipt_email": r"Recipient's name and contact information:[\s]*[^\n]+[\s]*([\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,})",
        "transaction_type": r"Transaction\s+type\s*[:\-]?\s*([A-Za-z ]*?payment)",
        "sub_payment_type": r"Sub\s+payment\s+type\s*[:\-]?\s*([A-Za-z ]+)",
        "beneficiary_name": r"Beneficiary's\s+name[:\-]?\s*(.*?)(?=\s+Debit amount|$)",
        "beneficiary_bank": r"Beneficiary's\s+bank[:\-]?\s*\n?([A-Za-z0-9 ,.\-&]*BANK)",
        "account_number": r"Beneficiary's account[:\s]*([A-Za-z0-9\*]+)",
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
        if key in selected_fields:
            match = re.search(pattern, pdf_text, re.IGNORECASE)
            if match:
                extracted_data[key] = match.group(1).strip()
            else:
                extracted_data[key] = "Not Found"

    return extracted_data

# Function to extract and clean invoice table
def process_invoice_table(table):
    invoice_details = []
    for page_table in table:
        if page_table:
            header = page_table[0]
            for row in page_table[1:]:
                row_data = {}
                for i, column in enumerate(row):
                    if i < len(header):
                        row_data[header[i]] = column if column else "Not Provided"
                invoice_details.append(row_data)
    return invoice_details

# Visual debug function
def visual_debug_each_cell(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]

        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
        }

        tables = page.find_tables(table_settings=table_settings)
        if not tables:
            st.warning("❌ No tables found for visualization.")
            return

        table = tables[0]
        table_data = table.extract()
        cell_bboxes = table.cells

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_title("📏 Visual Debug: Per-Cell Text + ht + wd", fontsize=14)
        ax.set_aspect('equal')

        all_x, all_y = [], []
        cell_idx = 0

        for row_idx, row in enumerate(table_data, 1):
            for col_idx, text in enumerate(row, 1):
                if cell_idx >= len(cell_bboxes):
                    break

                x0, top, x1, bottom = cell_bboxes[cell_idx]
                width, height = x1 - x0, bottom - top

                all_x.extend([x0, x1])
                all_y.extend([top, bottom])

                rect = patches.Rectangle((x0, top), width, height,
                                         linewidth=1.2, edgecolor='black', facecolor='none')
                ax.add_patch(rect)

                clean_text = text.strip() if text else ""
                ax.text(x0 + width / 2, top + height / 2,
                        clean_text, fontsize=8.5, color='black',
                        ha='center', va='center', fontweight='bold')

                ax.text(x0 + width / 2, bottom + 8,
                        f"wd: {width:.1f} pt", color='red',
                        fontsize=8, ha='center', fontweight='bold')

                ax.text(x0 + width - 5, top + 4,
                        f"ht: {height:.1f} pt", color='red',
                        fontsize=8, ha='right', fontweight='bold')

                ax.text(x0 + 2, top + 2, f"R{row_idx}C{col_idx}",
                        fontsize=6, color='blue', ha='left', va='top')

                cell_idx += 1

        if all_x and all_y:
            ax.set_xlim(min(all_x) - 30, max(all_x) + 30)
            ax.set_ylim(max(all_y) + 30, min(all_y) - 30)
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("⚠️ No valid cell coordinates to plot.")

# Master extraction function
def extract_payment_advice_data(pdf_path, selected_fields):
    pdf_text, table = extract_text_and_tables(pdf_path)
    extracted_data = extract_data_using_regex(pdf_text, selected_fields)
    invoice_details = process_invoice_table(table)
    return {**extracted_data, "invoice_details": invoice_details}

# Streamlit UI
st.set_page_config(page_title="Payment Advice Extractor", layout="centered")
st.title("📄 Payment Advice PDF Extractor")

# File upload widget
uploaded_file = st.file_uploader("Upload a Payment Advice PDF", type="pdf")

# Instructions and Info
# st.info(""" ... """)  # Optional info message removed for brevity

# Add a help section with an expandable FAQ/Instructions pop-up
with st.expander("📘 How to use this tool?"):
    st.write(""" 
    - First, upload the Payment Advice PDF using the **Upload** button.
    - Then, select the fields you'd like to extract. If you don't select any, you'll be prompted to select at least one field.
    - After processing, the extracted data will be displayed in JSON format, and you can download it.
    - Optionally, you can enable **Table Cell Debug** to visualize how data is being extracted from tables in the PDF.
    """)

# Check if the user uploads the file before selecting fields
if uploaded_file is None:
    st.warning("⚠️ Please upload a PDF file before selecting fields to extract.", icon="⚠️")
else:
    # Parameter selection by the user
    selected_fields = st.multiselect(
        "Select fields to extract:",
        options=["advice_date", "advice_ref", "receipt_name", "receipt_email", "transaction_type", "sub_payment_type",
                 "beneficiary_name", "beneficiary_bank", "account_number", "customer_reference", "debit_amount",
                 "remittance_amount", "handling_fee", "value_date", "remitter_name", "remitting_bank", "instruction_reference",
                 "other_reference", "remitter_to_beneficiary_info"],
        default=["advice_date", "advice_ref", "receipt_name", "beneficiary_bank"],
        help="Select the fields you want to extract from the payment advice."
    )

    # Check if no fields are selected
    if not selected_fields:
        st.warning("⚠️ Please select at least one field to extract.", icon="⚠️")

    # Show loading spinner if PDF is uploaded and fields selected
    if uploaded_file is not None and selected_fields:
        with st.spinner("⏳ Extracting data, please wait..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            try:
                result = extract_payment_advice_data(temp_file_path, selected_fields)

                st.subheader("🧾 Extracted JSON Output")
                st.json(result)

                json_data = json.dumps(result, indent=4)
                st.download_button("Download JSON", data=json_data, file_name="payment_advice_output.json", mime="application/json")

                # Optional visual debug
                if st.checkbox("🔍 Show Table Cell Debug Visualization"):
                    st.subheader("📊 Table Structure Debug View")
                    visual_debug_each_cell(temp_file_path)

            except Exception as e:
                st.error(f"❌ Error processing the PDF: {e}", icon="🚨")

            finally:
                os.remove(temp_file_path)
