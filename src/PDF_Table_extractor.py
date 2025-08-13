import streamlit as st
import pandas as pd
import json
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
from streamlit_drawable_canvas import st_canvas
import pdfplumber
import tempfile
import os
import sqlite3

# ---------- Streamlit Page Configuration ----------
st.set_page_config(
    page_title="PDF Table Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Convert PDF Pages to Images ----------
def convert_pdf_to_images(pdf_file):
    images = []
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(image)
    return images

# ---------- Extract Table from Cropped Area ----------
def extract_table_with_pdfplumber(pdf_file, page_num, coordinates):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    x1, y1, x2, y2 = coordinates
    with pdfplumber.open(tmp_path) as pdf:
        page = pdf.pages[page_num]
        crop_box = (x1 / 2, y1 / 2, x2 / 2, y2 / 2)
        cropped_page = page.crop(crop_box)
        table = cropped_page.extract_table()

    os.remove(tmp_path)

    if table:
        return pd.DataFrame(table[1:], columns=table[0])
    return None

# ---------- Convert DataFrame to JSON ----------
def dataframe_to_json(df, format_type="row_wise"):
    if df is None or df.empty:
        return {}
    df = df.dropna(how='all')
    df = df.replace('', np.nan)
    if format_type == "row_wise":
        return {"rows": df.dropna(how='all').to_dict(orient='records')}
    elif format_type == "column_wise":
        return df.dropna(how='all').to_dict(orient='list')
    elif format_type == "table":
        return df.to_dict(orient='records')
    return {}

# ---------- Save JSON to SQLite Database ----------
def save_json_to_db(json_data, format_type):
    conn = sqlite3.connect("extracted_tables.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS extracted_json (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            format_type TEXT,
            json_data TEXT
        )
    """)

    cursor.execute("INSERT INTO extracted_json (format_type, json_data) VALUES (?, ?)",
                   (format_type, json.dumps(json_data)))

    conn.commit()
    conn.close()

# ---------- Streamlit App ----------
def main():
    st.markdown('<div class="main-header">📄 PDF Table Extractor (Using pdfplumber)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">📁 Upload & Extract</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            images = convert_pdf_to_images(uploaded_file)
            page_num = st.selectbox("Select page", range(len(images)), format_func=lambda x: f"Page {x + 1}")
            current_image = images[page_num]

            st.markdown("**Draw a box around the table you want to extract:**")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="rgba(255, 0, 0, 0.8)",
                background_image=current_image,
                update_streamlit=True,
                width=current_image.width,
                height=current_image.height,
                drawing_mode="rect",
                key=f"canvas_{page_num}",
            )

            if st.button("🔍 Extract Table", type="primary"):
                if canvas_result.json_data and canvas_result.json_data["objects"]:
                    rect = canvas_result.json_data["objects"][-1]
                    x1 = rect["left"]
                    y1 = rect["top"]
                    x2 = x1 + rect["width"]
                    y2 = y1 + rect["height"]
                    coords = (x1, y1, x2, y2)
                    with st.spinner("Extracting table using pdfplumber..."):
                        extracted_df = extract_table_with_pdfplumber(uploaded_file, page_num, coords)
                    if extracted_df is not None:
                        st.session_state.extracted_df = extracted_df
                        st.success("✅ Table extracted successfully!")
                        st.dataframe(extracted_df, use_container_width=True)
                    else:
                        st.error("❌ No table found. Try adjusting your selection.")
                else:
                    st.warning("⚠️ Please draw a rectangle around the table first.")

    with col2:
        st.markdown('<div class="section-header">📋 JSON Output</div>', unsafe_allow_html=True)
        json_format = st.selectbox("Select JSON format", ["row_wise", "column_wise", "table"])

        if 'extracted_df' in st.session_state:
            df = st.session_state.extracted_df
            json_output = dataframe_to_json(df, json_format)
            json_str = json.dumps(json_output, indent=2, ensure_ascii=False)

            # Save JSON to database
            save_json_to_db(json_output, json_format)

            st.code(json_str, language="json")
            st.download_button("📥 Download JSON", json_str, file_name=f"extracted_table_{json_format}.json", mime="application/json")
            st.write(f"- **Rows:** {len(df)}")
            st.write(f"- **Columns:** {len(df.columns)}")
            st.write(f"- **Format:** {json_format}")
        else:
            st.info("👆 Upload a PDF and extract a table to see JSON output here.")

    with st.sidebar:
        st.markdown("## 📖 Instructions")
        st.markdown("""
        1. **Upload PDF**: Choose a PDF file containing tables  
        2. **Select Page**: Pick the page with your target table  
        3. **Draw Box**: Draw a rectangle around the table  
        4. **Extract**: Click the extract button  
        5. **View JSON**: Check the right panel for JSON output  
        6. **Download**: Save the extracted data  
        7. **Stored**: JSON is saved to the database automatically  
        """)
        st.markdown("## 🔧 Tools")
        st.markdown("- Uses **pdfplumber** for accurate table extraction")

if __name__ == "__main__":
    main()
