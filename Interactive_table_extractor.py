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

# -------------------------------
# Page Configuration & Styles
# -------------------------------
st.set_page_config(page_title="📄 PDF Table Extractor", layout="wide")
st.markdown("""
<style>
.main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
.section-header { font-size: 1.5rem; font-weight: bold; color: #333; margin-top: 2rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Helper Functions
# -------------------------------
def convert_pdf_to_images(pdf_file):
    images = []
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(image)
    return images

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

def dataframe_to_json(df, format_type="row_wise"):
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
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box in boxes_with_labels:
        x1, y1, x2, y2 = box['coords']
        label = box['label']
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 25), label, fill="blue", font=font)
    return annotated

# -------------------------------
# Main App Logic
# -------------------------------
def main():
    st.markdown('<div class="main-header">📄 PDF Table Extractor (Interactive)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">📁 Upload PDF & Draw Box</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file:
            images = convert_pdf_to_images(uploaded_file)
            page_num = st.selectbox("Select page", range(len(images)), format_func=lambda x: f"Page {x + 1}")
            zoom = st.slider("🔍 Zoom Level", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
            label_text = st.text_input("✏️ Label for Bounding Box (e.g., Table1)", "table")

            original_image = images[page_num]
            width, height = original_image.size
            zoomed_size = (int(width * zoom), int(height * zoom))
            display_image = original_image.resize(zoomed_size)
            display_np = np.array(display_image)

            st.markdown("🎯 **Draw a box around the table**")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="red",
                background_image=display_image,
                update_streamlit=True,
                width=zoomed_size[0],
                height=zoomed_size[1],
                drawing_mode="rect",
                key=f"canvas_{page_num}"
            )

            if "boxes" not in st.session_state:
                st.session_state.boxes = []

            if st.button("➕ Add Box & Extract Table"):
                if canvas_result.json_data and canvas_result.json_data["objects"]:
                    rect = canvas_result.json_data["objects"][-1]
                    x1 = int(rect["left"] / zoom)
                    y1 = int(rect["top"] / zoom)
                    x2 = int(x1 + rect["width"] / zoom)
                    y2 = int(y1 + rect["height"] / zoom)
                    coords = (x1, y1, x2, y2)

                    with st.spinner("⏳ Extracting table..."):
                        extracted_df = extract_table_with_pdfplumber(uploaded_file, page_num, coords)
                    
                    if extracted_df is not None:
                        st.session_state.extracted_df = extracted_df
                        st.session_state.boxes.append({"label": label_text, "coords": coords})
                        st.success("✅ Table extracted successfully!")
                        st.dataframe(extracted_df, use_container_width=True)
                    else:
                        st.error("❌ No table found. Try adjusting your selection.")
                else:
                    st.warning("⚠️ Please draw a bounding box first.")

    with col2:
        st.markdown('<div class="section-header">📋 Table JSON Output</div>', unsafe_allow_html=True)
        json_format = st.selectbox("Select JSON format", ["row_wise", "column_wise", "table"])

        if 'extracted_df' in st.session_state:
            df = st.session_state.extracted_df
            json_output = dataframe_to_json(df, json_format)
            json_str = json.dumps(json_output, indent=2, ensure_ascii=False)

            st.code(json_str, language="json")
            st.download_button("📥 Download JSON", json_str, file_name=f"{label_text}_{json_format}.json", mime="application/json")
            st.write(f"- **Rows:** {len(df)}")
            st.write(f"- **Columns:** {len(df.columns)}")

        st.markdown('<div class="section-header">🖼️ Annotated Image</div>', unsafe_allow_html=True)
        if "boxes" in st.session_state and st.session_state.boxes:
            annotated_img = draw_annotation(original_image, st.session_state.boxes)
            img_bytes = BytesIO()
            annotated_img.save(img_bytes, format="PNG")
            st.image(annotated_img, caption="Annotated Image", use_column_width=True)

            st.download_button(
                label="📥 Download Annotated Image",
                data=img_bytes.getvalue(),
                file_name="annotated_image.png",
                mime="image/png"
            )

    with st.sidebar:
        st.markdown("## 🛠️ Instructions")
        st.markdown("""
        1. Upload a PDF file
        2. Choose a page
        3. Zoom and draw a box over the table
        4. Label it and extract the table
        5. Download table in JSON format
        6. Download annotated image
        """)
        st.markdown("### 🔍 Tip")
        st.markdown("Zoom helps with small or dense tables.")

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    main()