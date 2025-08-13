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

# Set page config
st.set_page_config(
    page_title="PDF Table Extractor & Annotator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional custom style
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

def convert_pdf_to_images(pdf_file, zoom_factor=2.0):
    images = []
    pdf_file.seek(0)
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            mat = fitz.Matrix(zoom_factor, zoom_factor)
            pix = page.get_pixmap(matrix=mat)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(image)
    return images

def extract_table_with_pdfplumber(pdf_file, page_num, coordinates, zoom_factor=2.0):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    x1, y1, x2, y2 = coordinates
    with pdfplumber.open(tmp_path) as pdf:
        page = pdf.pages[page_num]
        crop_box = (
            x1 / zoom_factor,
            y1 / zoom_factor,
            x2 / zoom_factor,
            y2 / zoom_factor
        )
        cropped_page = page.crop(crop_box)
        table = cropped_page.extract_table()

    os.remove(tmp_path)

    if table:
        return pd.DataFrame(table[1:], columns=table[0])
    return None

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

def draw_annotations_on_image(image, objects, labels=None):
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Use a larger font for better visibility
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i, obj in enumerate(objects):
        if obj["type"] == "rect":
            x1 = obj["left"]
            y1 = obj["top"]
            x2 = x1 + obj["width"]
            y2 = y1 + obj["height"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            label = labels[i] if labels and i < len(labels) else "Table"
            draw.text((x1, max(0, y1 - 20)), f"**{label}**", fill="red", font=font)
        elif obj["type"] == "path":
            points = [(pt["x"], pt["y"]) for pt in obj["path"] if "x" in pt and "y" in pt]
            if len(points) > 1:
                draw.line(points, fill="blue", width=3)
    return annotated_image

def main():
    st.markdown('<div class="main-header">📄 PDF Table Extractor & Annotator</div>', unsafe_allow_html=True)

    # File uploader outside layout for guaranteed visibility
    uploaded_file = st.sidebar.file_uploader("📁 Upload a PDF", type="pdf")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">🔍 Annotate PDF</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            zoom = st.slider("Zoom Level", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
            images = convert_pdf_to_images(uploaded_file, zoom_factor=zoom)
            page_num = st.selectbox("Select page", range(len(images)), format_func=lambda x: f"Page {x + 1}")
            current_image = images[page_num]

            st.markdown("**Draw a rectangle or freehand annotate:**")
            drawing_mode = st.selectbox("Drawing mode", ["rect", "freedraw"])
            label_input = st.text_input("Label for annotation", value="Table")

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="rgba(255, 0, 0, 0.8)",
                background_image=current_image,
                update_streamlit=True,
                width=current_image.width,
                height=current_image.height,
                drawing_mode=drawing_mode,
                key=f"canvas_{page_num}",
            )

            if canvas_result.json_data and canvas_result.json_data["objects"]:
                labels = [label_input] * len(canvas_result.json_data["objects"])

                annotated_img = draw_annotations_on_image(current_image, canvas_result.json_data["objects"], labels=labels)
                st.image(annotated_img, caption="📌 Annotated Image", use_column_width=True)

                # Download button
                img_bytes = BytesIO()
                annotated_img.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                st.download_button(
                    label="📥 Download Annotated Image",
                    data=img_bytes,
                    file_name="annotated_image.png",
                    mime="image/png"
                )

            if st.button("📊 Extract Table", type="primary"):
                if canvas_result.json_data and canvas_result.json_data["objects"]:
                    rects = [obj for obj in canvas_result.json_data["objects"] if obj["type"] == "rect"]
                    if rects:
                        rect = rects[-1]
                        x1 = rect["left"]
                        y1 = rect["top"]
                        x2 = x1 + rect["width"]
                        y2 = y1 + rect["height"]
                        coords = (x1, y1, x2, y2)
                        with st.spinner("Extracting table..."):
                            extracted_df = extract_table_with_pdfplumber(uploaded_file, page_num, coords, zoom_factor=zoom)
                        if extracted_df is not None:
                            st.session_state.extracted_df = extracted_df
                            st.success("✅ Table extracted successfully!")
                            st.dataframe(extracted_df, use_container_width=True)
                        else:
                            st.error("❌ No table found. Try adjusting your selection.")
                    else:
                        st.warning("⚠️ Please draw a rectangle to extract a table.")
                else:
                    st.warning("⚠️ Please annotate first with a rectangle.")

    with col2:
        st.markdown('<div class="section-header">📋 JSON Output</div>', unsafe_allow_html=True)
        json_format = st.selectbox("Select JSON format", ["row_wise", "column_wise", "table"])

        if 'extracted_df' in st.session_state:
            df = st.session_state.extracted_df
            json_output = dataframe_to_json(df, json_format)
            json_str = json.dumps(json_output, indent=2, ensure_ascii=False)
            st.code(json_str, language="json")
            st.download_button("📥 Download JSON", json_str, file_name=f"extracted_table_{json_format}.json", mime="application/json")
            st.write(f"- **Rows:** {len(df)}")
            st.write(f"- **Columns:** {len(df.columns)}")
            st.write(f"- **Format:** `{json_format}`")
        else:
            st.info("👆 Extract a table to view JSON output.")

    with st.sidebar:
        st.markdown("## 📖 Instructions")
        st.markdown("""
        1. **Upload PDF**: Upload your file from the sidebar.
        2. **Zoom In/Out**: Use the slider to improve visibility.
        3. **Draw Rectangle**: Annotate the table area and label it.
        4. **Extract Table**: Click extract and view results.
        5. **Download JSON**: Choose format and download the table.
        """)
        st.markdown("## 🔧 Tools Used")
        st.markdown("- **pdfplumber** for table extraction")
        st.markdown("- **streamlit-drawable-canvas** for annotation")

if __name__ == "__main__":
    main()
