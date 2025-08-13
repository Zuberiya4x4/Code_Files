import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import fitz  # PyMuPDF
import pdfplumber
from io import BytesIO
import os
import json
from streamlit_drawable_canvas import st_canvas

# -------------------- Config --------------------
st.set_page_config(page_title="🧾 Field Tagging System", layout="wide")

# ------------------- Font -----------------------
def load_clean_font(size=18):
    
    try:
        font_path = os.path.join(os.path.expanduser("~"), "Desktop", "DejaVuSans.ttf")
        return ImageFont.truetype(font_path, size)
    except:
        return ImageFont.load_default()

def safe_text(text):
    try:
        return text.encode("latin-1", errors="ignore").decode("latin-1")
    except:
        return text.encode("ascii", errors="ignore").decode("ascii")

def draw_text_with_background(draw, position, text, text_color, bg_color, font):
    text = safe_text(text)
    text_bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(text_bbox, fill=bg_color)
    draw.text(position, text, fill=text_color, font=font)

def extract_title_value(text):
    parts = text.split(":")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    elif len(parts) == 1 and len(parts[0]) > 0:
        return "Field", parts[0].strip()
    return "Field", ""

def parse_table_to_json(df, mode="row"):
    df = df.dropna(how="all")
    if mode == "row":
        return {"invoice_details": df.to_dict(orient="records")}
    elif mode == "column":
        return {"invoice_details": df.to_dict(orient="dict")}
    else:
        return df

# ------------------- State Init ------------------------
if "tagged_fields" not in st.session_state:
    st.session_state.tagged_fields = []
if "annotated_img" not in st.session_state:
    st.session_state.annotated_img = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "json_table" not in st.session_state:
    st.session_state.json_table = {}
if "selected_page_index" not in st.session_state:
    st.session_state.selected_page_index = 0

# ------------------ Reset --------------------------
def clear_all():
    st.session_state.tagged_fields = []
    st.session_state.annotated_img = None
    st.session_state.pdf_bytes = None
    st.session_state.json_table = {}
    st.session_state.selected_page_index = 0
    st.experimental_rerun()

# ------------------- UI ------------------------------
st.title("🧾Field Tagging System With Table Extraction")
uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    is_pdf = file_ext == "pdf"
    images = []

    file_bytes = uploaded_file.read()
    st.session_state.pdf_bytes = file_bytes
    file_stream = BytesIO(file_bytes)

    if is_pdf:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        page_options = [f"Page {i+1}" for i in range(len(images))]
        selected_page = st.selectbox("Select PDF Page", page_options)
        st.session_state.selected_page_index = page_options.index(selected_page)
    else:
        image = Image.open(file_stream).convert("RGB")
        images.append(image)

    image = images[st.session_state.selected_page_index]

    # --------------- Split UI ------------------------
    col_canvas, col_json = st.columns([2.5, 1.5])
    label_font = load_clean_font(size=18)

    with col_canvas:
        st.subheader("📌 Draw Bounding Boxes")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=image,
            update_streamlit=True,
            height=image.height,
            width=image.width,
            drawing_mode="rect",
            key="canvas"
        )

        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)

        if canvas_result.json_data:
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    x, y = int(obj["left"]), int(obj["top"])
                    w, h = int(obj["width"]), int(obj["height"])
                    cropped = image.crop((x, y, x + w, y + h))
                    raw_text = pytesseract.image_to_string(cropped, config="--psm 6").strip()
                    title, value = extract_title_value(raw_text)

                    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
                    font_bbox = label_font.getbbox("Ag")
                    font_height = font_bbox[3] - font_bbox[1]
                    padding = 4
                    title_y = y - 2 * (font_height + padding)
                    value_y = y - (font_height + padding)

                    draw_text_with_background(draw, (x, value_y), f"value: {value}", "white", "#1976d2", label_font)
                    draw_text_with_background(draw, (x, title_y), f"title: {title}", "white", "#d32f2f", label_font)

                    new_field = {
                        "field_name": title,
                        "value": value,
                        "coordinates": {
                            "x": x, "y": y, "width": w, "height": h,
                            "page": st.session_state.selected_page_index + 1
                        }
                    }
                    if new_field not in st.session_state["tagged_fields"]:
                        st.session_state["tagged_fields"].append(new_field)

        st.session_state.annotated_img = annotated_img
        st.image(annotated_img, caption="🖍 Annotated Image", use_column_width=True)

        # Download button
        if st.session_state.annotated_img:
            buffered = BytesIO()
            st.session_state.annotated_img.save(buffered, format="PNG")
            st.download_button(
                label="📥 Download Annotated Image",
                data=buffered.getvalue(),
                file_name="annotated_image.png",
                mime="image/png"
            )

    # ------------- JSON Output & Table on Right ----------------
    with col_json:
        st.subheader("📋 Extracted Fields")
        st.json(st.session_state["tagged_fields"])

        if st.button("🧹 Clear Annotations"):
            st.session_state["tagged_fields"] = []

        if is_pdf:
            st.subheader("📊 Table View Format")
            table_mode = st.selectbox("Select Table Format", ["Row-wise JSON", "Column-wise JSON", "Tabular DataFrame"])

            if st.button("📋 Extract Table from PDF"):
                try:
                    with pdfplumber.open(BytesIO(st.session_state.pdf_bytes)) as pdf:
                        page = pdf.pages[st.session_state.selected_page_index]
                        tables = page.extract_tables()
                        if tables:
                            df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
                            if table_mode == "Row-wise JSON":
                                st.session_state.json_table = parse_table_to_json(df, mode="row")
                            elif table_mode == "Column-wise JSON":
                                st.session_state.json_table = parse_table_to_json(df, mode="column")
                            else:
                                st.session_state.json_table = df
                        else:
                            st.warning("No tables found on this page.")
                except Exception as e:
                    st.error(f"Error reading table: {e}")

            if st.session_state.json_table:
                st.subheader("🧾 Extracted Table")
                if isinstance(st.session_state.json_table, pd.DataFrame):
                    st.dataframe(st.session_state.json_table)
                else:
                    st.json(st.session_state.json_table)
