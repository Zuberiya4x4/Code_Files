# code works for both pdf and images
import streamlit as st
import pandas as pd
import pytesseract
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json
import os
import re
from streamlit_drawable_canvas import st_canvas

# Tesseract Path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="OCR Tagger", layout="wide")

# ---------------- Helpers ----------------
def load_font(size=18):
    try:
        return ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", size)
    except:
        return ImageFont.load_default()

def draw_text(draw, position, text, font, text_color="white", bg_color="black"):
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=bg_color)
    draw.text(position, text, fill=text_color, font=font)

def update_annotated_image(image, boxes, stroke_color, stroke_width, font, highlight_idx=-1):
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    for i, box in enumerate(boxes):
        coords = box["coordinates"]
        x, y, w, h = coords["x"], coords["y"], coords["width"], coords["height"]
        color = "#00FF00" if i == highlight_idx else stroke_color
        draw.rectangle([x, y, x + w, y + h], outline=color, width=stroke_width)
        draw_text(draw, (x, y - 40), f"type: {box['type']}", font, bg_color="#6a1b9a")
        draw_text(draw, (x, y - 25), f"title: {box['title']}", font, bg_color="#d32f2f")
        draw_text(draw, (x, y - 10), f"value: {box['value']}", font, bg_color="#1976d2")
    return annotated_img

def clean_and_extract_text(raw_text):
    cleaned = raw_text.replace("TINR", "INR").replace("INRSO", "INR 50").replace("INR5O", "INR 50")
    cleaned = re.sub(r'\bS0\b', "50", cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned.strip())

    currency_match = re.search(r'INR\s*[\d\s,\.]+', cleaned)
    if currency_match:
        amount_str = currency_match.group()
        numeric_part = re.sub(r'[^\d.]', '', amount_str)
        try:
            formatted_amount = f"INR {float(numeric_part):,.2f}"
        except ValueError:
            formatted_amount = amount_str
        title = "Amount"
        value = formatted_amount
    else:
        match = re.match(r"(.*?):\s*(.*)", cleaned)
        if match:
            title, value = match.groups()
        else:
            title = "Field"
            value = cleaned

    return title.strip(), value.strip()

# ---------------- State ----------------
for key in ["boxes", "json_table", "selected_page_index", "highlight_box_idx", "table_view_mode"]:
    if key not in st.session_state:
        if key == "boxes":
            st.session_state[key] = []
        elif key == "json_table":
            st.session_state[key] = {}
        elif key == "table_view_mode":
            st.session_state[key] = "row-wise"
        else:
            st.session_state[key] = 0

# ---------------- Sidebar ----------------
st.sidebar.header("Annotation Controls")
stroke_width = st.sidebar.slider("Stroke Width", 1, 10, 2)
stroke_color = st.sidebar.color_picker("Box Color", "#FF0000")
field_type = st.sidebar.selectbox("Field Type", ["text", "date", "email", "currency", "number", "label"])
zoom_level = st.sidebar.slider("Zoom Level (%)", 50, 300, 100)

# ---------------- File Upload ----------------
st.title("🖼️ OCR Annotator (Images + PDFs)")
uploaded_file = st.sidebar.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    is_pdf = file_ext == "pdf"
    images = []

    if is_pdf:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        selected_page = st.selectbox("Select Page", [f"Page {i+1}" for i in range(len(images))])
        st.session_state.selected_page_index = int(selected_page.split(" ")[1]) - 1
    else:
        image = Image.open(uploaded_file).convert("RGB")
        images.append(image)
        st.session_state.selected_page_index = 0

    image = images[st.session_state.selected_page_index]
    zoom = zoom_level / 100.0
    zoomed_size = (int(image.width * zoom), int(image.height * zoom))
    zoomed_image = image.resize(zoomed_size)
    font = load_font(16)

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("Draw Bounding Boxes")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=zoomed_image,
            update_streamlit=True,
            height=zoomed_size[1],
            width=zoomed_size[0],
            drawing_mode="rect",
            key="canvas",
            display_toolbar=True
        )

        if canvas_result.json_data and canvas_result.json_data["objects"]:
            new_boxes = []
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    x = int(obj["left"] / zoom)
                    y = int(obj["top"] / zoom)
                    w = int(obj["width"] / zoom)
                    h = int(obj["height"] / zoom)
                    cropped = image.crop((x, y, x + w, y + h)).convert("L")
                    raw_text = pytesseract.image_to_string(cropped, config="--psm 6")
                    title, value = clean_and_extract_text(raw_text)
                    new_boxes.append({
                        "title": title,
                        "value": value,
                        "type": field_type,
                        "coordinates": {
                            "x": x, "y": y, "width": w, "height": h,
                            "page": st.session_state.selected_page_index + 1
                        }
                    })
            st.session_state.boxes = new_boxes

        annotated_img = update_annotated_image(image, st.session_state.boxes, stroke_color, stroke_width, font, highlight_idx=st.session_state.highlight_box_idx)
        st.image(annotated_img, caption="Annotated Image", use_column_width=True)

        buffer = BytesIO()
        annotated_img.save(buffer, format="PNG")
        st.download_button("📥 Download Annotated Image", buffer.getvalue(), "annotated_output.png", "image/png")

    with right_col:
        st.subheader("📝 Fields Extracted")

        if st.button("🧹 Clear All", key="clear_all"):
            st.session_state.boxes = []
            st.experimental_rerun()

        if st.session_state.boxes:
            for i, box in enumerate(st.session_state.boxes):
                with st.expander(f"Field {i+1} ({box['type']})"):
                    box["type"] = st.selectbox("Type", ["text", "date", "email", "currency", "number", "label"], index=["text", "date", "email", "currency", "number", "label"].index(box["type"]), key=f"type_{i}")
                    box["title"] = st.text_input("Title", value=box["title"], key=f"title_{i}")
                    box["value"] = st.text_area("Value", value=box["value"], key=f"value_{i}")
                    st.json(box["coordinates"])
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Update", key=f"update_{i}"):
                            st.success(f"Field {i+1} updated.")
                    with col2:
                        if st.button("Delete", key=f"delete_{i}"):
                            st.session_state.boxes.pop(i)
                            st.experimental_rerun()

        st.subheader("📊 Extract Table")
        st.session_state.table_view_mode = st.selectbox("Table JSON Format", ["row-wise", "column-wise", "table"])

        if st.button("🧾 Extract Table"):
            if is_pdf:
                uploaded_file.seek(0)
                with pdfplumber.open(uploaded_file) as pdf:
                    page_index = st.session_state.selected_page_index
                    page = pdf.pages[page_index]
                    tables = page.extract_tables()
                    if tables:
                        headers = tables[0][0]
                        rows = tables[0][1:]
                        if st.session_state.table_view_mode == "row-wise":
                            table_data = [
                                {headers[i].strip(): row[i].strip() if row[i] else "" for i in range(len(headers))}
                                for row in rows
                            ]
                            st.session_state.json_table = {
                                "page": page_index + 1,
                                "invoice_details": table_data
                            }
                        elif st.session_state.table_view_mode == "column-wise":
                            col_dict = {header.strip(): [] for header in headers}
                            for row in rows:
                                for i, header in enumerate(headers):
                                    col_dict[header.strip()].append(row[i].strip() if row[i] else "")
                            st.session_state.json_table = {
                                "page": page_index + 1,
                                "invoice_details": col_dict
                            }
                        elif st.session_state.table_view_mode == "table":
                            st.session_state.json_table = {
                                "page": page_index + 1,
                                "invoice_details": tables[0]
                            }
                        st.success("Table extracted successfully.")
                    else:
                        st.warning("No table found.")
            else:
                st.warning("Only PDFs are supported for table extraction.")

        if st.session_state.json_table:
            st.write("📋 Extracted Table:")
            st.json(st.session_state.json_table)

        st.subheader("📦 Export JSON")
        full_data = {
            "fields": st.session_state.boxes,
            "table": st.session_state.json_table
        }

        st.json(full_data)
        json_bytes = BytesIO(json.dumps(full_data, indent=2).encode("utf-8"))
        st.download_button("📄 Download JSON", data=json_bytes, file_name="extracted_data.json", mime="application/json")
else:
    st.info("Upload an image or PDF to begin tagging.")
