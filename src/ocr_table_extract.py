import streamlit as st
import pandas as pd
import pytesseract
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json
import os
from streamlit_drawable_canvas import st_canvas

# ---------------- Config ----------------
st.set_page_config(page_title="🧾 OCR Tagging & Table Extractor", layout="wide")

# ---------------- Font ------------------
def load_font(size=18):
    try:
        font_path = "C:\\Windows\\Fonts\\arial.ttf"  # Adjust for Mac/Linux
        return ImageFont.truetype(font_path, size)
    except Exception as e:
        print("Font load error:", e)
        return ImageFont.load_default()

def draw_text(draw, position, text, font, text_color="white", bg_color="black"):
    try:
        bbox = draw.textbbox(position, text, font=font)
        draw.rectangle(bbox, fill=bg_color)
        draw.text(position, text, fill=text_color, font=font)
    except Exception as e:
        print(f"Draw text error: {e}")

# ---------------- State Init ----------------
if "boxes" not in st.session_state:
    st.session_state.boxes = []
if "json_table" not in st.session_state:
    st.session_state.json_table = {}
if "selected_page_index" not in st.session_state:
    st.session_state.selected_page_index = 0
if "update_trigger" not in st.session_state:
    st.session_state.update_trigger = False

# ---------------- Utils ----------------
def parse_table_to_json(df, mode="row"):
    df = df.dropna(how="all")
    if mode == "row":
        return {"invoice_details": df.to_dict(orient="records")}
    elif mode == "column":
        return {"invoice_details": df.to_dict(orient="dict")}
    else:
        return df.to_dict()

def update_annotated_image(image, boxes, stroke_color, stroke_width, font):
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    for box in boxes:
        coords = box["coordinates"]
        x, y, w, h = coords["x"], coords["y"], coords["width"], coords["height"]
        draw.rectangle([x, y, x + w, y + h], outline=stroke_color, width=stroke_width)
        draw_text(draw, (x, y - 40), f"type: {box['type']}", font, bg_color="#6a1b9a")
        draw_text(draw, (x, y - 25), f"title: {box['title']}", font, bg_color="#d32f2f")
        draw_text(draw, (x, y - 10), f"value: {box['value']}", font, bg_color="#1976d2")
    return annotated_img

# ---------------- UI ----------------
st.title("🧾 OCR Tagging + Table Extractor")
uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    is_pdf = file_ext == "pdf"
    images = []
    file_bytes = uploaded_file.read()
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
    font = load_font(16)

    left, right = st.columns([1.4, 3.6])

    with left:
        st.header("🛠️ Controls")
        stroke_width = st.slider("Stroke Width", 1, 10, 2)
        stroke_color = st.color_picker("Box Color", "#FF0000")
        field_type = st.selectbox("Select Field Type", ["text", "date", "email", "currency", "number", "label"])
        json_orient = st.selectbox("JSON Orientation", ["row", "column", "table"])
        page_numbers = st.text_input("Pages (comma-separated)", value=str(st.session_state.selected_page_index + 1))
        zoom_level = st.slider("Zoom Level (%)", 50, 300, 100)

        if st.button("📋 Extract Table from PDF"):
            try:
                with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                    selected_pages = [int(p.strip()) - 1 for p in page_numbers.split(",") if p.strip().isdigit()]
                    all_tables = []
                    for p in selected_pages:
                        page = pdf.pages[p]
                        tables = page.extract_tables()
                        if tables:
                            df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
                            all_tables.append(df)
                    if all_tables:
                        df = pd.concat(all_tables)
                        st.session_state.json_table = parse_table_to_json(df, mode=json_orient)
                    else:
                        st.warning("No tables found.")
            except Exception as e:
                st.error(f"Table extract error: {e}")

        # 🔄 Clear All Button
        if st.button("🧹 Clear All"):
            st.session_state.boxes = []
            st.session_state.json_table = {}
            st.experimental_rerun()

    with right:
        canvas_col, text_col = st.columns([1.5, 2.5])

        with canvas_col:
            st.subheader("🖍 Annotate Canvas")
            zoom_factor = zoom_level / 100.0
            zoomed_size = (int(image.width * zoom_factor), int(image.height * zoom_factor))
            zoomed_image = image.resize(zoomed_size)

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=zoomed_image,
                update_streamlit=True,
                height=zoomed_size[1],
                width=zoomed_size[0],
                drawing_mode="rect",
                key="canvas"
            )

            if canvas_result.json_data and canvas_result.json_data["objects"]:
                new_boxes = []
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "rect":
                        x = int(obj["left"] / zoom_factor)
                        y = int(obj["top"] / zoom_factor)
                        w = int(obj["width"] / zoom_factor)
                        h = int(obj["height"] / zoom_factor)
                        cropped = image.crop((x, y, x + w, y + h))
                        text = pytesseract.image_to_string(cropped, config="--psm 6").strip()
                        box = {
                            "title": text.split(":")[0].strip() if ":" in text else "Field",
                            "value": text.split(":")[1].strip() if ":" in text else text,
                            "type": field_type,
                            "coordinates": {
                                "x": x, "y": y, "width": w, "height": h,
                                "page": st.session_state.selected_page_index + 1
                            }
                        }
                        new_boxes.append(box)
                if len(new_boxes) != len(st.session_state.boxes):
                    st.session_state.boxes = new_boxes

            annotated_img = update_annotated_image(image, st.session_state.boxes, stroke_color, stroke_width, font)
            st.image(annotated_img, caption="Annotated Image", use_column_width=True)

            buffer = BytesIO()
            annotated_img.save(buffer, format="PNG")
            st.download_button(
                label="📥 Download Annotated Image",
                data=buffer.getvalue(),
                file_name="annotated_output.png",
                mime="image/png"
            )

        with text_col:
            st.subheader("📋 Text Result")
            for i, box in enumerate(st.session_state.boxes):
                with st.expander(f"📝 Field {i+1} ({box['type']})"):
                    new_type = st.selectbox(
                        "Field Type", ["text", "date", "email", "currency", "number", "label"],
                        index=["text", "date", "email", "currency", "number", "label"].index(box["type"]),
                        key=f"type_{i}"
                    )
                    new_title = st.text_input("Title", value=box["title"], key=f"title_{i}")
                    new_value = st.text_area("Value", value=box["value"], key=f"value_{i}")

                    if st.button("✅ Update", key=f"update_{i}"):
                        box["type"] = new_type
                        box["title"] = new_title
                        box["value"] = new_value
                        st.session_state.update_trigger = not st.session_state.update_trigger
                        st.experimental_rerun()

                    st.markdown("**Coordinates:**")
                    st.json(box["coordinates"])
                    if st.button("🗑 Delete", key=f"del_{i}"):
                        st.session_state.boxes.pop(i)
                        st.experimental_rerun()

            if st.session_state.json_table:
                st.subheader("🧾 Extracted Table JSON")
                st.json(st.session_state.json_table)

            # ✅ Display and Download JSON Output
            st.subheader("⬇️ Export All Data")
            full_data = {
                "fields": st.session_state.boxes,
                "table": st.session_state.json_table
            }

            # Show full JSON
            st.json(full_data)

            # Download button
            json_bytes = BytesIO(json.dumps(full_data, indent=2).encode("utf-8"))
            st.download_button(
                label="📥 Download JSON",
                data=json_bytes,
                file_name="extracted_data.json",
                mime="application/json"
            )
