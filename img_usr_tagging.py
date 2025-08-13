import os
import cv2
import json
import tempfile
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from rapidocr import RapidOCR
from rapid_table import ModelType, RapidTable, RapidTableInput
from streamlit_drawable_canvas import st_canvas

# Initialize session state for extracted tables
if "extracted_tables" not in st.session_state:
    st.session_state.extracted_tables = []

# Table Recognizer (OCR + Structure)
class Rapid:
    def __init__(self):
        self.ocr_engine = RapidOCR()
        input_args = RapidTableInput(
            model_type=ModelType.SLANETPLUS,
            engine_cfg={"use_cann": True, "cann_ep_cfg.gpu_id": 0}
        )
        self.table_engine = RapidTable(input_args)

    def recognize_table(self, image_np):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            cv2.imwrite(f.name, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            image_path = f.name

        ori_ocr_res = self.ocr_engine(image_path)
        ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
        results = self.table_engine(image_path, ocr_results=ocr_results)

        soup = BeautifulSoup(results.pred_html, "html.parser")
        table = soup.find("table")
        if not table:
            return None

        rows = table.find_all("tr")
        headers = [th.get_text(strip=True) for th in rows[0].find_all(["td", "th"])]
        table_rows = []
        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if len(cells) == len(headers):
                table_rows.append(dict(zip(headers, cells)))
        return {"headers": headers, "rows": table_rows}

def convert_cv2_to_bytes(cv2_img):
    img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

def draw_annotated_image(image_np, boxes_with_labels):
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for item in boxes_with_labels:
        box = item['bbox']
        label = item['label']
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1] - 15), label, fill="blue", font=font)

    return img

# Streamlit UI
st.set_page_config(layout="wide", page_title="📄 Manual Table Extractor with Zoom")
st.title("🖋️ Draw, Label, and Zoom Bounding Boxes to Extract Tables")

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
rapid = Rapid()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Clear tables for new upload
    st.session_state.extracted_tables = []

    st.sidebar.header("📝 Drawing & View Settings")
    drawing_mode = st.sidebar.selectbox("Drawing Mode", ["rect"])
    label_input = st.sidebar.text_input("Label for Current Box", "table")
    zoom_factor = st.sidebar.slider("🔍 Zoom", min_value=0.2, max_value=3.0, value=1.0, step=0.1)

    # Resize image for zoom
    zoomed_width = int(image.width * zoom_factor)
    zoomed_height = int(image.height * zoom_factor)
    image_zoomed = image.resize((zoomed_width, zoomed_height))
    image_np_zoomed = np.array(image_zoomed)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📐 Draw & Label Regions")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 255, 0.3)",
            stroke_width=2,
            background_image=image_zoomed,
            update_streamlit=True,
            height=zoomed_height,
            width=zoomed_width,
            drawing_mode=drawing_mode,
            key="canvas"
        )

    boxes_with_labels = []

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        for i, obj in enumerate(objects):
            if obj["type"] == "rect":
                left = int(obj["left"] / zoom_factor)
                top = int(obj["top"] / zoom_factor)
                width = int(obj["width"] / zoom_factor)
                height = int(obj["height"] / zoom_factor)

                x1 = max(0, left)
                y1 = max(0, top)
                x2 = min(image_np.shape[1], left + width)
                y2 = min(image_np.shape[0], top + height)

                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue

                label = obj.get("label", f"{label_input}_{i+1}")
                boxes_with_labels.append({
                    "label": label,
                    "bbox": [x1, y1, x2, y2]
                })

                # Extract table if labeled
                if "table" in label.lower():
                    cropped_np = image_np[y1:y2, x1:x2]
                    if cropped_np.size > 0:
                        table_data = rapid.recognize_table(cropped_np)
                        if table_data:
                            st.session_state.extracted_tables.append(table_data)
                        else:
                            st.warning(f"⚠️ No table detected in box {i + 1}.")
                    else:
                        st.warning(f"⚠️ Empty crop in box {i + 1}, skipped.")

    with col2:
        st.subheader("📋 Extracted Table Data")

        if not st.session_state.extracted_tables:
            st.info("🖼️ Draw and label boxes as 'table' to extract table data.")
        else:
            selected_idx = st.selectbox(
                "Select Table", 
                range(1, len(st.session_state.extracted_tables) + 1), 
                format_func=lambda x: f"Table {x}"
            )
            selected_table = st.session_state.extracted_tables[selected_idx - 1]

            view = st.radio("View Format", ["Row-wise JSON", "Column-wise JSON", "Full Table"], horizontal=True)
            headers = selected_table["headers"]
            rows = selected_table["rows"]

            if view == "Row-wise JSON":
                st.json(rows)
            elif view == "Column-wise JSON":
                cols = {header: [row.get(header, "") for row in rows] for header in headers}
                st.json(cols)
            else:
                st.json(selected_table)

            st.download_button(
                label="💾 Download Table JSON",
                data=json.dumps(selected_table, indent=2, ensure_ascii=False),
                file_name=f"table_{selected_idx}.json",
                mime="application/json"
            )

    if boxes_with_labels:
        st.subheader("🖼️ Annotated Image with Labels")
        annotated_img = draw_annotated_image(image_np, boxes_with_labels)
        annotated_bytes = convert_cv2_to_bytes(np.array(annotated_img))
        st.image(annotated_bytes, use_column_width=True)

        st.download_button(
            label="📥 Download Annotated Image",
            data=annotated_bytes,
            file_name="annotated_image.png",
            mime="image/png"
        )
