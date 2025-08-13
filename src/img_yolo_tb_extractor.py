import os
import re
import cv2
import json
import tempfile
import streamlit as st
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from pathlib import Path
from huggingface_hub import hf_hub_download
from streamlit_drawable_canvas import st_canvas

from rapidocr import RapidOCR
from rapid_table import ModelType, RapidTable, RapidTableInput
from doclayout_yolo import YOLOv10


def sanitize_text(text: str) -> str:
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


class TableExtractor:
    def __init__(self):
        ocr_engine = RapidOCR()
        config = RapidTableInput(
            model_type=ModelType.SLANETPLUS,
            engine_cfg={"use_cann": False}
        )
        self.ocr_engine = ocr_engine
        self.table_engine = RapidTable(config)

    def extract_table(self, image_path: str):
        ocr_result = self.ocr_engine(image_path)

        # ✅ Safely check array content
        if (
            ocr_result.boxes is None or len(ocr_result.boxes) == 0 or
            ocr_result.txts is None or len(ocr_result.txts) == 0 or
            ocr_result.scores is None or len(ocr_result.scores) == 0
        ):
            return None

        ocr_data = [ocr_result.boxes, ocr_result.txts, ocr_result.scores]

        try:
            table_output = self.table_engine(image_path, ocr_results=ocr_data)
            soup = BeautifulSoup(table_output.pred_html, "html.parser")
            table = soup.find("table")
        except Exception as e:
            print(f"Table extraction failed: {e}")
            return None

        if not table:
            return None

        headers = [cell.get_text(strip=True) for cell in table.find_all("tr")[0].find_all(["td", "th"])]
        rows_data = []

        for row in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if len(cells) == len(headers):
                rows_data.append(dict(zip(headers, cells)))

        return {"headers": headers, "rows": rows_data}


class LayoutAnalyzer:
    LABELS = {
        0: "title", 1: "plain_text", 2: "abandon", 3: "figure", 4: "figure_caption",
        5: "table", 6: "table_caption", 7: "table_footnote", 8: "isolate_formula", 9: "formula_caption"
    }

    def __init__(self):
        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt"
        )
        self.detector = YOLOv10(model_path)

    def detect_tables(self, image_path: str, threshold=0.3):
        results = self.detector.predict(image_path, imgsz=1024, conf=threshold, device="cpu")
        img = cv2.imread(image_path)
        tables = []

        for box in results[0].boxes:
            label_id = int(box.cls[0])
            label = self.LABELS.get(label_id, "unknown")

            if label == "table":
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_table = img[y1:y2, x1:x2]

                tables.append({
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "crop_img": cropped_table
                })

        return tables, img


def format_output(rows_data, headers, output_type):
    if output_type == "Row-wise":
        return {f"row_{i+1}": row for i, row in enumerate(rows_data)}
    elif output_type == "Column-wise":
        column_data = {header: [] for header in headers}
        for row in rows_data:
            for key, value in row.items():
                column_data[key].append(value)
        return column_data
    elif output_type == "Table":
        return rows_data
    return {}


def main():
    st.set_page_config(layout="wide")
    st.title("📄 Table Extractor from Image (Draw or YOLO)")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    detection_mode = st.radio("Choose Table Selection Mode:", ["Draw manually", "Auto detect (YOLO)"])
    output_type = st.selectbox("Select Output Format", ["Row-wise", "Column-wise", "Table"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_file.read())
            img_path = tmp.name

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        table_extractor = TableExtractor()
        all_rows = []
        headers_set = set()

        if detection_mode == "Draw manually":
            st.subheader("🖍️ Draw a rectangle around each table")
            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.3)",
                stroke_width=3,
                background_image=image_pil,
                update_streamlit=True,
                height=image.shape[0],
                width=image.shape[1],
                drawing_mode="rect",
                key="canvas"
            )

            if canvas_result.json_data and canvas_result.json_data["objects"]:
                for obj in canvas_result.json_data["objects"]:
                    left = int(obj["left"])
                    top = int(obj["top"])
                    width = int(obj["width"])
                    height = int(obj["height"])
                    x1, y1, x2, y2 = left, top, left + width, top + height

                    cropped = image[y1:y2, x1:x2]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as crop_tmp:
                        cv2.imwrite(crop_tmp.name, cropped)
                        table_data = table_extractor.extract_table(crop_tmp.name)

                    if table_data:
                        all_rows.extend(table_data["rows"])
                        headers_set.update(table_data["headers"])

        elif detection_mode == "Auto detect (YOLO)":
            layout_analyzer = LayoutAnalyzer()
            tables, _ = layout_analyzer.detect_tables(img_path)

            for table in tables:
                if table["confidence"] < 0.7:
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as crop_tmp:
                    cv2.imwrite(crop_tmp.name, table["crop_img"])
                    table_data = table_extractor.extract_table(crop_tmp.name)

                if table_data:
                    all_rows.extend(table_data["rows"])
                    headers_set.update(table_data["headers"])

        if all_rows:
            output_json = format_output(all_rows, list(headers_set), output_type)

            st.subheader(f"📋 Extracted Table Data ({output_type})")
            st.json(output_json)

            json_str = json.dumps(output_json, indent=4)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{output_type.lower().replace(' ', '_')}_table_data.json",
                mime="application/json"
            )
        else:
            st.warning("⚠️ No tables detected or extracted.")


if __name__ == "__main__":
    main()
