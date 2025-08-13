import streamlit as st
from streamlit_drawable_canvas import st_canvas
from pdf2image import convert_from_path
from PIL import Image
import json
import tempfile
import os

st.set_page_config(layout="wide")
st.title("📄 Document Template Annotator")

# --- 1. Upload PDF or Image ---
uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    # Convert PDF to image (first page only)
    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            images = convert_from_path(temp_pdf.name, dpi=150)
            image = images[0]
    else:
        image = Image.open(uploaded_file)

    # --- 2. Select Label ---
    label = st.text_input("Label for new box", value="")

    # --- 3. Draw Canvas ---
    st.subheader("Annotate Document")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 255, 0.3)",
        stroke_width=2,
        stroke_color="#0000FF",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="rect",
        key="canvas",
    )

    # --- 4. Capture Annotations ---
    if canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        boxes = []
        for obj in objects:
            if obj["type"] == "rect":
                box = {
                    "label": label or obj.get("name", "Unnamed"),
                    "position": {
                        "x": int(obj["left"]),
                        "y": int(obj["top"])
                    },
                    "size": {
                        "width": int(obj["width"]),
                        "height": int(obj["height"])
                    }
                }
                boxes.append(box)

        # --- 5. Display Extracted Fields ---
        st.subheader("📋 Extracted Fields")
        for box in boxes:
            st.markdown(f"""
            **{box['label']}**
            - Position: ({box['position']['x']}, {box['position']['y']})
            - Size: {box['size']['width']} x {box['size']['height']}
            """)

        # --- 6. Export Button ---
        st.download_button(
            "📥 Export Data",
            data=json.dumps(boxes, indent=2),
            file_name="template.json",
            mime="application/json"
        )
